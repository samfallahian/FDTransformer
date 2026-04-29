import os
import torch
import h5py
import numpy as np
import sys
import pandas as pd
import pysindy as ps
from sklearn.metrics import mean_squared_error

from pysindy_config import (
    configure_project_imports,
    load_config_from_args,
    make_parser,
    output_path,
    resolve_path,
    select_device,
)


def load_models(config):
    project_root = configure_project_imports(config)
    from transformer.transformer_model_v1 import OrderedTransformerV1

    transformer_checkpoint = resolve_path(config, ("models", "transformer_checkpoint"), required=True)
    encoder_checkpoint = resolve_path(config, ("models", "encoder_checkpoint"), required=True)
    device = select_device(torch, config["runtime"].get("device"))
    
    # Patch for torch._dynamo compatibility
    try:
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class Dummy: pass
            torch._dynamo.convert_frame.ConvertFrameBox = Dummy
    except: pass

    # Add transformer directory to sys.path
    TRANS_DIR = os.path.join(project_root, "transformer")
    if TRANS_DIR not in sys.path:
        sys.path.insert(0, TRANS_DIR)
    
    # Define a dummy Config class in __main__ to satisfy the unpickler
    class Config:
        pass
    import __main__
    __main__.Config = Config

    checkpoint = torch.load(transformer_checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        transformer = checkpoint['model']
        if hasattr(transformer, '_orig_mod'): transformer = transformer._orig_mod
    else:
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()

    ae = torch.jit.load(encoder_checkpoint, map_location=device)
    ae.eval()
    
    return transformer, ae

def get_vorticity_enstrophy(V, x, y, z):
    # V: (nx, ny, nz, 3)
    grad_u = np.gradient(V[..., 0], x, y, z)
    grad_v = np.gradient(V[..., 1], x, y, z)
    grad_w = np.gradient(V[..., 2], x, y, z)
    
    # wx = dw/dy - dv/dz
    # wy = du/dz - dw/dx
    # wz = dv/dx - du/dy
    wx = grad_w[1] - grad_v[2]
    wy = grad_u[2] - grad_w[0]
    wz = grad_v[0] - grad_u[1]
    enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
    return wx, wy, wz, enstrophy

def run_sindy_recovery(wx, wy, wz, enstrophy):
    X = np.stack([wx.flatten(), wy.flatten(), wz.flatten()], axis=-1)
    y = enstrophy.flatten().reshape(-1, 1)
    
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-12)
    
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['wx', 'wy', 'wz'])
    
    optimizer.fit(X_poly, y)
    coefs = np.asarray(optimizer.coef_[0])
    
    y_pred = np.asarray(X_poly) @ coefs.T
    mse = mean_squared_error(y, y_pred)
    
    results = {'MSE': mse}
    for i, name in enumerate(poly_names):
        results[name] = coefs[i]
    return results

def main(config):
    configure_project_imports(config)
    from helpers.TransformLatent import FloatConverter

    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)
    output_csv = output_path(config, "all_params_recovery_results")
    trend_figure = output_path(config, "all_params_coefficient_trends")
    transformer, ae = load_models(config)
    converter = FloatConverter()
    triplet_idx = config["runtime"]["triplet_idx"]
    batch_size = config["runtime"]["batch_size"]
    n_search = config["runtime"]["n_search"]
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        originals = f['originals']
        
        # Get unique params from first 100k samples for speed
        params_sample = data[:n_search, 0, 0, 51]
        unique_params = np.unique(params_sample)
        print(f"Unique Reynolds Numbers to evaluate: {unique_params}")
        
        all_results = []
        
        for p_target in unique_params:
            print(f"\nProcessing Reynolds Number: {p_target:.2f}")
            
            # Find indices for this param
            mask = np.abs(params_sample - p_target) < 0.01
            indices_for_param = np.where(mask)[0]
            
            # Map YZ to grid
            coords = data[indices_for_param, 0, 0, 48:50]
            u_yz, u_idx = np.unique(coords, axis=0, return_index=True)
            indices_for_grid = indices_for_param[u_idx]
            
            unique_y = np.sort(np.unique(u_yz[:, 0]))
            unique_z = np.sort(np.unique(u_yz[:, 1]))
            ny, nz = len(unique_y), len(unique_z)
            nx = 26
            
            print(f"Grid size: {nx}x{ny}x{nz}")
            
            y_map = {val: i for i, val in enumerate(unique_y)}
            z_map = {val: i for i, val in enumerate(unique_z)}
            
            V_raw = np.zeros((nx, ny, nz, 3))
            V_enc = np.zeros((nx, ny, nz, 3))
            V_pred = np.zeros((nx, ny, nz, 3))
            
            # Batched prediction for Predicted, Raw, and Encoded sources
            print(f"Running batched reconstruction for {p_target:.2f}...")
            device = select_device(torch, config["runtime"].get("device"))
            transformer.to(device)
            ae.to(device)
            
            # We must use sorted indices for HDF5 access
            sorted_indices = np.sort(indices_for_grid)
            
            for i in range(0, len(sorted_indices), batch_size):
                batch_idx = sorted_indices[i:i+batch_size]
                curr_B = len(batch_idx)
                
                # Get samples (Increasing order required)
                samples = data[batch_idx.tolist()] # (B, 8, 26, 52)
                origs = originals[batch_idx.tolist()] # (B, 26, 3)
                samples_flat = samples.reshape(curr_B, -1, 52)
                
                # 1. Fill Raw
                for b_i in range(curr_B):
                    idx = batch_idx[b_i]
                    # We need the y_val, z_val for each idx in the batch
                    # They are in samples[b_i, 0, 0, 48:50]
                    y_val = samples[b_i, 0, 0, 48]
                    z_val = samples[b_i, 0, 0, 49]
                    iy, iz = y_map[y_val], z_map[z_val]
                    V_raw[:, iy, iz, :] = origs[b_i]
                
                # 2. Fill Encoded
                t8_latents_gt = torch.from_numpy(samples[:, -1, :, :47]).float().to(device)
                with torch.no_grad():
                    decoded_gt = ae.decode(t8_latents_gt.reshape(-1, 47))
                    v_63_gt = decoded_gt.reshape(curr_B, 26, 125, 3)[:, :, triplet_idx, :].cpu().numpy()
                    for b_i in range(curr_B):
                        y_val = samples[b_i, 0, 0, 48]
                        z_val = samples[b_i, 0, 0, 49]
                        iy, iz = y_map[y_val], z_map[z_val]
                        V_enc[:, iy, iz, :] = converter.unconvert(v_63_gt[b_i])

                # 3. Fill Predicted
                inputs = torch.from_numpy(samples_flat[:, :182, :]).float().to(device)
                with torch.no_grad():
                    current_seq = inputs.clone()
                    t8_preds = []
                    for step in range(182, 208):
                        step_out = transformer(current_seq)
                        next_latent = step_out[:, -1, :] # (B, 47)
                        t8_preds.append(next_latent.unsqueeze(1))
                        
                        if step < 207:
                            new_tokens = torch.from_numpy(samples_flat[:, step:step+1, :]).float().to(device)
                            new_tokens[:, 0, :47] = next_latent
                            current_seq = torch.cat([current_seq, new_tokens], dim=1)
                    
                    all_t8_latents = torch.cat(t8_preds, dim=1) # (B, 26, 47)
                    all_t8_latents_flat = all_t8_latents.reshape(-1, 47)
                    decoded = ae.decode(all_t8_latents_flat) # (B*26, 375)
                    v_63 = decoded.reshape(curr_B, 26, 125, 3)[:, :, triplet_idx, :].cpu().numpy()
                    
                    for b_i in range(curr_B):
                        y_val = samples[b_i, 0, 0, 48]
                        z_val = samples[b_i, 0, 0, 49]
                        iy, iz = y_map[y_val], z_map[z_val]
                        V_pred[:, iy, iz, :] = converter.unconvert(v_63[b_i])

            # Now V_raw, V_enc, V_pred are filled
            
            # Coordinates
            x_coords = data[0, 0, :, 47]
            
            # Run SINDy for each source
            sources = {'Raw': V_raw, 'Encoded': V_enc, 'Predicted': V_pred}
            for label, V in sources.items():
                wx, wy, wz, enst = get_vorticity_enstrophy(V, x_coords, unique_y, unique_z)
                res = run_sindy_recovery(wx, wy, wz, enst)
                res['Reynolds_Number'] = p_target
                res['Source'] = label
                all_results.append(res)

        # Save and Print results
        df = pd.DataFrame(all_results)
        cols = ['Reynolds_Number', 'Source', 'MSE', 'wx^2', 'wy^2', 'wz^2', 'wx', 'wy', 'wz', '1']
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(output_csv, index=False)
        print("\nSummary Results Table:")
        print(df.to_string(index=False))

        # Visualizations
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        for i, term in enumerate(['wx^2', 'wy^2', 'wz^2']):
            plt.subplot(3, 1, i+1)
            for source in ['Raw', 'Encoded', 'Predicted']:
                subset = df[df['Source'] == source]
                plt.plot(subset['Reynolds_Number'], subset[term], marker='o', label=f'{source} {term}')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            plt.title(f'Recovered Coefficient for {term} across Reynolds Numbers')
            plt.ylabel('Coefficient Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(trend_figure)
        print(f"\nTrend figure saved to: {trend_figure}")

if __name__ == "__main__":
    parser = make_parser("Reproduce SINDy recovery over every Reynolds parameter in the HDF5 data.")
    args = parser.parse_args()
    main(load_config_from_args(args))
