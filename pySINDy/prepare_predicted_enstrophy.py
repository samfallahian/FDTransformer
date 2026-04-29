import os
import torch
import h5py
import numpy as np
import sys

from pysindy_config import (
    configure_project_imports,
    load_config_from_args,
    make_parser,
    output_path,
    resolve_path,
)


def load_models(config):
    configure_project_imports(config)
    from transformer.transformer_model_v1 import OrderedTransformerV1

    transformer_checkpoint = resolve_path(config, ("models", "transformer_checkpoint"), required=True)
    encoder_checkpoint = resolve_path(config, ("models", "encoder_checkpoint"), required=True)
    device = config["runtime"]["device"]
    
    # Patch for torch._dynamo compatibility
    try:
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class Dummy: pass
            torch._dynamo.convert_frame.ConvertFrameBox = Dummy
    except: pass

    # Add transformer directory to sys.path
    project_root = configure_project_imports(config)
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


def prepare_predicted_enstrophy(config):
    configure_project_imports(config)
    from helpers.TransformLatent import FloatConverter

    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)
    out_path = output_path(config, "predicted_grad")
    transformer, ae = load_models(config)
    converter = FloatConverter()
    triplet_idx = config["runtime"]["triplet_idx"]
    n_search = config["runtime"]["n_search"]
    p_target = config["runtime"]["p_target"]
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        
        params = data[:n_search, 0, 0, 51]
        mask = np.abs(params - p_target) < 0.01
        indices = np.where(mask)[0]
        
        coords = data[indices, 0, 0, 48:50]
        u_yz, u_idx = np.unique(coords, axis=0, return_index=True)
        indices = indices[u_idx]
        
        unique_y = np.sort(np.unique(u_yz[:, 0]))
        unique_z = np.sort(np.unique(u_yz[:, 1]))
        ny, nz = len(unique_y), len(unique_z)
        nx = 26
        
        V_pred = np.zeros((nx, ny, nz, 3))
        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}
        
        for idx in indices:
            y_val = data[idx, 0, 0, 48]
            z_val = data[idx, 0, 0, 49]
            iy, iz = y_map[y_val], z_map[z_val]
            
            # Predict 8th timestep
            # Input is first 7 timesteps: (7 * 26) points = 182 tokens
            sample_data = data[idx] # (8, 26, 52)
            sample_flat = sample_data.reshape(-1, 52)
            inputs = torch.from_numpy(sample_flat[:182]).float().unsqueeze(0) # (1, 182, 52)
            
            with torch.no_grad():
                # Transformer output for all input tokens
                outputs = transformer(inputs[:, :-1, :]) # (1, 181, 47)
                # We need the prediction for token 182, 183, ..., 207
                # Autoregressive prediction
                current_seq = inputs.clone()
                t8_preds = []
                for step in range(182, 208):
                    step_out = transformer(current_seq)
                    next_latent = step_out[:, -1, :] # (1, 47)
                    t8_preds.append(next_latent)
                    
                    if step < 207:
                        new_token = torch.from_numpy(sample_flat[step:step+1]).float().unsqueeze(0)
                        new_token[0, 0, :47] = next_latent
                        current_seq = torch.cat([current_seq, new_token], dim=1)
                
                t8_latents = torch.cat(t8_preds, dim=0) # (26, 47)
                decoded = ae.decode(t8_latents) # (26, 375)
                v_full = decoded.reshape(26, 125, 3)
                v_63 = v_full[:, triplet_idx, :].numpy()
                V_pred[:, iy, iz, :] = converter.unconvert(v_63)

        x, y, z = data[0, 0, :, 47], unique_y, unique_z
        grad_u = np.gradient(V_pred[..., 0], x, y, z)
        grad_v = np.gradient(V_pred[..., 1], x, y, z)
        grad_w = np.gradient(V_pred[..., 2], x, y, z)
        
        wx, wy, wz = grad_w[1]-grad_v[2], grad_u[2]-grad_w[0], grad_v[0]-grad_u[1]
        enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
        
        np.savez(out_path, V=V_pred, wx=wx, wy=wy, wz=wz, enstrophy=enstrophy)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = make_parser("Prepare predicted velocity gradients and enstrophy for SINDy.")
    args = parser.parse_args()
    prepare_predicted_enstrophy(load_config_from_args(args))
