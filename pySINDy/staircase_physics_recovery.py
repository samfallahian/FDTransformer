import os
import torch
import h5py
import numpy as np
import pandas as pd
import pysindy as ps
import sys
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# PROJECT SETUP
# =============================================================================
# We need to make sure the project root is in the system path so we can import
# our custom modules like FloatConverter and the Transformer model.
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TransformLatent import FloatConverter
from transformer.transformer_model_v1 import OrderedTransformerV1

def load_models():
    """
    Loads the pre-trained Transformer and Autoencoder models.
    The Transformer predicts future latent states, and the Autoencoder's decoder
    converts those latents back into physical velocity fields.
    """
    TRANSFORMER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/transformer/best_ordered_transformer_v1.pt"
    ENCODER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
    
    # Detect and use the best available device (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"--- Loading models onto device: {DEVICE} ---")
    
    # This is a technical patch to handle some internal PyTorch compatibility issues
    # with the 'dynamo' optimization engine during model loading.
    try:
        import torch._dynamo.convert_frame as t_cf
        if not hasattr(t_cf, 'ConvertFrameBox'):
            class Dummy: pass
            t_cf.ConvertFrameBox = Dummy
    except Exception:
        pass

    # Ensure the transformer directory is available for imports
    TRANS_DIR = os.path.join(PROJECT_ROOT, "transformer")
    if TRANS_DIR not in sys.path:
        sys.path.insert(0, TRANS_DIR)
    
    # We define a dummy Config class to match the structure expected by the 
    # pickled transformer checkpoint.
    class Config:
        pass
    import __main__
    __main__.Config = Config

    # Load the Transformer
    print(f"Loading Transformer from: {TRANSFORMER_CHECKPOINT}")
    checkpoint = torch.load(TRANSFORMER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        transformer = checkpoint['model']
        if hasattr(transformer, '_orig_mod'): 
            transformer = transformer._orig_mod
    else:
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()
    transformer.to(DEVICE)

    # Load the Autoencoder (TorchScript format)
    print(f"Loading Autoencoder from: {ENCODER_CHECKPOINT}")
    ae = torch.jit.load(ENCODER_CHECKPOINT, map_location=DEVICE)
    ae.eval()
    ae.to(DEVICE)
    
    print("Models loaded successfully.\n")
    return transformer, ae, DEVICE

# =============================================================================
# PHYSICS CALCULATIONS
# =============================================================================
def calculate_vorticity(V, x, y, z):
    """
    Computes the vorticity field (curl of velocity) from a 3D velocity grid.
    Vorticity represents the local rotation of the fluid.
    """
    # Calculate gradients for each velocity component (u, v, w) across x, y, z
    grad_u = np.gradient(V[..., 0], x, y, z)
    grad_v = np.gradient(V[..., 1], x, y, z)
    grad_w = np.gradient(V[..., 2], x, y, z)
    
    # Curl formula: w = nabla x V
    wx = grad_w[1] - grad_v[2]
    wy = grad_u[2] - grad_w[0]
    wz = grad_v[0] - grad_u[1]
    return wx, wy, wz

def run_sindy_ke(u, v, w, ke):
    """
    Uses SINDy to find the algebraic relationship for Kinetic Energy.
    Expected: KE = 0.5 * (u^2 + v^2 + w^2)
    """
    X = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=-1)
    y = ke.flatten().reshape(-1, 1)
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-10)
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['u', 'v', 'w'])
    optimizer.fit(X_poly, y)
    coefs = np.asarray(optimizer.coef_[0])
    results = {f"KE_{n}": coefs[i] for i, n in enumerate(poly_names)}
    y_pred = np.asarray(X_poly) @ coefs.T
    results['KE_MSE'] = np.mean((y.flatten() - y_pred)**2)
    return results

def run_sindy_helicity(u, v, w, wx, wy, wz, helicity):
    """
    Uses SINDy to find the algebraic relationship for Helicity.
    Expected: Helicity = u*wx + v*wy + w*wz
    """
    X = np.stack([u.flatten(), v.flatten(), w.flatten(), wx.flatten(), wy.flatten(), wz.flatten()], axis=-1)
    y = helicity.flatten().reshape(-1, 1)
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-10)
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['u', 'v', 'w', 'wx', 'wy', 'wz'])
    optimizer.fit(X_poly, y)
    coefs = np.asarray(optimizer.coef_[0])
    results = {f"Helicity_{n}": coefs[i] for i, n in enumerate(poly_names)}
    y_pred = np.asarray(X_poly) @ coefs.T
    results['Helicity_MSE'] = np.mean((y.flatten() - y_pred)**2)
    return results

def run_sindy_enstrophy(wx, wy, wz, enstrophy):
    """
    Uses SINDy to find the algebraic relationship for Enstrophy.
    Expected: Enstrophy = 0.5 * (wx^2 + wy^2 + wz^2)
    """
    X = np.stack([wx.flatten(), wy.flatten(), wz.flatten()], axis=-1)
    y = enstrophy.flatten().reshape(-1, 1)
    library = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-10)
    library.fit(X)
    X_poly = library.transform(X)
    poly_names = library.get_feature_names(['wx', 'wy', 'wz'])
    optimizer.fit(X_poly, y)
    coefs = np.asarray(optimizer.coef_[0])
    results = {f"Enstrophy_{n}": coefs[i] for i, n in enumerate(poly_names)}
    y_pred = np.asarray(X_poly) @ coefs.T
    results['Enstrophy_MSE'] = np.mean((y.flatten() - y_pred)**2)
    return results

# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================
def main():
    print("--- Starting Staircase Physics Evaluation ---")
    h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
    
    # 1. Load models and setup helper objects
    transformer, ae, device = load_models()
    converter = FloatConverter()
    TRIPLET_IDX = 62  # Index of the center velocity vector in the reconstructed 5x5x5 cube
    
    # Parameters for the evaluation
    REYNOLDS_NUMBERS = [3.6, 4.6, 5.2, 6.4, 6.6, 7.2, 7.8, 8.4, 10.4, 11.4]
    TEMPORAL_CONTEXTS = [7, 6, 5, 4, 3, 2, 1] # Providing T1 through T_ctx as ground truth history
    
    CSV_PATH = os.path.join(PROJECT_ROOT, 'Documentation/staircase_physics_results.csv')
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    all_results = []
    
    # 2. Handle Restarts: Load existing results from CSV if it exists
    if os.path.exists(CSV_PATH):
        try:
            print(f"Reading existing results from {CSV_PATH} to resume work...")
            df_existing = pd.read_csv(CSV_PATH)
            all_results = df_existing.to_dict('records')
            print(f"Found {len(all_results)} existing records. Skipping completed tasks.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}. Starting fresh.")

    # 3. Open HDF5 and prepare for data extraction
    print(f"Opening HDF5 dataset: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        data_ds = f['data']
        print("Extracting Reynolds number parameters from the entire dataset (1M samples)...")
        # Since we don't have a separate index for the Reynolds number, we must
        # scan the metadata field (index 51) across all samples to find matches.
        full_params = data_ds[:, 0, 0, 51]
        
        # 4. Loop through each Reynolds Number (Flow Condition)
        for reynolds in tqdm(REYNOLDS_NUMBERS, desc="Overall Progress (Reynolds)"):
            
            # Check if this Reynolds number has already been fully processed
            contexts_done = [r['TemporalContext'] for r in all_results if abs(r['Reynolds'] - reynolds) < 0.01]
            if set(TEMPORAL_CONTEXTS).issubset(set(contexts_done)):
                # print(f"Skipping Reynolds {reynolds} (already complete)")
                continue

            print(f"\n>>> PROCESSING REYNOLDS: {reynolds} <<<")
            
            # Filter the dataset for samples matching this Reynolds number
            mask = np.abs(full_params - reynolds) < 0.05
            indices = np.where(mask)[0]
            print(f"Found {len(indices)} samples for Re={reynolds}. Selecting 1000 for grid identification...")
            
            if len(indices) > 1000:
                np.random.seed(42)
                indices = np.random.choice(indices, 1000, replace=False)
            indices = np.sort(indices) 
            
            # 5. Identify the spatial grid (Y, Z coordinates) within the selected samples
            coords = data_ds[indices, 0, 0, 48:50] # Y, Z are at indices 48, 49
            unique_yz, u_idx = np.unique(coords, axis=0, return_index=True)
            
            # We limit the grid size to 30 unique points for processing speed
            if len(u_idx) > 30:
                u_idx = u_idx[:30]
            
            grid_indices = indices[u_idx]
            grid_coords = coords[u_idx]
            unique_y = np.sort(np.unique(grid_coords[:, 0]))
            unique_z = np.sort(np.unique(grid_coords[:, 1]))
            
            print(f"Using {len(grid_indices)} grid points. Spatial extent: {len(unique_y)} Y-planes x {len(unique_z)} Z-planes.")
            print("Explanation: The 'Spatial extent' represents the number of unique Y and Z coordinate planes ")
            print("covered by the selected 30 grid points. This defines the dimensions of the reconstructed ")
            print("3D velocity volume (nx, ny, nz) used for physics calculations.")
            
            # Setup coordinate maps for grid reconstruction
            ny, nz = len(unique_y), len(unique_z)
            nx = 26 # Constant X resolution in our dataset
            y_map = {val: i for i, val in enumerate(unique_y)}
            z_map = {val: i for i, val in enumerate(unique_z)}
            x_coords = data_ds[0, 0, :, 47]
            
            # 6. Loop through Temporal Context Levels (The "Staircase")
            for t_ctx in TEMPORAL_CONTEXTS:
                # Skip this specific context if already in results
                if any(abs(r['Reynolds'] - reynolds) < 0.01 and r['TemporalContext'] == t_ctx for r in all_results):
                    continue
                
                print(f"--- Predicting T8 using {t_ctx} Ground-Truth Timesteps as history ---")
                V = np.zeros((nx, ny, nz, 3)) # Placeholder for the reconstructed velocity volume
                V_gt = np.zeros((nx, ny, nz, 3)) # Ground truth T8 volume
                
                # 7. Autoregressive Prediction for each spatial point in the grid
                for idx in tqdm(grid_indices, desc=f"Predicting Grid (Re={reynolds}, T={t_ctx})", leave=True):
                    y_val = data_ds[idx, 0, 0, 48]
                    z_val = data_ds[idx, 0, 0, 49]
                    iy, iz = y_map[y_val], z_map[z_val]
                    
                    sample_data = data_ds[idx] # Full 8-timestep sample (8, 26, 52)
                    sample_flat = sample_data.reshape(-1, 52)
                    
                    # Store Ground Truth for RMSE calculation (Timestep 8, indices 182 to 207)
                    t8_gt_latent = sample_flat[182:208, :47]
                    decoded_gt = ae.decode(torch.from_numpy(t8_gt_latent).float().to(device))
                    v_gt_full = decoded_gt.reshape(26, 125, 3)
                    v_gt_center = v_gt_full[:, TRIPLET_IDX, :].cpu().numpy()
                    V_gt[:, iy, iz, :] = converter.unconvert(v_gt_center)
                    
                    # Provide history tokens (T1 to T_ctx)
                    ctx_len = t_ctx * 26
                    inputs = torch.from_numpy(sample_flat[:ctx_len]).float().unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        current_seq = inputs.clone()
                        # Predict point-by-point until the end of T8 (total 208 tokens)
                        for step in range(ctx_len, 208):
                            step_out = transformer(current_seq)
                            next_latent = step_out[:, -1, :] # Model predicts next state
                            
                            # Prepare the next token in the sequence using original coordinates/metadata
                            new_token = torch.from_numpy(sample_flat[step:step+1]).float().unsqueeze(0).to(device)
                            new_token[0, 0, :47] = next_latent # Overwrite with predicted latent features
                            current_seq = torch.cat([current_seq, new_token], dim=1)
                        
                        # Extract the predicted tokens for Timestep 8 (indices 182 to 207)
                        t8_latents = current_seq[0, 182:208, :47]
                        
                        # Decode these latents back into a local velocity cube
                        decoded = ae.decode(t8_latents) 
                        v_full = decoded.reshape(26, 125, 3)
                        v_center = v_full[:, TRIPLET_IDX, :].cpu().numpy()
                        # Denormalize velocity values
                        V[:, iy, iz, :] = converter.unconvert(v_center)
                
                # 8. Physical Property Calculation on the predicted volume
                print(f"Calculating physics on predicted field (Re={reynolds}, T={t_ctx})...")
                u, v, w = V[..., 0], V[..., 1], V[..., 2]
                wx, wy, wz = calculate_vorticity(V, x_coords, unique_y, unique_z)
                ke = 0.5 * (u**2 + v**2 + w**2)
                helicity = u*wx + v*wy + w*wz
                enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
                
                # 9. SINDy Formula Recovery
                print(f"Running SINDy regressions for Re={reynolds}, T={t_ctx}...")
                ke_res = run_sindy_ke(u, v, w, ke)
                hel_res = run_sindy_helicity(u, v, w, wx, wy, wz, helicity)
                ens_res = run_sindy_enstrophy(wx, wy, wz, enstrophy)
                
                # 10. Calculate T8 Prediction RMSE
                t8_rmse = np.sqrt(np.mean((V - V_gt)**2))
                
                # Package results
                row = {
                    'Reynolds': reynolds,
                    'TemporalContext': t_ctx,
                    'T8_RMSE': t8_rmse
                }
                row.update(ke_res)
                row.update(hel_res)
                row.update(ens_res)
                all_results.append(row)
                
                # Intermediate save to ensure we don't lose work
                pd.DataFrame(all_results).to_csv(CSV_PATH, index=False)
                print(f"Step complete. Results saved to {CSV_PATH}")

    # 10. Final Completion and Plotting
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n--- Evaluation Finished! ---")
    print(f"Final results table: {CSV_PATH}")
    
    plot_results(df)

def plot_results(df):
    """
    Generates summary plots showing how SINDy MSE and Prediction RMSE change 
    with Reynolds number and temporal context.
    """
    print("Generating visualization...")
    metrics = ['T8_RMSE', 'KE_MSE', 'Helicity_MSE', 'Enstrophy_MSE']
    reynolds_list = df['Reynolds'].unique()
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for reynolds in reynolds_list:
            subset = df[df['Reynolds'] == reynolds].sort_values('TemporalContext')
            ax.plot(subset['TemporalContext'], subset[metric], marker='o', label=f"Re={reynolds}")
        
        if metric != 'T8_RMSE':
            ax.set_yscale('log')
        
        ax.set_title(f'{metric} vs Temporal Context (Timesteps)')
        ax.set_xlabel('Number of Ground Truth Timesteps provided as context')
        ax.set_ylabel('Metric Value (Log Scale)' if metric != 'T8_RMSE' else 'Metric Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    PLOT_PATH = os.path.join(PROJECT_ROOT, 'Documentation/staircase_physics_trends.png')
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    main()
