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
        # Fix for NotImplementedError: The operator 'aten::native_dropout' is not currently implemented for the MPS device
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
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
    
    # Divergence: div V = nabla . V
    div_V = grad_u[0] + grad_v[1] + grad_w[2]
    
    return wx, wy, wz, div_V

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
    print("--- Starting Interpolation Physics Evaluation ---")
    h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
    
    # 1. Load models and setup helper objects
    transformer, ae, device = load_models()
    converter = FloatConverter()
    TRIPLET_IDX = 62  # Index of the center velocity vector in the reconstructed 5x5x5 cube
    
    # Parameters for the evaluation
    REYNOLDS_NUMBERS = [3.6, 4.6, 5.2, 6.4, 6.6, 7.2, 7.8, 8.4, 10.4, 11.4]
    
    # Define Transitions: (HistoryEnd, TargetStep, JumpType)
    # T1->T2 is history length 1 (tokens 0-26), predicting tokens 26-52 (T2)
    TRANSITIONS = []
    # 1. Ti -> Ti+1 (Jump 1)
    for i in range(1, 8):
        TRANSITIONS.append((i, i + 1, "Jump 1"))
    # 2. Ti -> Ti+2 (Jump 2)
    for i in range(1, 7):
        TRANSITIONS.append((i, i + 2, "Jump 2"))
    # 3. Ti -> Ti+3 (Jump 3)
    for i in range(1, 6):
        TRANSITIONS.append((i, i + 3, "Jump 3"))
    # 4. T1 -> Tj (Fixed Context T1)
    for j in range(4, 9):
        # Avoid duplicates if already added (T1->T4 is Jump 3, T1->T2 is Jump 1)
        if not any(t[0] == 1 and t[1] == j for t in TRANSITIONS):
            TRANSITIONS.append((1, j, "Fixed T1"))
    
    CSV_PATH = os.path.join(PROJECT_ROOT, 'Documentation/interpolation_physics_results.csv')
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    all_results = []
    
    # 2. Handle Restarts: Load existing results from CSV if it exists
    if os.path.exists(CSV_PATH):
        try:
            print(f"Reading existing results from {CSV_PATH} to resume work...")
            df_existing = pd.read_csv(CSV_PATH)
            # Check if columns are compatible
            if 'HistoryEnd' in df_existing.columns and 'TargetStep' in df_existing.columns:
                all_results = df_existing.to_dict('records')
                print(f"Found {len(all_results)} existing records. Skipping completed tasks.")
            else:
                print("Existing CSV has incompatible format. Starting fresh.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}. Starting fresh.")

    # 3. Open HDF5 and prepare for data extraction
    print(f"Opening HDF5 dataset: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        data_ds = f['data']
        print("Extracting Reynolds number parameters from the entire dataset (1M samples)...")
        full_params = data_ds[:, 0, 0, 51]
        
        # 4. Loop through each Reynolds Number (Flow Condition)
        for reynolds in tqdm(REYNOLDS_NUMBERS, desc="Overall Progress (Reynolds)"):
            
            # Check if this Reynolds number has already been fully processed
            processed_pairs = []
            if all_results and 'HistoryEnd' in all_results[0] and 'TargetStep' in all_results[0]:
                processed_pairs = [(r['HistoryEnd'], r['TargetStep']) for r in all_results if abs(r['Reynolds'] - reynolds) < 0.01]
            
            all_required_pairs = [(t[0], t[1]) for t in TRANSITIONS]
            if processed_pairs and all(pair in processed_pairs for pair in all_required_pairs):
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
            
            # Setup coordinate maps for grid reconstruction
            ny, nz = len(unique_y), len(unique_z)
            nx = 26 # Constant X resolution in our dataset
            y_map = {val: i for i, val in enumerate(unique_y)}
            z_map = {val: i for i, val in enumerate(unique_z)}
            x_coords = data_ds[0, 0, :, 47]
            
            # 6. Loop through Transitions
            for history_end, target_step, jump_type in TRANSITIONS:
                # Skip if already in results
                if any(abs(r.get('Reynolds', 0) - reynolds) < 0.01 and r.get('HistoryEnd') == history_end and r.get('TargetStep') == target_step for r in all_results):
                    continue
                
                print(f"--- Predicting T{target_step} using T1-T{history_end} as history ({jump_type}) ---")
                V = np.zeros((nx, ny, nz, 3)) # Predicted velocity volume
                V_gt = np.zeros((nx, ny, nz, 3)) # Ground truth volume
                
                # 7. Autoregressive Prediction for each spatial point in the grid
                desc_str = f"Predicting Grid (Re={reynolds}, T{history_end}->T{target_step})"
                for idx in tqdm(grid_indices, desc=desc_str, leave=True):
                    y_val = data_ds[idx, 0, 0, 48]
                    z_val = data_ds[idx, 0, 0, 49]
                    iy, iz = y_map[y_val], z_map[z_val]
                    
                    sample_data = data_ds[idx] # (8, 26, 52)
                    sample_flat = sample_data.reshape(-1, 52)
                    
                    # Store Ground Truth for target timestep (T_target)
                    # Indices are 0-based, so T1 is 0:26, T2 is 26:52, etc.
                    # T_target tokens are [(target_step-1)*26 : target_step*26]
                    start_token_idx = (target_step - 1) * 26
                    end_token_idx = target_step * 26
                    
                    t_target_gt_latent = sample_flat[start_token_idx:end_token_idx, :47]
                    decoded_gt = ae.decode(torch.from_numpy(t_target_gt_latent).float().to(device))
                    v_gt_full = decoded_gt.reshape(26, 125, 3)
                    v_gt_center = v_gt_full[:, TRIPLET_IDX, :].detach().cpu().numpy()
                    V_gt[:, iy, iz, :] = converter.unconvert(v_gt_center)
                    
                    # Provide history tokens (T1 to T_history_end)
                    ctx_len = history_end * 26
                    inputs = torch.from_numpy(sample_flat[:ctx_len]).float().unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        current_seq = inputs.clone()
                        # Predict point-by-point until the end of Target Step
                        for step in range(ctx_len, end_token_idx):
                            step_out = transformer(current_seq)
                            next_latent = step_out[:, -1, :] 
                            
                            new_token = torch.from_numpy(sample_flat[step:step+1]).float().unsqueeze(0).to(device)
                            new_token[0, 0, :47] = next_latent 
                            current_seq = torch.cat([current_seq, new_token], dim=1)
                        
                        # Extract the predicted tokens for target T
                        t_target_latents = current_seq[0, start_token_idx:end_token_idx, :47]
                        decoded = ae.decode(t_target_latents) 
                        v_full = decoded.reshape(26, 125, 3)
                        v_center = v_full[:, TRIPLET_IDX, :].detach().cpu().numpy()
                        V[:, iy, iz, :] = converter.unconvert(v_center)
                
                # 8. Physical Property Calculation
                print(f"Calculating physics on predicted T{target_step} field...")
                u, v, w = V[..., 0], V[..., 1], V[..., 2]
                wx, wy, wz, div_V = calculate_vorticity(V, x_coords, unique_y, unique_z)
                ke = 0.5 * (u**2 + v**2 + w**2)
                helicity = u*wx + v*wy + w*wz
                enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
                
                # 9. SINDy Formula Recovery
                print(f"Running SINDy regressions...")
                ke_res = run_sindy_ke(u, v, w, ke)
                hel_res = run_sindy_helicity(u, v, w, wx, wy, wz, helicity)
                ens_res = run_sindy_enstrophy(wx, wy, wz, enstrophy)
                
                # 10. Additional Physics: Divergence
                div_rmse = np.sqrt(np.mean(div_V**2))
                
                # 11. Calculate Prediction RMSE
                pred_rmse = np.sqrt(np.mean((V - V_gt)**2))
                
                # Package results
                row = {
                    'Reynolds': reynolds,
                    'HistoryEnd': history_end,
                    'TargetStep': target_step,
                    'JumpType': jump_type,
                    'TransitionLabel': f"T1-{history_end}->T{target_step}",
                    'RMSE': pred_rmse,
                    'Divergence_RMSE': div_rmse
                }
                row.update(ke_res)
                row.update(hel_res)
                row.update(ens_res)
                all_results.append(row)
                
                pd.DataFrame(all_results).to_csv(CSV_PATH, index=False)
                print(f"Step complete. Results saved to {CSV_PATH}")

    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n--- Evaluation Finished! ---")
    plot_results(df)

def plot_results(df):
    print("Generating visualizations...")
    metrics = ['RMSE', 'Enstrophy_MSE', 'Divergence_RMSE', 'KE_MSE', 'Helicity_MSE']
    jump_types = df['JumpType'].unique()
    reynolds_list = sorted(df['Reynolds'].unique())
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        fig, axes = plt.subplots(len(jump_types), 1, figsize=(14, 6 * len(jump_types)), sharex=False)
        if len(jump_types) == 1:
            axes = [axes]
            
        for idx, jump in enumerate(jump_types):
            ax = axes[idx]
            jump_df = df[df['JumpType'] == jump]
            
            for reynolds in reynolds_list:
                subset = jump_df[jump_df['Reynolds'] == reynolds].sort_values('TargetStep')
                if subset.empty:
                    continue
                ax.plot(subset['TransitionLabel'], subset[metric], marker='o', label=f"Re={reynolds}")
            
            if 'MSE' in metric:
                ax.set_yscale('log')
            
            ax.set_title(f'{metric} for {jump}')
            ax.set_xlabel('Transition')
            ax.set_ylabel(metric)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        PLOT_PATH = os.path.join(PROJECT_ROOT, f'Documentation/interpolation_{metric.lower()}.png')
        PLOT_PATH_PDF = os.path.join(PROJECT_ROOT, f'Documentation/interpolation_{metric.lower()}.pdf')
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=600, bbox_inches='tight')
        plt.savefig(PLOT_PATH_PDF, dpi=600, bbox_inches='tight')
        print(f"Plots saved to {PLOT_PATH} and {PLOT_PATH_PDF}")

if __name__ == "__main__":
    main()
