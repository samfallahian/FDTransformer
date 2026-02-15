import os
import sys
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3d projection

# Add project root to sys.path to allow imports from other modules
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
sys.path.insert(0, PROJECT_ROOT)

# Import model definitions
# Assuming Ordered_100_TrainTransformer_v1.py is in the same directory as transformer_model_v1.py
from transformer.transformer_model_v1 import OrderedTransformerV1
from encoder.permutations.model_09_residual_ae import ResidualAE
from TransformLatent import FloatConverter

# ANSI Colors for Rainbow effect and highlighting
class Colors:
    CSI = "\033["
    RED = f"{CSI}91m"
    GREEN = f"{CSI}92m"
    YELLOW = f"{CSI}93m"
    BLUE = f"{CSI}94m"
    MAGENTA = f"{CSI}95m"
    CYAN = f"{CSI}96m"
    BOLD = f"{CSI}1m"
    RESET = f"{CSI}0m"
    
    @staticmethod
    def rainbow(text):
        """Create rainbow effect for text"""
        colors = [Colors.RED, Colors.YELLOW, Colors.GREEN, Colors.CYAN, Colors.BLUE, Colors.MAGENTA]
        result = []
        k = 0
        for char in text:
            if char.strip():
                result.append(f"{colors[k % len(colors)]}{char}")
                k += 1
            else:
                result.append(char)
        return ''.join(result) + Colors.RESET

# --- Configuration ---
class Config:
    # Model checkpoints
    TRANSFORMER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/transformer/best_ordered_transformer_v1.pt"
    ENCODER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/Model_09_Residual_AE_epoch_500.pt"
    
    # Data path
    EVAL_H5 = "/Users/kkreth/PycharmProjects/data/evaluation_data.h5"
    VAL_H5 = "/Users/kkreth/PycharmProjects/data/validation_data.h5"
    
    @staticmethod
    def get_data_path():
        if os.path.exists(Config.EVAL_H5):
            return Config.EVAL_H5
        return Config.VAL_H5
    
    # Device
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Dimensions
    LATENT_DIM = 47
    NUM_X = 26
    NUM_TIME = 8
    SEQ_LEN = NUM_X * NUM_TIME # 208
    INPUT_DIM = 52
    
    # For reporting
    TRIPLET_IDX = 62 # 63rd triplet (0-indexed 62)
    
    # Evaluation target positions (from training script)
    TARGET_POSITIONS = list(range(SEQ_LEN - 4, SEQ_LEN)) 
    TARGET_POSITIONS_2 = list(range(SEQ_LEN - 8, SEQ_LEN))
    TARGET_POSITIONS_3 = list(range(SEQ_LEN - 16, SEQ_LEN))
    
    # Fast evaluation
    LIMIT_SAMPLES = 10000 # Only process 10,000 of the 1MM records

# --- Dataset ---
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, max_samples=None):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_available = f['data'].shape[0]
            self.has_originals = 'originals' in f
            if max_samples is not None:
                self.length = min(max_samples, total_available)
            else:
                self.length = total_available
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        data = self._file['data'][idx] # (8, 26, 52)
        # Flatten time and space: (208, 52)
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        
        if self.has_originals:
            orig = self._file['originals'][idx] # (26, 3)
            return torch.from_numpy(data).float(), torch.from_numpy(orig).float()
            
        return torch.from_numpy(data).float(), torch.zeros((Config.NUM_X, 3))

def load_models():
    # 1. Load Transformer
    print(f"Loading Transformer from: {Colors.CYAN}{Config.TRANSFORMER_CHECKPOINT}{Colors.RESET}")
    checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=Config.DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Use the embedded model object for maximum compatibility
        transformer = checkpoint['model']
    else:
        # Reconstruct if necessary (using config in checkpoint)
        print("Reconstructing Transformer model from checkpoint config...")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        
    transformer.to(Config.DEVICE)
    transformer.eval()
    
    # 2. Load Residual AE (Encoder/Decoder)
    print(f"Loading Residual AE from: {Colors.CYAN}{Config.ENCODER_CHECKPOINT}{Colors.RESET}")
    ae_checkpoint = torch.load(Config.ENCODER_CHECKPOINT, map_location=Config.DEVICE, weights_only=False)
    
    if isinstance(ae_checkpoint, dict) and 'model' in ae_checkpoint:
        ae = ae_checkpoint['model']
    else:
        # Standard initialization for ResidualAE
        ae = ResidualAE()
        if isinstance(ae_checkpoint, dict) and 'model_state_dict' in ae_checkpoint:
            ae.load_state_dict(ae_checkpoint['model_state_dict'])
        else:
            ae.load_state_dict(ae_checkpoint)
            
    ae.to(Config.DEVICE)
    ae.eval()
    
    return transformer, ae

def main():
    # Big Rainbow Message
    msg = f"USING DEVICE: {Config.DEVICE.upper()} - MODEL IS IN EVAL MODE AND CANNOT BE TRAINED"
    print("\n" + "="*80)
    print(Colors.rainbow(f"  {msg}  "))
    print("="*80 + "\n")
    
    # Load models
    try:
        transformer, ae = load_models()
        converter = FloatConverter()
    except Exception as e:
        print(f"{Colors.RED}Error loading models: {e}{Colors.RESET}")
        return

    # Load dataset
    try:
        data_path = Config.get_data_path()
        print(f"Using dataset: {Colors.YELLOW}{data_path}{Colors.RESET}")
        dataset = EvalDataset(data_path, max_samples=Config.LIMIT_SAMPLES)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    except Exception as e:
        print(f"{Colors.RED}Error loading dataset: {e}{Colors.RESET}")
        return

    print(f"Validation dataset size: {len(dataset)}")
    
    # Evaluation Loop
    total_samples_processed = 0
    detailed_reports = []
    NUM_DETAILED = 3
    stats_data = []

    with torch.no_grad():
        for batch_idx, (batch, originals_batch) in enumerate(tqdm(loader, desc="Evaluating Transformer")):
            batch = batch.to(Config.DEVICE) # (B, 208, 52)
            originals_batch = originals_batch.to(Config.DEVICE) # (B, 26, 3)
            B = batch.shape[0]
            
            # 1. Transformer Prediction
            inputs = batch[:, :-1, :]
            outputs = transformer(inputs) # (B, 207, Config.LATENT_DIM)
            
            # 2. Extract 8th time step predictions
            t8_target_indices = range(182, 208)
            t8_output_indices = [i-1 for i in t8_target_indices]
            
            pred_latents_t8 = outputs[:, t8_output_indices, :] # (B, 26, 47)
            gt_latents_t8 = batch[:, t8_target_indices, :Config.LATENT_DIM] # (B, 26, 47)
            
            # Metadata for 8th time step
            coords_t8 = batch[:, t8_target_indices, 47:50] # (B, 26, 3)
            param_t8 = batch[:, t8_target_indices, 51] # (B, 26)
            
            # 3. Decode Latents to Velocities
            pred_latents_flat = pred_latents_t8.reshape(-1, Config.LATENT_DIM)
            gt_latents_flat = gt_latents_t8.reshape(-1, Config.LATENT_DIM)
            
            pred_velocities_full = ae.decode(pred_latents_flat) # (B*26, 375)
            gt_velocities_full = ae.decode(gt_latents_flat) # (B*26, 375)
            
            # Reshape and extract 63rd triplet (Central Velocity)
            pred_v_63 = pred_velocities_full.reshape(B, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
            gt_v_63 = gt_velocities_full.reshape(B, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
            
            # 4. Denormalize for statistics
            pred_v_63_np = pred_v_63.cpu().numpy()
            pred_denorm_v = converter.unconvert(pred_v_63_np.reshape(-1, 3)).reshape(B, 26, 3)
            
            if dataset.has_originals:
                gt_denorm_v = originals_batch.cpu().numpy()
            else:
                gt_v_63_np = gt_v_63.cpu().numpy()
                gt_denorm_v = converter.unconvert(gt_v_63_np.reshape(-1, 3)).reshape(B, 26, 3)
                
            # Calculate squared errors
            sq_errors = np.sum((gt_denorm_v - pred_denorm_v)**2, axis=2) # (B, 26)
            params = param_t8[:, 0].cpu().numpy() # (B,)
            
            for i in range(B):
                for j in range(26):
                    stats_data.append({
                        'param': params[i],
                        'pos_idx': j,
                        'sq_error': sq_errors[i, j],
                        'y': coords_t8[i, j, 1].item(),
                        'z': coords_t8[i, j, 2].item()
                    })
            
            # Detailed reporting for first few samples
            if batch_idx == 0:
                for i in range(min(B, NUM_DETAILED)):
                    sample_report = {
                        'sample_idx': i,
                        'param': params[i],
                        'y': coords_t8[i, 0, 1].item(),
                        'z': coords_t8[i, 0, 2].item(),
                        'positions': []
                    }
                    for j in range(26):
                        original_pos_idx = t8_target_indices[j]
                        is_target = original_pos_idx in Config.TARGET_POSITIONS
                        is_target_2 = original_pos_idx in Config.TARGET_POSITIONS_2
                        is_target_3 = original_pos_idx in Config.TARGET_POSITIONS_3
                        
                        target_label = "L4" if is_target else "L8" if is_target_2 else "L16" if is_target_3 else "T8"
                        
                        sample_report['positions'].append({
                            'idx': original_pos_idx,
                            'x': coords_t8[i, j, 0].item(),
                            'label': target_label,
                            'orig_v': gt_denorm_v[i, j],
                            'gt_v': gt_v_63[i, j].cpu().numpy(),
                            'pred_v': pred_v_63[i, j].cpu().numpy(),
                            'pred_denorm': pred_denorm_v[i, j]
                        })
                    detailed_reports.append(sample_report)

            total_samples_processed += B

    # --- Statistics Calculation ---
    print(f"\n{Colors.BOLD}CALCULATING SUMMARY STATISTICS...{Colors.RESET}")
    df = pd.DataFrame(stats_data)
    
    # RMSE per experiment
    rmse_per_param = df.groupby('param')['sq_error'].mean().apply(np.sqrt)
    
    # RMSE per position
    rmse_per_pos = df.groupby('pos_idx')['sq_error'].mean().apply(np.sqrt)
    
    # RMSE for prediction windows
    # j ranges from 0 (idx 182) to 25 (idx 207)
    # L4: indices 204-207 -> j: 22-25
    # L8: indices 200-207 -> j: 18-25
    # L16: indices 192-207 -> j: 10-25
    rmse_l4 = np.sqrt(df[df['pos_idx'] >= 22]['sq_error'].mean())
    rmse_l8 = np.sqrt(df[df['pos_idx'] >= 18]['sq_error'].mean())
    rmse_l16 = np.sqrt(df[df['pos_idx'] >= 10]['sq_error'].mean())
    rmse_overall = np.sqrt(df['sq_error'].mean())

    # --- Figures ---
    print(f"{Colors.BOLD}GENERATING FIGURES...{Colors.RESET}")
    
    # 1. RMSE vs Position
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_per_pos.index, rmse_per_pos.values, marker='o', linestyle='-', color='b')
    plt.axvspan(22, 25, alpha=0.2, color='red', label='L4 Window')
    plt.axvspan(18, 25, alpha=0.1, color='orange', label='L8 Window')
    plt.axvspan(10, 25, alpha=0.05, color='yellow', label='L16 Window')
    plt.title('RMSE per Position in T8')
    plt.xlabel('Position Index (0-25)')
    plt.ylabel('RMSE (Velocity Units)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig('rmse_per_position.png')
    print(f"Saved: {Colors.CYAN}rmse_per_position.png{Colors.RESET}")
    
    # 2. RMSE per Window
    plt.figure(figsize=(8, 6))
    windows = ['L4 (Last 4)', 'L8 (Last 8)', 'L16 (Last 16)', 'Overall T8']
    vals = [rmse_l4, rmse_l8, rmse_l16, rmse_overall]
    colors = ['red', 'orange', 'gold', 'green']
    plt.bar(windows, vals, color=colors)
    plt.title('RMSE per Prediction Window')
    plt.ylabel('RMSE (Velocity Units)')
    for i, v in enumerate(vals):
        plt.text(i, v + (max(vals)*0.01), f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig('rmse_per_window.png')
    print(f"Saved: {Colors.CYAN}rmse_per_window.png{Colors.RESET}")
    
    # 3. RMSE per Experiment
    plt.figure(figsize=(12, 6))
    rmse_per_param.plot(kind='bar', color='skyblue')
    plt.title('RMSE per Experiment (Param)')
    plt.ylabel('RMSE (Velocity Units)')
    plt.xlabel('Experiment Param')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rmse_per_experiment.png')
    print(f"Saved: {Colors.CYAN}rmse_per_experiment.png{Colors.RESET}")

    # 4. RMSE vs Y/Z Coordinate space (Heatmap-style Scatter)
    plt.figure(figsize=(10, 8))
    # Aggregate by Y, Z
    yz_stats = df.groupby(['y', 'z'])['sq_error'].mean().apply(np.sqrt).reset_index()
    yz_stats.columns = ['y', 'z', 'rmse']
    
    sc = plt.scatter(yz_stats['y'], yz_stats['z'], c=yz_stats['rmse'], cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(sc, label='RMSE')
    plt.title('RMSE Distribution in Y-Z Coordinate Space')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('rmse_yz_space.png')
    print(f"Saved: {Colors.CYAN}rmse_yz_space.png{Colors.RESET}")

    # 5. 3D Error Density/Surface Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Using the aggregated Y, Z stats
    # We'll use a scatter3D but could also try to interpolate for a surface
    img = ax.scatter3D(yz_stats['y'], yz_stats['z'], yz_stats['rmse'], 
                       c=yz_stats['rmse'], cmap='magma', s=60)
    
    ax.set_title('3D Error Magnitude across Y-Z Space')
    ax.set_xlabel('Y Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('RMSE')
    fig.colorbar(img, ax=ax, label='RMSE', shrink=0.5, aspect=10)
    
    # Add a "shadow" on the floor for better depth perception
    ax.scatter3D(yz_stats['y'], yz_stats['z'], np.zeros_like(yz_stats['rmse']), 
                 c='gray', alpha=0.1, s=10)
    
    plt.savefig('rmse_3d_density.png')
    print(f"Saved: {Colors.CYAN}rmse_3d_density.png{Colors.RESET}")

    # 6. Creative Plot: Hexbin Error Density
    plt.figure(figsize=(10, 8))
    # We use all raw points for hexbin to show "density" of error samples if they overlap
    hb = plt.hexbin(df['y'], df['z'], C=df['sq_error'].apply(np.sqrt), gridsize=25, cmap='magma', reduce_C_function=np.mean)
    cb = plt.colorbar(hb, label='Mean RMSE')
    plt.title('Hexbin RMSE Density Map (Y-Z Plane)')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.savefig('rmse_yz_hexbin.png')
    print(f"Saved: {Colors.CYAN}rmse_yz_hexbin.png{Colors.RESET}")

    # --- Final Report ---
    print(f"\n{Colors.BOLD}SUMMARY STATISTICS{Colors.RESET}")
    print("-" * 50)
    print(f"Overall RMSE:       {Colors.GREEN}{rmse_overall:.6f}{Colors.RESET}")
    print(f"RMSE (Last 4):      {Colors.YELLOW}{rmse_l4:.6f}{Colors.RESET}")
    print(f"RMSE (Last 8):      {Colors.YELLOW}{rmse_l8:.6f}{Colors.RESET}")
    print(f"RMSE (Last 16):     {Colors.YELLOW}{rmse_l16:.6f}{Colors.RESET}")
    print("-" * 50)
    print(f"RMSE per Experiment (First 10 Params):\n{rmse_per_param.head(10)}")
    print("-" * 50)

    print(f"\n{Colors.BOLD}DETAILED SAMPLES (First {NUM_DETAILED}){Colors.RESET}")
    for report in detailed_reports:
        print(f"{Colors.MAGENTA}Sample {report['sample_idx']} | Param: {report['param']:.2f} | Y: {report['y']:.4f} | Z: {report['z']:.4f}{Colors.RESET}")
        header = f"{'Pos':<4} | {'X':<6} | {'Label':<6} | {'Truth (vx, vy, vz)':<30} | {'Predicted (vx, vy, vz)':<30} | {'Error'}"
        print(header)
        print("-" * len(header))
        
        for pos in report['positions']:
            gt = pos['orig_v']
            pred_denorm = pos['pred_denorm']
            err = np.linalg.norm(gt - pred_denorm)
            
            gt_str = f"({gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f})"
            pred_denorm_str = f"({pred_denorm[0]:.4f}, {pred_denorm[1]:.4f}, {pred_denorm[2]:.4f})"
            
            row_style = Colors.BOLD if "L" in pos['label'] else ""
            err_col = f"{Colors.CYAN}{err:.6f}{Colors.RESET}"
            
            print(f"{row_style}{pos['idx']-182:<4} | {pos['x']:<6.1f} | {pos['label']:<6} | {gt_str:<30} | {pred_denorm_str:<30} | {err_col}{Colors.RESET}")
        print()

    print(f"{Colors.GREEN}Evaluation complete!{Colors.RESET}")

if __name__ == "__main__":
    main()
