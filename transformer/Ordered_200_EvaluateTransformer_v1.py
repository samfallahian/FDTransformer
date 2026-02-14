import os
import sys
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

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
    all_gt_velocities = []
    all_pred_velocities = []
    
    # For detailed reporting, we'll pick the first few samples
    detailed_reports = []
    NUM_DETAILED = 3

    with torch.no_grad():
        for batch_idx, (batch, originals_batch) in enumerate(tqdm(loader, desc="Evaluating Transformer")):
            batch = batch.to(Config.DEVICE) # (B, 208, 52)
            originals_batch = originals_batch.to(Config.DEVICE) # (B, 26, 3)
            B = batch.shape[0]
            
            # 1. Transformer Prediction (One-step ahead for all positions)
            # Input: tokens 0 to 206
            # Output: predicts latents for tokens 1 to 207
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :Config.LATENT_DIM]
            
            outputs = transformer(inputs) # (B, 207, Config.LATENT_DIM)
            
            # 2. Extract 8th time step predictions
            # 8th time step in original batch: indices 182 to 207 (26 points)
            # In 'outputs' tensor (indices 0 to 206), these correspond to:
            # Target 182 is output index 181
            # ...
            # Target 207 is output index 206
            t8_target_indices = range(182, 208)
            t8_output_indices = [i-1 for i in t8_target_indices]
            
            pred_latents_t8 = outputs[:, t8_output_indices, :] # (B, 26, 47)
            gt_latents_t8 = batch[:, t8_target_indices, :Config.LATENT_DIM] # (B, 26, 47)
            
            # Metadata for 8th time step
            coords_t8 = batch[:, t8_target_indices, 47:50] # (B, 26, 3)
            rel_time_t8 = batch[:, t8_target_indices, 50] # (B, 26)
            param_t8 = batch[:, t8_target_indices, 51] # (B, 26)
            
            # 3. Decode Latents to Velocities
            # Flatten to (B*26, 47) for AE
            pred_latents_flat = pred_latents_t8.reshape(-1, Config.LATENT_DIM)
            gt_latents_flat = gt_latents_t8.reshape(-1, Config.LATENT_DIM)
            
            pred_velocities_full = ae.decode(pred_latents_flat) # (B*26, 375)
            gt_velocities_full = ae.decode(gt_latents_flat) # (B*26, 375)
            
            # Reshape to (B, 26, 125, 3)
            pred_v_triplets = pred_velocities_full.reshape(B, 26, 125, 3)
            gt_v_triplets = gt_velocities_full.reshape(B, 26, 125, 3)
            
            # 4. Extract 63rd triplet (Central Velocity)
            pred_v_63 = pred_v_triplets[:, :, Config.TRIPLET_IDX, :] # (B, 26, 3)
            gt_v_63 = gt_v_triplets[:, :, Config.TRIPLET_IDX, :] # (B, 26, 3)
            
            # Accumulate for overall metrics
            # (In a real script you might want to save to CSV or calculate MSE here)
            
            # Detailed reporting for first few samples
            if batch_idx == 0:
                for i in range(min(B, NUM_DETAILED)):
                    sample_report = {
                        'sample_idx': i,
                        'param': param_t8[i, 0].item(),
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
                            'orig_v': originals_batch[i, j].cpu().numpy() if dataset.has_originals else None,
                            'gt_v': gt_v_63[i, j].cpu().numpy(),
                            'pred_v': pred_v_63[i, j].cpu().numpy()
                        })
                    detailed_reports.append(sample_report)

            total_samples_processed += B

    # --- Final Report ---
    print(f"\n{Colors.BOLD}REPORTING ON DE-ENCODED VELOCITIES (8th Time Step, 63rd Triplet){Colors.RESET}")
    print("-" * 180)
    
    # Answer the user's question about time
    print(f"{Colors.YELLOW}Note on Actual Time:{Colors.RESET}")
    print("The validation HDF5 dataset only contains RELATIVE time (0-7).")
    print("Actual (absolute) timestamps were not preserved in the training/validation cubes.\n")

    for report in detailed_reports:
        print(f"{Colors.MAGENTA}Sample {report['sample_idx']} | Param: {report['param']:.2f} | Y: {report['y']} | Z: {report['z']}{Colors.RESET}")
        header = f"{'Pos':<4} | {'X':<6} | {'Label':<6} | {'Originals (vx, vy, vz)':<30} | {'Auto-encoder Truth (vx, vy, vz)':<32} | {'Predicted (vx, vy, vz)':<30} | {'Predicted de-Normalized (vx, vy, vz)':<38} | {'Error'}"
        print(header)
        print("-" * len(header))
        
        for pos in report['positions']:
            gt = pos['gt_v']
            pred = pos['pred_v']
            orig = pos['orig_v']
            pred_denorm = converter.unconvert(pred)
            if orig is not None:
                # Calculate absolute error between Predicted de-Normalized and Originals
                err = np.linalg.norm(orig - pred_denorm)
                err_str = f"{err:.6f}"
            else:
                err_str = "N/A"
            
            orig_str = f"({orig[0]:.4f}, {orig[1]:.4f}, {orig[2]:.4f})" if orig is not None else "N/A"
            gt_str = f"({gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f})"
            pred_str = f"({pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f})"
            pred_denorm_str = f"({pred_denorm[0]:.4f}, {pred_denorm[1]:.4f}, {pred_denorm[2]:.4f})"
            
            # Highlight target rows (L4) in Bold, non-CYAN columns will inherit this
            row_style = Colors.BOLD if "L4" in pos['label'] else ""
            
            # Specifically color the requested columns CYAN
            orig_col = f"{Colors.CYAN}{orig_str:<30}{Colors.RESET}{row_style}"
            pred_denorm_col = f"{Colors.CYAN}{pred_denorm_str:<38}{Colors.RESET}{row_style}"
            err_col = f"{Colors.CYAN}{err_str}{Colors.RESET}"
            
            print(f"{row_style}{pos['idx']:<4} | {pos['x']:<6.1f} | {pos['label']:<6} | {orig_col} | {gt_str:<32} | {pred_str:<30} | {pred_denorm_col} | {err_col}{Colors.RESET}")
        print()

    print(f"{Colors.GREEN}Evaluation complete!{Colors.RESET}")

if __name__ == "__main__":
    main()
