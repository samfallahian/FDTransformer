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
    ENCODER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
    
    # Data path
    EVAL_H5 = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
    VAL_H5 = "/Users/kkreth/PycharmProjects/data/transformer_input/validation_data.h5"
    
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
    NUM_TIME = 80
    SEQ_LEN = NUM_X * NUM_TIME # 2080
    INPUT_DIM = 52
    
    # For reporting
    TRIPLET_IDX = 62 # 63rd triplet (0-indexed 62)
    
    # Evaluation target positions (from training script)
    # SEQ_LEN - 4, SEQ_LEN - 8, SEQ_LEN - 16
    TARGET_POSITIONS = list(range(SEQ_LEN - 4, SEQ_LEN)) 
    TARGET_POSITIONS_2 = list(range(SEQ_LEN - 8, SEQ_LEN))
    TARGET_POSITIONS_3 = list(range(SEQ_LEN - 16, SEQ_LEN))
    
    # Fast evaluation
    LIMIT_SAMPLES = 100 # Only process 10,000 of the 1MM records
    
    # Staircase Evaluation Settings
    # We want to predict the 8th spatial position of the 80th time period (Global Index 2061)
    # given 1 to 7 spatial positions of the 80th time period as context.
    # 80th time period starts at 79 * 26 = 2054. 8th position is 2054 + 7 = 2061.
    STAIRCASE_TARGET_IDX = 2061 
    STAIRCASE_CONTEXT_COUNTS = list(range(1, 8)) # 1, 2, 3, 4, 5, 6, 7

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
        data = self._file['data'][idx] # (80, 26, 52)
        # Flatten time and space: (2080, 52)
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        
        if self.has_originals:
            orig = self._file['originals'][idx] # (26, 3)
            return torch.from_numpy(data).float(), torch.from_numpy(orig).float()
            
        return torch.from_numpy(data).float(), torch.zeros((Config.NUM_X, 3))

def load_models():
    # Patch for torch._dynamo compatibility issues (e.g., missing ConvertFrameBox)
    try:
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class DummyConvertFrameBox:
                def __setstate__(self, state):
                    self.__dict__.update(state)
            torch._dynamo.convert_frame.ConvertFrameBox = DummyConvertFrameBox
    except (ImportError, AttributeError):
        pass

    # 1. Load Transformer
    print(f"Loading Transformer from: {Colors.CYAN}{Config.TRANSFORMER_CHECKPOINT}{Colors.RESET}")
    checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=Config.DEVICE, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Use the embedded model object for maximum compatibility
        transformer = checkpoint['model']
        
        # If it's a compiled model (OptimizedModule), get the original model
        if hasattr(transformer, '_orig_mod'):
            print("Detected compiled model, extracting original module...")
            transformer = transformer._orig_mod
        elif hasattr(transformer, 'module'):
            # In some cases it might be wrapped in DataParallel/DistributedDataParallel
            transformer = transformer.module
    else:
        # Reconstruct if necessary (using config in checkpoint)
        print("Reconstructing Transformer model from checkpoint config...")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        
    transformer.to(Config.DEVICE)
    transformer.eval()
    
    # 2. Load Encoder/Decoder (TorchScript "one file" approach)
    print(f"Loading Scripted AE from: {Colors.CYAN}{Config.ENCODER_CHECKPOINT}{Colors.RESET}")
    ae = torch.jit.load(Config.ENCODER_CHECKPOINT, map_location=Config.DEVICE)
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
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    except Exception as e:
        print(f"{Colors.RED}Error loading dataset: {e}{Colors.RESET}")
        return

    print(f"Validation dataset size: {len(dataset)}")
    
    # Evaluation Loop
    total_samples_processed = 0
    detailed_reports = []
    NUM_DETAILED = 3
    stats_data = []
    staircase_data = [] # To store {context_count: [sq_errors]}

    with torch.no_grad():
        for batch_idx, (batch, originals_batch) in enumerate(tqdm(loader, desc="Evaluating Transformer")):
            batch = batch.to(Config.DEVICE) # (B, 2080, 52)
            originals_batch = originals_batch.to(Config.DEVICE) # (B, 26, 3)
            B = batch.shape[0]
            
            # 1. Standard Transformer Prediction (Full sequence)
            inputs = batch[:, :-1, :]
            outputs = transformer(inputs) # (B, 2079, Config.LATENT_DIM)
            
            # 2. Extract 80th time step predictions
            # 80th time period starts at 79 * 26 = 2054, ends at 80 * 26 = 2080
            t80_target_indices = range(2054, 2080)
            t80_output_indices = [i-1 for i in t80_target_indices]
            
            pred_latents_t80 = outputs[:, t80_output_indices, :] # (B, 26, 47)
            gt_latents_t80 = batch[:, t80_target_indices, :Config.LATENT_DIM] # (B, 26, 47)
            
            # Metadata for 80th time step
            coords_t80 = batch[:, t80_target_indices, 47:50] # (B, 26, 3)
            param_t80 = batch[:, t80_target_indices, 51] # (B, 26)
            
            # 3. Decode Latents to Velocities
            pred_latents_flat = pred_latents_t80.reshape(-1, Config.LATENT_DIM)
            gt_latents_flat = gt_latents_t80.reshape(-1, Config.LATENT_DIM)
            
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
            params = param_t80[:, 0].cpu().numpy() # (B,)
            
            for i in range(B):
                for j in range(26):
                    stats_data.append({
                        'param': params[i],
                        'pos_idx': j,
                        'sq_error': sq_errors[i, j],
                        'y': coords_t80[i, j, 1].item(),
                        'z': coords_t80[i, j, 2].item()
                    })
            
            # --- Staircase Evaluation ---
            # Predict STAIRCASE_TARGET_IDX (2061) given varying context from T80 (2054 onwards)
            target_idx = Config.STAIRCASE_TARGET_IDX
            
            # GT Velocity for point 2061
            # In our current batch processing for T80, index 2061 is j=7
            gt_v_staircase = gt_denorm_v[:, 7, :] # (B, 3)
            
            for k in Config.STAIRCASE_CONTEXT_COUNTS:
                # k is number of points from T80 given.
                # If k=7, we have 2054...2060. Prediction for 2061 is at index 2060.
                # If k < 7, we need to autoregressively predict up to 2061.
                
                # Context sequence length
                context_len = 2054 + k
                
                if k == 7:
                    # Single step prediction sufficient
                    staircase_pred_latent = outputs[:, context_len - 1, :] # (B, 47)
                else:
                    # Autoregressive prediction
                    # We start with context_len points and need to reach 2061+1 points
                    # predict_autoregressive expects (1, L, 52)
                    # We'll do it manually here for the batch to be faster
                    current_seq = batch[:, :context_len, :].clone()
                    for step in range(context_len, target_idx + 1):
                        step_out = transformer(current_seq)
                        next_latent = step_out[:, -1, :] # (B, 47)
                        
                        # Prepare next input: we need to append the next point's metadata
                        # but we only have metadata for the batch.
                        if step < Config.SEQ_LEN:
                            # Use metadata from the original batch
                            new_token = batch[:, step:step+1, :].clone()
                            new_token[:, 0, :Config.LATENT_DIM] = next_latent
                            current_seq = torch.cat([current_seq, new_token], dim=1)
                    staircase_pred_latent = next_latent
                
                # Decode and denormalize
                dec_v = ae.decode(staircase_pred_latent) # (B, 375)
                pred_v_63_staircase = dec_v.reshape(B, 125, 3)[:, Config.TRIPLET_IDX, :]
                pred_denorm_v_staircase = converter.unconvert(pred_v_63_staircase.cpu().numpy()) # (B, 3)
                
                # Calculate SQ Error
                sq_err_staircase = np.sum((gt_v_staircase - pred_denorm_v_staircase)**2, axis=1) # (B,)
                
                for err in sq_err_staircase:
                    staircase_data.append({'context_count': k, 'sq_error': err})
            
            # Detailed reporting for first few samples
            if batch_idx == 0:
                for i in range(min(B, NUM_DETAILED)):
                    sample_report = {
                        'sample_idx': i,
                        'param': params[i],
                        'y': coords_t80[i, 0, 1].item(),
                        'z': coords_t80[i, 0, 2].item(),
                        'positions': []
                    }
                    for j in range(26):
                        original_pos_idx = t80_target_indices[j]
                        is_target = original_pos_idx in Config.TARGET_POSITIONS
                        is_target_2 = original_pos_idx in Config.TARGET_POSITIONS_2
                        is_target_3 = original_pos_idx in Config.TARGET_POSITIONS_3
                        
                        target_label = "L4" if is_target else "L8" if is_target_2 else "L16" if is_target_3 else "T80"
                        
                        sample_report['positions'].append({
                            'idx': original_pos_idx,
                            'x': coords_t80[i, j, 0].item(),
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

    # Staircase RMSE
    df_staircase = pd.DataFrame(staircase_data)
    rmse_staircase = df_staircase.groupby('context_count')['sq_error'].mean().apply(np.sqrt)
    
    # RMSE for prediction windows
    # j ranges from 0 (idx 2054) to 25 (idx 2079)
    # L4: indices 2076-2079 -> j: 22-25
    # L8: indices 2072-2079 -> j: 18-25
    # L16: indices 2064-2079 -> j: 10-25
    rmse_l4 = np.sqrt(df[df['pos_idx'] >= 22]['sq_error'].mean())
    rmse_l8 = np.sqrt(df[df['pos_idx'] >= 18]['sq_error'].mean())
    rmse_l16 = np.sqrt(df[df['pos_idx'] >= 10]['sq_error'].mean())
    rmse_overall = np.sqrt(df['sq_error'].mean())

    # --- Export Results ---
    print(f"\n{Colors.BOLD}EXPORTING RESULTS...{Colors.RESET}")
    # Aggregate Y, Z stats for plotting
    yz_stats = df.groupby(['y', 'z'])['sq_error'].mean().apply(np.sqrt).reset_index()
    yz_stats.columns = ['y', 'z', 'rmse']

    results = {
        'rmse_per_pos': rmse_per_pos.to_dict(),
        'rmse_staircase': rmse_staircase.to_dict(),
        'rmse_per_param': rmse_per_param.to_dict(),
        'yz_stats': yz_stats.to_dict(orient='records'),
        'rmse_l4': float(rmse_l4),
        'rmse_l8': float(rmse_l8),
        'rmse_l16': float(rmse_l16),
        'rmse_overall': float(rmse_overall),
        'detailed_reports': detailed_reports
    }

    # Helper to convert numpy types for JSON serialization
    def default_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open('evaluation_results.json', 'w') as f:
        import json
        json.dump(results, f, default=default_converter)
    print(f"Results exported to: {Colors.CYAN}evaluation_results.json{Colors.RESET}")

    # --- Final Report ---
    print(f"\n{Colors.BOLD}SUMMARY STATISTICS (Velocity Units){Colors.RESET}")
    print("-" * 50)
    print(f"Overall RMSE:       {Colors.GREEN}{rmse_overall:.4e}{Colors.RESET}")
    print(f"RMSE (Last 4):      {Colors.YELLOW}{rmse_l4:.4e}{Colors.RESET}")
    print(f"RMSE (Last 8):      {Colors.YELLOW}{rmse_l8:.4e}{Colors.RESET}")
    print(f"RMSE (Last 16):     {Colors.YELLOW}{rmse_l16:.4e}{Colors.RESET}")
    print("-" * 50)
    print(f"STAIRCASE EVALUATION (Velocity Units):")
    for k, val in rmse_staircase.items():
        print(f"Given {k} in T80 -> Pred pos 8 RMSE: {Colors.CYAN}{val:.4e}{Colors.RESET}")
    print("-" * 50)
    print(f"RMSE per Experiment (First 10 Params):\n{rmse_per_param.head(10).apply(lambda x: f'{x:.4e}')}")
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
            err_col = f"{Colors.CYAN}{err:.4e}{Colors.RESET}"
            
            print(f"{row_style}{pos['idx']-2054:<4} | {pos['x']:<6.1f} | {pos['label']:<6} | {gt_str:<30} | {pred_denorm_str:<30} | {err_col}{Colors.RESET}")
        print()

    print(f"{Colors.GREEN}Evaluation complete!{Colors.RESET}")

if __name__ == "__main__":
    main()
