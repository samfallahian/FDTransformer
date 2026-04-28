import os
import sys
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3d projection
from concurrent.futures import ThreadPoolExecutor

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
    
    # Batch size
    BATCH_SIZE = 1 # Reduced to avoid MPS OOM
    
    # Micro-batching for evaluation loops
    # If BATCH_SIZE > 1, some operations might still OOM.
    # We can further process the batch in smaller chunks.
    MICRO_BATCH_SIZE = 1 
    
    # Fast evaluation
    LIMIT_SAMPLES = 10 # Randomly select 50 of the 1MM records (to speed up multiple permutations)
    
    # Staircase settings
    STAIRCASE_CONTEXTS = [1, 10, 20, 40, 60, 79]
    
    # Interleave Evaluation Settings
    # 1. Predict each even frame given the odd frame (T1->T2, T1-3->T4, ...)
    # 2. Predict every 2nd and 3rd only given 1 (C=1, P=2)
    # 3. Predict every 2nd, 3rd, 4th, 5th given 1 (C=1, P=4)
    # 4. Predict P=1 given C=2, P=2 given C=2, ...
    # 5. Collapse limit: RMSE > 0.05
    RMSE_LIMIT = 0.05

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
                # Randomly pick indices from the whole dataset
                self.indices = np.random.choice(total_available, self.length, replace=False)
                # Sort indices to improve HDF5 access performance
                self.indices.sort()
            else:
                self.length = total_available
                self.indices = np.arange(total_available)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        
        # Map the requested idx to our random index
        actual_idx = self.indices[idx]
        data = self._file['data'][actual_idx] # (80, 26, 52)
        # Flatten time and space: (2080, 52)
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        
        if self.has_originals:
            orig = self._file['originals'][actual_idx] # (26, 3)
            return torch.from_numpy(data).float(), torch.from_numpy(orig).float()
            
        return torch.from_numpy(data).float(), torch.zeros((Config.NUM_X, 3))

def load_models(device=None):
    if device is None:
        device = Config.DEVICE
        
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
    print(f"Loading Transformer to {Colors.MAGENTA}{device}{Colors.RESET} from: {Colors.CYAN}{Config.TRANSFORMER_CHECKPOINT}{Colors.RESET}")
    checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Use the embedded model object for maximum compatibility
        transformer = checkpoint['model']
        
        # If it's a compiled model (OptimizedModule), get the original model
        if hasattr(transformer, '_orig_mod'):
            print(f"Detected compiled model for {device}, extracting original module...")
            transformer = transformer._orig_mod
        elif hasattr(transformer, 'module'):
            # In some cases it might be wrapped in DataParallel/DistributedDataParallel
            transformer = transformer.module
    else:
        # Reconstruct if necessary (using config in checkpoint)
        print(f"Reconstructing Transformer model for {device} from checkpoint config...")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        
    transformer.to(device)
    transformer.eval()
    
    # 2. Load Encoder/Decoder (TorchScript "one file" approach)
    print(f"Loading Scripted AE to {Colors.MAGENTA}{device}{Colors.RESET} from: {Colors.CYAN}{Config.ENCODER_CHECKPOINT}{Colors.RESET}")
    ae = torch.jit.load(Config.ENCODER_CHECKPOINT, map_location=device)
    ae.to(device)
    ae.eval()
    
    return transformer, ae

def evaluate_permutation(transformer, ae, converter, batch, num_context_t, num_predict_t, triplet_idx=62, device='cpu'):
    """
    Evaluates a single permutation: (Context Time Steps, Prediction Time Steps).
    Returns the average RMSE across the prediction window.
    Processes in micro-batches to avoid MPS OOM.
    """
    B_full = batch.shape[0]
    num_x = 26
    latent_dim = 47
    
    context_len = num_context_t * num_x
    predict_len = num_predict_t * num_x
    total_len = context_len + predict_len
    
    if total_len > 2080:
        return None
    
    # Indices for the prediction window (time steps starting from context_len)
    pred_indices = range(context_len, total_len)
    
    all_rmse = []
    
    # Process each sample in the batch individually (Micro-batching)
    for i in range(0, B_full, Config.MICRO_BATCH_SIZE):
        micro_batch = batch[i:i+Config.MICRO_BATCH_SIZE]
        B = micro_batch.shape[0]
        
        # Autoregressive prediction
        current_seq = micro_batch[:, :context_len, :].clone()
        
        for step in range(context_len, total_len):
            step_out = transformer(current_seq)
            next_latent = step_out[:, -1, :] # (B, 47)
            
            # Prepare next token using metadata from 'micro_batch'
            new_token = micro_batch[:, step:step+1, :].clone()
            new_token[:, 0, :latent_dim] = next_latent
            current_seq = torch.cat([current_seq, new_token], dim=1)
            
        # Extract predicted and ground truth latents for the prediction window
        pred_latents = current_seq[:, pred_indices, :latent_dim]
        gt_latents = micro_batch[:, pred_indices, :latent_dim]
        
        # Decode and denormalize
        pred_latents_flat = pred_latents.reshape(-1, latent_dim)
        gt_latents_flat = gt_latents.reshape(-1, latent_dim)
        
        pred_dec_v = ae.decode(pred_latents_flat) # (B*pred_len, 375)
        gt_dec_v = ae.decode(gt_latents_flat) # (B*pred_len, 375)
        
        # Extract 63rd triplet
        pred_v_63 = pred_dec_v.reshape(B, num_predict_t, num_x, 125, 3)[:, :, :, triplet_idx, :]
        gt_v_63 = gt_dec_v.reshape(B, num_predict_t, num_x, 125, 3)[:, :, :, triplet_idx, :]
        
        # Denormalize
        pred_v_63_np = pred_v_63.cpu().numpy()
        gt_v_63_np = gt_v_63.cpu().numpy()
        
        pred_denorm = converter.unconvert(pred_v_63_np.reshape(-1, 3)).reshape(B, num_predict_t, num_x, 3)
        gt_denorm = converter.unconvert(gt_v_63_np.reshape(-1, 3)).reshape(B, num_predict_t, num_x, 3)
        
        # Calculate RMSE
        sq_err = np.sum((gt_denorm - pred_denorm)**2, axis=-1) # (B, P, 26)
        rmse_per_sample = np.sqrt(np.mean(sq_err, axis=(1, 2))) # (B,)
        all_rmse.extend(rmse_per_sample.tolist())
        
        # Explicit memory cleanup
        if device == "mps":
            torch.mps.empty_cache()
    
    return float(np.mean(all_rmse))

def evaluate_on_device(device, indices, data_path, has_originals):
    """Run evaluation on a specific device using a subset of data indices."""
    print(f"{Colors.BOLD}Starting evaluation on {Colors.CYAN}{device}{Colors.RESET} for {len(indices)} samples")
    
    try:
        transformer, ae = load_models(device)
        converter = FloatConverter()
    except Exception as e:
        print(f"{Colors.RED}Error loading models on {device}: {e}{Colors.RESET}")
        return None

    # Custom subset loader
    class IndexedEvalDataset(EvalDataset):
        def __init__(self, h5_path, indices):
            super().__init__(h5_path)
            self.indices = indices
            self.length = len(indices)

    dataset = IndexedEvalDataset(data_path, indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    stats_data = []
    staircase_data = []
    interleave_results = []
    detailed_reports = []
    NUM_DETAILED = 1 if device == "cpu" else 2 # Just to have some diversity
    total_samples_processed = 0

    with torch.no_grad():
        for batch_idx, (batch, originals_batch) in enumerate(tqdm(loader, desc=f"Eval {device}")):
            batch = batch.to(device)
            originals_batch = originals_batch.to(device)
            B = batch.shape[0]
            
            # 1. Standard Transformer Prediction
            outputs_list = []
            for i in range(0, B, Config.MICRO_BATCH_SIZE):
                micro_inputs = batch[i:i+Config.MICRO_BATCH_SIZE, :-1, :]
                micro_outputs = transformer(micro_inputs)
                outputs_list.append(micro_outputs)
                if device == "mps":
                    torch.mps.empty_cache()
            
            outputs = torch.cat(outputs_list, dim=0)
            
            # 2. Extract 80th time step predictions
            t80_target_indices = range(2054, 2080)
            t80_output_indices = [i-1 for i in t80_target_indices]
            
            pred_latents_t80 = outputs[:, t80_output_indices, :]
            gt_latents_t80 = batch[:, t80_target_indices, :Config.LATENT_DIM]
            
            coords_t80 = batch[:, t80_target_indices, 47:50]
            param_t80 = batch[:, t80_target_indices, 51]
            
            # 3. Decode Latents to Velocities
            pred_v_63_list = []
            gt_v_63_list = []
            
            for i in range(0, B, Config.MICRO_BATCH_SIZE):
                m_pred_latents = pred_latents_t80[i:i+Config.MICRO_BATCH_SIZE]
                m_gt_latents = gt_latents_t80[i:i+Config.MICRO_BATCH_SIZE]
                mB = m_pred_latents.shape[0]
                
                m_pred_latents_flat = m_pred_latents.reshape(-1, Config.LATENT_DIM)
                m_gt_latents_flat = m_gt_latents.reshape(-1, Config.LATENT_DIM)
                
                m_pred_velocities_full = ae.decode(m_pred_latents_flat)
                m_gt_velocities_full = ae.decode(m_gt_latents_flat)
                
                m_pred_v_63 = m_pred_velocities_full.reshape(mB, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
                m_gt_v_63 = m_gt_velocities_full.reshape(mB, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
                
                pred_v_63_list.append(m_pred_v_63)
                gt_v_63_list.append(m_gt_v_63)
                
                if device == "mps":
                    torch.mps.empty_cache()

            pred_v_63 = torch.cat(pred_v_63_list, dim=0)
            gt_v_63 = torch.cat(gt_v_63_list, dim=0)
            
            # 4. Denormalize
            pred_v_63_np = pred_v_63.cpu().numpy()
            pred_denorm_v = converter.unconvert(pred_v_63_np.reshape(-1, 3)).reshape(B, 26, 3)
            
            if has_originals:
                gt_denorm_v = originals_batch.cpu().numpy()
            else:
                gt_v_63_np = gt_v_63.cpu().numpy()
                gt_denorm_v = converter.unconvert(gt_v_63_np.reshape(-1, 3)).reshape(B, 26, 3)
                
            sq_errors = np.sum((gt_denorm_v - pred_denorm_v)**2, axis=2)
            params = param_t80[:, 0].cpu().numpy()
            
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
            for c in Config.STAIRCASE_CONTEXTS:
                p = 80 - c
                rmse = evaluate_permutation(transformer, ae, converter, batch, c, p, device=device)
                staircase_data.append({'context_time_steps': c, 'rmse': rmse})

            # --- New Permutations Evaluation ---
            for n in range(1, 41):
                c = 2*n - 1
                p = 1
                if c + p > 80: break
                rmse = evaluate_permutation(transformer, ae, converter, batch, c, p, device=device)
                interleave_results.append({'mode': 'interleave', 'c': c, 'p': p, 'rmse': rmse})
                if rmse > Config.RMSE_LIMIT: break

            p_jump = 2
            while p_jump < 80:
                rmse = evaluate_permutation(transformer, ae, converter, batch, 1, p_jump, device=device)
                interleave_results.append({'mode': 'jump_c1', 'c': 1, 'p': p_jump, 'rmse': rmse})
                if rmse > Config.RMSE_LIMIT: break
                p_jump *= 2

            for c_var in [2, 5, 10, 20]:
                for p_var in [1, 2, 5, 10, 20]:
                    if c_var + p_var > 80: break
                    rmse = evaluate_permutation(transformer, ae, converter, batch, c_var, p_var, device=device)
                    interleave_results.append({'mode': f'var_c{c_var}', 'c': c_var, 'p': p_var, 'rmse': rmse})
                    if rmse > Config.RMSE_LIMIT: break
            
            if batch_idx == 0:
                for i in range(min(B, NUM_DETAILED)):
                    sample_report = {
                        'sample_idx': total_samples_processed + i,
                        'param': params[i],
                        'y': coords_t80[i, 0, 1].item(),
                        'z': coords_t80[i, 0, 2].item(),
                        'positions': []
                    }
                    for j in range(26):
                        original_pos_idx = t80_target_indices[j]
                        target_label = "L4" if original_pos_idx in Config.TARGET_POSITIONS else \
                                       "L8" if original_pos_idx in Config.TARGET_POSITIONS_2 else \
                                       "L16" if original_pos_idx in Config.TARGET_POSITIONS_3 else "T80"
                        
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

    return {
        'stats_data': stats_data,
        'staircase_data': staircase_data,
        'interleave_results': interleave_results,
        'detailed_reports': detailed_reports,
        'total_samples_processed': total_samples_processed
    }

def main():
    # Big Rainbow Message
    msg = f"INTERLEAVED EVALUATION: CPU + {Config.DEVICE.upper()}"
    print("\n" + "="*80)
    print(Colors.rainbow(f"  {msg}  "))
    print("="*80 + "\n")
    
    # Load dataset to get indices
    try:
        data_path = Config.get_data_path()
        print(f"Using dataset: {Colors.YELLOW}{data_path}{Colors.RESET}")
        
        # Initial access to get total available and check for originals
        with h5py.File(data_path, 'r') as f:
            total_available = f['data'].shape[0]
            has_originals = 'originals' in f
            
        limit = min(Config.LIMIT_SAMPLES, total_available)
        all_indices = np.random.choice(total_available, limit, replace=False)
        all_indices.sort()
        
        # Split indices for CPU and GPU
        mid = len(all_indices) // 2
        gpu_indices = all_indices[:mid]
        cpu_indices = all_indices[mid:]
        
        print(f"Total samples: {len(all_indices)} (GPU: {len(gpu_indices)}, CPU: {len(cpu_indices)})")
    except Exception as e:
        print(f"{Colors.RED}Error preparing dataset: {e}{Colors.RESET}")
        return

    # Run evaluation in parallel
    results_list = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if len(gpu_indices) > 0:
            futures.append(executor.submit(evaluate_on_device, Config.DEVICE, gpu_indices, data_path, has_originals))
        if len(cpu_indices) > 0:
            futures.append(executor.submit(evaluate_on_device, 'cpu', cpu_indices, data_path, has_originals))
            
        for future in futures:
            res = future.result()
            if res:
                results_list.append(res)

    # Merge results
    stats_data = []
    staircase_data = []
    interleave_results = []
    detailed_reports = []
    total_samples_processed = 0
    
    for res in results_list:
        stats_data.extend(res['stats_data'])
        staircase_data.extend(res['staircase_data'])
        interleave_results.extend(res['interleave_results'])
        detailed_reports.extend(res['detailed_reports'])
        total_samples_processed += res['total_samples_processed']

    if total_samples_processed == 0:
        print(f"{Colors.RED}No samples were processed.{Colors.RESET}")
        return

    # --- Statistics Calculation ---
    print(f"\n{Colors.BOLD}CALCULATING SUMMARY STATISTICS...{Colors.RESET}")
    df = pd.DataFrame(stats_data)
    
    # RMSE per experiment
    rmse_per_param = df.groupby('param')['sq_error'].mean().apply(np.sqrt)
    
    # RMSE per position
    rmse_per_pos = df.groupby('pos_idx')['sq_error'].mean().apply(np.sqrt)

    # Staircase RMSE
    df_staircase = pd.DataFrame(staircase_data)
    rmse_staircase = df_staircase.groupby('context_time_steps')['rmse'].mean()
    
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

    # Aggregate Interleave Results
    df_interleave = pd.DataFrame(interleave_results)
    summary_interleave = df_interleave.groupby(['mode', 'c', 'p'])['rmse'].mean().reset_index()

    results = {
        'rmse_per_pos': rmse_per_pos.to_dict(),
        'rmse_staircase': rmse_staircase.to_dict(),
        'rmse_per_param': rmse_per_param.to_dict(),
        'yz_stats': yz_stats.to_dict(orient='records'),
        'rmse_l4': float(rmse_l4),
        'rmse_l8': float(rmse_l8),
        'rmse_l16': float(rmse_l16),
        'rmse_overall': float(rmse_overall),
        'detailed_reports': detailed_reports,
        'interleave_summary': summary_interleave.to_dict(orient='records')
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
        print(f"Given {k} time steps -> T80 RMSE: {Colors.CYAN}{val:.4e}{Colors.RESET}")
    print("-" * 50)
    print(f"RMSE per Experiment (First 10 Params):\n{rmse_per_param.head(10).apply(lambda x: f'{x:.4e}')}")
    print("-" * 50)

    print(f"\n{Colors.BOLD}DETAILED SAMPLES{Colors.RESET}")
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
