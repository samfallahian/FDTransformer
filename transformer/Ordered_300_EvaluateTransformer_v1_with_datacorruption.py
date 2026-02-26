import os
import sys
import torch
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from other modules
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
sys.path.insert(0, PROJECT_ROOT)

# Import model definitions
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
    NUM_TIME = 8
    SEQ_LEN = NUM_X * NUM_TIME # 208
    INPUT_DIM = 52
    
    # For reporting
    TRIPLET_IDX = 62 # 63rd triplet (0-indexed 62)
    
    # Fast evaluation for corruption test
    LIMIT_SAMPLES = 100 # Test at least 100 files to get good metrics

# --- Dataset ---
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, max_samples=None, random_seed=None):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_available = f['data'].shape[0]
            self.has_originals = 'originals' in f
            
            if max_samples is not None:
                self.length = min(max_samples, total_available)
                # Select random indices from the entire available set
                if random_seed is not None:
                    np.random.seed(random_seed)
                self.indices = np.random.choice(total_available, self.length, replace=False)
            else:
                self.length = total_available
                self.indices = np.arange(total_available)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        
        # Map the requested idx to our sampled indices
        real_idx = self.indices[idx]
        
        data = self._file['data'][real_idx] # (8, 26, 52)
        # Flatten time and space: (208, 52)
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        
        if self.has_originals:
            orig = self._file['originals'][real_idx] # (26, 3)
            return torch.from_numpy(data).float(), torch.from_numpy(orig).float()
            
        return torch.from_numpy(data).float(), torch.zeros((Config.NUM_X, 3))

def load_models():
    import torch
    # 1. Load Transformer
    print(f"Loading Transformer from: {Colors.CYAN}{Config.TRANSFORMER_CHECKPOINT}{Colors.RESET}")
    
    # We want to avoid using the potentially compiled model object directly if it causes issues.
    # Instead, we reconstruct it using the state_dict if possible.
    
    try:
        # Detected torch._dynamo mismatch, so let's try to fix it before loading.
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class Dummy: pass
            torch._dynamo.convert_frame.ConvertFrameBox = Dummy
    except ImportError:
        pass

    try:
        checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=Config.DEVICE, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise e

    # Reconstruct from config and state_dict is safer than using the serialized model object
    # if there are torch version/dynamo mismatches.
    if isinstance(checkpoint, dict) and 'config' in checkpoint and 'model_state_dict' in checkpoint:
        print("Reconstructing Transformer model from checkpoint config and state_dict...")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        
        # Strip '_orig_mod.' prefix if it exists (happens if model was compiled with torch.compile)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('_orig_mod.', '') 
            new_state_dict[name] = v
        
        transformer.load_state_dict(new_state_dict)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        print("Using the embedded model object...")
        transformer = checkpoint['model']
        # If it's a compiled model, try to get the original
        if hasattr(transformer, '_orig_mod'):
            print("Extracting original model from OptimizedModule...")
            transformer = transformer._orig_mod
    else:
        raise ValueError("Could not find model or model_state_dict in checkpoint")
        
    transformer.to(Config.DEVICE)
    transformer.eval() # MANDATORY EVAL MODE
        
    transformer.to(Config.DEVICE)
    transformer.eval() # MANDATORY EVAL MODE
    
    # 2. Load Encoder/Decoder
    print(f"Loading Scripted AE from: {Colors.CYAN}{Config.ENCODER_CHECKPOINT}{Colors.RESET}")
    ae = torch.jit.load(Config.ENCODER_CHECKPOINT, map_location=Config.DEVICE)
    ae.to(Config.DEVICE)
    ae.eval() # MANDATORY EVAL MODE
    
    return transformer, ae

def corrupt_data(data, corruption_fraction):
    """
    Corrupt the latent part of the input (first 47 dimensions) 
    by replacing a fraction of values with random values between -1 and 1.
    data: (B, T, 52) tensor
    """
    if corruption_fraction <= 0:
        return data.clone()
        
    corrupted_data = data.clone()
    B, T, C = corrupted_data.shape
    
    # We only corrupt the latent dimensions (0 to 46)
    num_elements_to_corrupt = int(corruption_fraction * B * T * Config.LATENT_DIM)
    
    if num_elements_to_corrupt > 0:
        # Generate random indices for corruption within the latent space
        # We'll flatten the B, T, LATENT_DIM indices
        flat_indices = torch.randperm(B * T * Config.LATENT_DIM)[:num_elements_to_corrupt]
        
        # Random values between -1 and 1
        # torch.rand gives [0, 1), so 2*torch.rand-1 gives [-1, 1)
        random_values = 2 * torch.rand(num_elements_to_corrupt, device=data.device) - 1
        
        # Reshape corrupted_data to easily apply indices
        # We only care about the first LATENT_DIM columns
        latents = corrupted_data[:, :, :Config.LATENT_DIM].reshape(-1)
        latents[flat_indices] = random_values
        corrupted_data[:, :, :Config.LATENT_DIM] = latents.reshape(B, T, Config.LATENT_DIM)
        
    return corrupted_data

def evaluate_at_corruption(transformer, ae, converter, loader, corruption_level):
    """
    Evaluates the model performance at a specific corruption level.
    """
    total_sq_error = 0
    total_count = 0
    
    with torch.no_grad():
        for batch, originals_batch in loader:
            batch = batch.to(Config.DEVICE)
            originals_batch = originals_batch.to(Config.DEVICE)
            B = batch.shape[0]
            
            # Apply corruption to the input (latents)
            corrupted_batch = corrupt_data(batch, corruption_level)
            
            # 1. Transformer Prediction using corrupted input
            # Predicted next latent at each position
            inputs = corrupted_batch[:, :-1, :]
            outputs = transformer(inputs) # (B, 207, Config.LATENT_DIM)
            
            # 2. Extract 8th time step predictions (positions 182-207)
            # The prediction for index i is at output index i-1
            t8_target_indices = range(182, 208)
            t8_output_indices = [i-1 for i in t8_target_indices]
            
            pred_latents_t8 = outputs[:, t8_output_indices, :] # (B, 26, 47)
            
            # 3. Decode Latents to Velocities
            pred_latents_flat = pred_latents_t8.reshape(-1, Config.LATENT_DIM)
            pred_velocities_full = ae.decode(pred_latents_flat) # (B*26, 375)
            
            # Reshape and extract 63rd triplet (Central Velocity)
            pred_v_63 = pred_velocities_full.reshape(B, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
            
            # 4. Denormalize
            pred_v_63_np = pred_v_63.cpu().numpy()
            pred_denorm_v = converter.unconvert(pred_v_63_np.reshape(-1, 3)).reshape(B, 26, 3)
            
            # Truth from originals_batch
            gt_denorm_v = originals_batch.cpu().numpy()
                
            # Calculate squared errors
            sq_errors = np.sum((gt_denorm_v - pred_denorm_v)**2, axis=2) # (B, 26)
            total_sq_error += np.sum(sq_errors)
            total_count += B * 26
            
    rmse = np.sqrt(total_sq_error / total_count)
    return rmse

def main():
    # Big Rainbow Message
    msg = f"EVALUATING MODEL PERFORMANCE UNDER DATA CORRUPTION - MODELS IN EVAL MODE"
    print("\n" + "="*80)
    print(Colors.rainbow(f"  {msg}  "))
    print("="*80 + "\n")
    
    # Load models
    try:
        transformer, ae = load_models()
        converter = FloatConverter()
    except Exception as e:
        print(f"{Colors.RED}Error loading models: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return

    # Load datasets
    try:
        data_path = Config.get_data_path()
        print(f"Using dataset: {Colors.YELLOW}{data_path}{Colors.RESET}")
        
        # Pass None to random_seed for truly different samples each run, 
        # or a fixed number for reproducibility.
        dataset = EvalDataset(data_path, max_samples=Config.LIMIT_SAMPLES, random_seed=None)
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    except Exception as e:
        print(f"{Colors.RED}Error loading dataset: {e}{Colors.RESET}")
        return

    print(f"Number of files (samples) being tested: {len(dataset)}")
    
    corruption_levels = np.linspace(0, 1.0, 101) # 0% to 100% in 1% increments
    results = []

    print(f"\nStarting evaluation across {len(corruption_levels)} corruption levels...")
    for level in tqdm(corruption_levels, desc="Corruption Sweep"):
        rmse = evaluate_at_corruption(transformer, ae, converter, loader, level)
        results.append({
            'corruption_percent': level * 100,
            'rmse': rmse
        })

    df_results = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = "/Users/kkreth/PycharmProjects/cgan/transformer/corruption_deterioration_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {Colors.CYAN}{csv_path}{Colors.RESET}")

    # Plotting deterioration
    plt.figure(figsize=(12, 7))
    plt.plot(df_results['corruption_percent'], df_results['rmse'], marker='o', markersize=4, linestyle='-', color='red', linewidth=2)
    plt.title('Model Performance Deterioration vs. Input Data Corruption', fontsize=14)
    plt.xlabel('Data Corruption (%)', fontsize=12)
    plt.ylabel('RMSE (Velocity Units)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Annotate baseline
    baseline_rmse = df_results.iloc[0]['rmse']
    plt.annotate(f'Baseline: {baseline_rmse:.4f}', 
                 xy=(0, baseline_rmse), 
                 xytext=(10, baseline_rmse + (df_results['rmse'].max() - baseline_rmse)*0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plot_path = "/Users/kkreth/PycharmProjects/cgan/transformer/corruption_deterioration_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {Colors.CYAN}{plot_path}{Colors.RESET}")
    
    # Final summary
    print(f"\n{Colors.BOLD}DETERIORATION SUMMARY{Colors.RESET}")
    print("-" * 40)
    print(f"0% Corruption (Baseline) RMSE: {Colors.GREEN}{baseline_rmse:.6f}{Colors.RESET}")
    print(f"50% Corruption RMSE:          {Colors.YELLOW}{df_results.iloc[50]['rmse']:.6f}{Colors.RESET}")
    print(f"100% Corruption RMSE:         {Colors.RED}{df_results.iloc[100]['rmse']:.6f}{Colors.RESET}")
    print("-" * 40)
    print(f"{Colors.GREEN}Evaluation complete!{Colors.RESET}")

if __name__ == "__main__":
    main()
