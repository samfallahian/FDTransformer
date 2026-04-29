import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
import wandb
import shutil
import glob
import argparse

# Add current directory to path to import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformer_model_v1 import OrderedTransformerV1
from transformer_config import add_config_arg, load_config, optional_int, resolve_path

# --- Configuration ---
def print_rainbow(text):
    """Prints text in rainbow colors to the console."""
    colors = [
        '\033[91m', # Red
        '\033[93m', # Yellow
        '\033[92m', # Green
        '\033[96m', # Cyan
        '\033[94m', # Blue
        '\033[95m'  # Magenta
    ]
    reset = '\033[0m'
    colored_text = "".join(colors[i % len(colors)] + char for i, char in enumerate(text))
    print(colored_text + reset)

def select_device(requested="auto"):
    requested = (requested or "auto").lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to MPS/CPU.")
        return "mps" if mps_available else "cpu"
    if requested == "mps" and not mps_available:
        print("MPS was requested but is not available. Falling back to CPU.")
        return "cpu"
    return requested

class Config:
    # Data paths
    TRAIN_H5 = None
    VAL_H5 = None
    CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_BASE_NAME = "best_ordered_transformer_v1"
    
    # Model architecture
    LATENT_DIM = 47
    NUM_X = 26
    NUM_TIME = 80
    SEQ_LEN = NUM_X * NUM_TIME # 2080
    
    INPUT_DIM = 52 # 47 latents + 3 xyz + 1 rel_time + 1 param
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    BIAS = True
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    MAX_CHECKPOINTS = 5
    STAIRCASE_EVAL_FREQ = 0 # How often to run staircase eval (0 to disable) - This is a VERY expensive computation.
    DEVICE = select_device("auto")
    NUM_WORKERS = 2
    WANDB_PROJECT = "transformer_OG_prepared_cubes"
    WANDB_MODE = "online"
    
    # Fast experimentation
    LIMIT_SAMPLES = None # Set to an integer to shrink the dataset for experiments.
    
    # Target positions for focused evaluation (Last 4, 8, 16 positions in the last time period)
    TARGET_POSITIONS = list(range(SEQ_LEN - 4, SEQ_LEN)) 
    TARGET_POSITIONS_2 = list(range(SEQ_LEN - 8, SEQ_LEN))
    TARGET_POSITIONS_3 = list(range(SEQ_LEN - 16, SEQ_LEN))

def refresh_derived_config():
    Config.SEQ_LEN = Config.NUM_X * Config.NUM_TIME
    Config.TARGET_POSITIONS = list(range(Config.SEQ_LEN - 4, Config.SEQ_LEN))
    Config.TARGET_POSITIONS_2 = list(range(Config.SEQ_LEN - 8, Config.SEQ_LEN))
    Config.TARGET_POSITIONS_3 = list(range(Config.SEQ_LEN - 16, Config.SEQ_LEN))

def configure(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]
    data = cfg["data"]
    model_cfg = cfg["model"]
    training = cfg["training"]

    Config.TRAIN_H5 = resolve_path(args.train_h5 or paths["training_h5"])
    Config.VAL_H5 = resolve_path(args.val_h5 or paths["validation_h5"])
    Config.CHECKPOINT_DIR = resolve_path(args.checkpoint_dir or paths["checkpoint_dir"])
    Config.CHECKPOINT_BASE_NAME = args.checkpoint_base_name or training["checkpoint_base_name"]

    Config.LATENT_DIM = data.get("latent_dim", Config.LATENT_DIM)
    Config.NUM_X = data.get("num_x", Config.NUM_X)
    Config.NUM_TIME = args.num_time if args.num_time is not None else data.get("num_time", Config.NUM_TIME)
    Config.INPUT_DIM = data.get("input_dim", Config.INPUT_DIM)

    Config.EMBED_SIZE = model_cfg.get("embed_size", Config.EMBED_SIZE)
    Config.N_HEADS = model_cfg.get("n_heads", Config.N_HEADS)
    Config.N_LAYERS = model_cfg.get("n_layers", Config.N_LAYERS)
    Config.DROPOUT = model_cfg.get("dropout", Config.DROPOUT)
    Config.BIAS = model_cfg.get("bias", Config.BIAS)

    Config.BATCH_SIZE = args.batch_size if args.batch_size is not None else training["batch_size"]
    Config.LEARNING_RATE = args.learning_rate if args.learning_rate is not None else training["learning_rate"]
    Config.EPOCHS = args.epochs if args.epochs is not None else training["epochs"]
    Config.MAX_CHECKPOINTS = args.max_checkpoints if args.max_checkpoints is not None else training["max_checkpoints"]
    Config.STAIRCASE_EVAL_FREQ = (
        args.staircase_eval_freq
        if args.staircase_eval_freq is not None
        else training["staircase_eval_freq"]
    )
    Config.NUM_WORKERS = args.num_workers if args.num_workers is not None else training["num_workers"]
    Config.LIMIT_SAMPLES = optional_int(args.limit_samples) if args.limit_samples is not None else optional_int(training.get("limit_samples"))
    Config.DEVICE = select_device(args.device or training.get("device", "auto"))
    Config.WANDB_PROJECT = args.wandb_project or training["wandb_project"]
    Config.WANDB_MODE = args.wandb_mode or training.get("wandb_mode", "online")
    refresh_derived_config()

# --- Dataset ---
class TransformerDataset(Dataset):
    def __init__(self, h5_path, max_samples=None):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            data_shape = f['data'].shape
            total_available = data_shape[0]
            sample_shape = data_shape[1:]
            if len(sample_shape) == 3:
                seq_len = sample_shape[0] * sample_shape[1]
            elif len(sample_shape) == 2:
                seq_len = sample_shape[0]
            else:
                raise ValueError(f"Unsupported sample layout {sample_shape}.")
            if seq_len != Config.SEQ_LEN:
                raise ValueError(
                    f"HDF5 sequence length is {seq_len}, but config expects {Config.SEQ_LEN}. "
                    f"Set --num-time {seq_len // Config.NUM_X} or update data.num_time in the config."
                )
            if max_samples is not None:
                self.length = min(max_samples, total_available)
                # Randomly pick indices from the available data if we are limiting samples
                # This ensures we get different samples on different runs, but keep them 
                # consistent across epochs in a single run.
                if self.length < total_available:
                    self.indices = np.random.choice(total_available, self.length, replace=False)
                else:
                    self.indices = None
            else:
                self.length = total_available
                self.indices = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        
        # If we have a subset of random indices, pick the actual index from that list
        fetch_idx = self.indices[idx] if self.indices is not None else idx
        data = self._file['data'][fetch_idx] # (NUM_TIME, NUM_X, INPUT_DIM)
        # Flatten time and space: (Config.SEQ_LEN, 52)
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        return torch.from_numpy(data).float()

# --- Training Setup ---
def train():
    """
    Main training loop.
    
    LOSS EXPLANATION:
    The "latent loss" (MSE) measures the model's ability to predict the next 47 latent features 
    in the sequence. Because we have flattened the configured time-by-space grid into one sequence,
    the model is effectively learning to:
    1. Predict the next spatial point within the same time step.
    2. Predict the first spatial point of the next time step (when at the end of a time row).
    
    We track:
    - overall_loss: MSE averaged across all 207 prediction steps and all 47 latent features.
    - target_pos_loss: MSE specifically for the last 4 positions of the last time step.
    - time_step_losses: MSE broken down by which time step the target belongs to.
    """
    # Initialize wandb
    wandb.init(
        project=Config.WANDB_PROJECT,
        mode=Config.WANDB_MODE,
        config={
            "learning_rate": Config.LEARNING_RATE,
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "embed_size": Config.EMBED_SIZE,
            "n_heads": Config.N_HEADS,
            "n_layers": Config.N_LAYERS,
            "dropout": Config.DROPOUT,
            "device": Config.DEVICE,
            "input_dim": Config.INPUT_DIM,
            "latent_dim": Config.LATENT_DIM
        }
    )
    
    # Save the model definition file to wandb
    model_def_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_model_v1.py")
    if os.path.exists(model_def_path):
        wandb.save(model_def_path)
        with open(model_def_path, 'r') as f:
            wandb.config.update({"model_definition_code": f.read()})
    
    print(f"Using device: {Config.DEVICE}")
    if Config.DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("CUDA detected and enabled for training.")
    print(f"Loading training data from: {Config.TRAIN_H5}")
    print(f"Loading validation data from: {Config.VAL_H5}")
    
    # Datasets and Loaders
    try:
        train_dataset = TransformerDataset(Config.TRAIN_H5, max_samples=Config.LIMIT_SAMPLES)
        val_limit = max(1, Config.LIMIT_SAMPLES // 10) if Config.LIMIT_SAMPLES else None
        val_dataset = TransformerDataset(Config.VAL_H5, max_samples=val_limit)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    loader_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": Config.DEVICE == "cuda",
    }
    if Config.NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    
    model = OrderedTransformerV1(Config).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # --- Load Checkpoint ---
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    model_save_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.CHECKPOINT_BASE_NAME}.pt")
    
    # Initialize best_checkpoints list: list of dicts {'val_loss', 'epoch', 'path'}
    best_checkpoints = []
    
    if os.path.exists(model_save_path):
        print(f"Loading existing checkpoint from: {model_save_path}")
        try:
            # Use weights_only=False to allow loading the full dictionary and embedded model object
            checkpoint = torch.load(model_save_path, map_location=Config.DEVICE, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4e})")
                
                # Add the loaded checkpoint to our best list
                # Ensure it has an epoch-named copy for management
                loaded_epoch_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.CHECKPOINT_BASE_NAME}_epoch_{start_epoch}.pt")
                if not os.path.exists(loaded_epoch_path):
                    shutil.copy2(model_save_path, loaded_epoch_path)
                
                best_checkpoints.append({
                    'val_loss': best_val_loss,
                    'epoch': start_epoch,
                    'path': loaded_epoch_path
                })
            else:
                # Fallback for legacy state_dict only files
                model.load_state_dict(checkpoint)
                print("Legacy state_dict loaded successfully.")
                # We don't have loss/epoch info for legacy, so we start fresh for best_checkpoints
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print("No existing checkpoint found. Starting from scratch.")
    
    # Also scan for any other existing epoch-named checkpoints to populate the top 5
    existing_checkpoints = glob.glob(os.path.join(Config.CHECKPOINT_DIR, f"{Config.CHECKPOINT_BASE_NAME}_epoch_*.pt"))
    for cp_path in existing_checkpoints:
        # Avoid adding the one we already added
        if any(cp['path'] == cp_path for cp in best_checkpoints):
            continue
        try:
            # Just load the metadata to get loss and epoch
            cp_data = torch.load(cp_path, map_location="cpu", weights_only=False)
            if isinstance(cp_data, dict) and 'best_val_loss' in cp_data:
                best_checkpoints.append({
                    'val_loss': cp_data['best_val_loss'],
                    'epoch': cp_data.get('epoch', 0),
                    'path': cp_path
                })
        except:
            pass
            
    # Keep only the top MAX_CHECKPOINTS and sort them
    best_checkpoints.sort(key=lambda x: x['val_loss'])
    if len(best_checkpoints) > Config.MAX_CHECKPOINTS:
        to_purge = best_checkpoints[Config.MAX_CHECKPOINTS:]
        best_checkpoints = best_checkpoints[:Config.MAX_CHECKPOINTS]
        for cp in to_purge:
            if os.path.exists(cp['path']):
                os.remove(cp['path'])
                print(f"Purged old checkpoint: {cp['path']}")
    
    if best_checkpoints:
        best_val_loss = best_checkpoints[0]['val_loss']
    
    # Track previous losses for coloring (Epoch level trend)
    prev_val_loss = None
    prev_target_loss = None
    prev_target2_loss = None
    prev_target3_loss = None
    prev_ar_losses = {k: None for k in [7, 6, 5, 4, 3, 2, 1]}
    
    # Helper for coloring based on trend
    def get_colored_str(val, prev_val, fmt=".4e"):
        val_str = f"{val:{fmt}}"
        if prev_val is None:
            return val_str
        # Green if improved (lower), Red if worse or same
        color = "\033[32m" if val < prev_val else "\033[31m"
        return f"{color}{val_str}\033[0m"

    # Pre-calculate indices for batch-level target loss calculation
    eval_indices_1 = [p - 1 for p in Config.TARGET_POSITIONS]
    eval_indices_2 = [p - 1 for p in Config.TARGET_POSITIONS_2]
    eval_indices_3 = [p - 1 for p in Config.TARGET_POSITIONS_3]

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        total_loss = 0
        
        # Track previous losses for batch-level trend coloring
        last_batch_l4 = None
        last_batch_l8 = None
        last_batch_l16 = None
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch in pbar:
            batch = batch.to(Config.DEVICE)
            
            # Input: tokens 0 to 206
            # Target: latents of tokens 1 to 207
            inputs = batch[:, :-1, :] 
            targets = batch[:, 1:, :Config.LATENT_DIM] 
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate target losses for the current batch to show in progress bar
            with torch.no_grad():
                l4 = criterion(outputs[:, eval_indices_1, :], targets[:, eval_indices_1, :]).item()
                l8 = criterion(outputs[:, eval_indices_2, :], targets[:, eval_indices_2, :]).item()
                l16 = criterion(outputs[:, eval_indices_3, :], targets[:, eval_indices_3, :]).item()
            
            pbar.set_postfix({
                'AvgAll': f"{loss.item():.2e}",
                'L4': get_colored_str(l4, last_batch_l4, ".2e"),
                'L8': get_colored_str(l8, last_batch_l8, ".2e"),
                'L16': get_colored_str(l16, last_batch_l16, ".2e")
            })
            
            last_batch_l4, last_batch_l8, last_batch_l16 = l4, l8, l16
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        target_pos_loss = 0 
        target_pos2_loss = 0
        target_pos3_loss = 0
        
        # Track loss per time step.
        # Note: predicting time t requires information from at least the start of time t or end of t-1.
        time_step_losses = np.zeros(Config.NUM_TIME)
        
        # Track autoregressive losses for the 8th time step starting from different context lengths
        ar_scenarios = [7, 6, 5, 4, 3, 2, 1]
        ar_total_losses = {k: 0.0 for k in ar_scenarios}
        
        # Determine if we should run staircase eval this epoch
        do_staircase = Config.STAIRCASE_EVAL_FREQ > 0 and (epoch + 1) % Config.STAIRCASE_EVAL_FREQ == 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(Config.DEVICE)
                inputs = batch[:, :-1, :]
                targets = batch[:, 1:, :Config.LATENT_DIM]
                
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Eval on target positions (Last 4, 8, 16)
                # output index i predicts target index i+1
                
                # Last 4
                eval_indices_1 = [p - 1 for p in Config.TARGET_POSITIONS]
                target_outputs_1 = outputs[:, eval_indices_1, :]
                target_targets_1 = targets[:, eval_indices_1, :]
                target_pos_loss += criterion(target_outputs_1, target_targets_1).item()
                
                # Last 8
                eval_indices_2 = [p - 1 for p in Config.TARGET_POSITIONS_2]
                target_outputs_2 = outputs[:, eval_indices_2, :]
                target_targets_2 = targets[:, eval_indices_2, :]
                target_pos2_loss += criterion(target_outputs_2, target_targets_2).item()
                
                # Last 16
                eval_indices_3 = [p - 1 for p in Config.TARGET_POSITIONS_3]
                target_outputs_3 = outputs[:, eval_indices_3, :]
                target_targets_3 = targets[:, eval_indices_3, :]
                target_pos3_loss += criterion(target_outputs_3, target_targets_3).item()
                
                # Eval per time step
                # Each time step has NUM_X (26) positions.
                # output[i] predicts target[i+1].
                # Time step t corresponds to positions [t*NUM_X, (t+1)*NUM_X)
                for t in range(Config.NUM_TIME):
                    t_start = t * Config.NUM_X
                    t_end = (t + 1) * Config.NUM_X
                    
                    # Adjust indices because outputs/targets are shifted by 1 relative to original batch
                    # original batch indices [0...207]
                    # targets indices [1...207] correspond to target indices [0...206] in 'targets' tensor
                    # So original index 'j' is at index 'j-1' in 'targets'
                    
                    target_indices_for_t = [j - 1 for j in range(t_start, t_end) if j > 0]
                    if target_indices_for_t:
                        t_out = outputs[:, target_indices_for_t, :]
                        t_tgt = targets[:, target_indices_for_t, :]
                        time_step_losses[t] += criterion(t_out, t_tgt).item()
                
                # --- Autoregressive evaluation for predicting the 8th time slice ---
                # We start predicting from the end of step k and evaluate on the 8th step.
                if do_staircase:
                    for k in ar_scenarios:
                        num_to_predict = (Config.NUM_TIME - k) * Config.NUM_X
                        # Autoregressively predict until the end of the sequence
                        pred_latents = model.predict_autoregressive(batch, num_to_predict)
                        
                        # Evaluation is strictly on the 8th time step (the last 26 positions)
                        step8_pred = pred_latents[:, -Config.NUM_X:, :]
                        step8_target = batch[:, -Config.NUM_X:, :Config.LATENT_DIM]
                        ar_total_losses[k] += criterion(step8_pred, step8_target).item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_target_loss = target_pos_loss / len(val_loader)
        avg_target2_loss = target_pos2_loss / len(val_loader)
        avg_target3_loss = target_pos3_loss / len(val_loader)
        avg_time_step_losses = time_step_losses / len(val_loader)
        avg_ar_losses = {k: ar_total_losses[k] / len(val_loader) for k in ar_scenarios}
        
        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "target_pos_loss": avg_target_loss,
            "target_pos2_loss": avg_target2_loss,
            "target_pos3_loss": avg_target3_loss,
            # Scenario 1: Even steps (2, 4, 6, 8)
            "val_loss_even_steps": np.mean(avg_time_step_losses[1::2]),
            # Scenario 2: final step only
            "val_loss_last_step": avg_time_step_losses[-1],
            # Legacy metric name kept for older dashboards.
            "val_loss_8th_step": avg_time_step_losses[min(7, Config.NUM_TIME - 1)]
        }
        if do_staircase:
            for k, loss_val in avg_ar_losses.items():
                log_dict[f"ar_loss_T8_given_T1-{k}"] = loss_val
            
        for t in range(Config.NUM_TIME):
            log_dict[f"val_loss_time_step_{t+1}"] = avg_time_step_losses[t]
            
        wandb.log(log_dict)
        
        print(f"Epoch {epoch+1}: Avg Train (All): {avg_train_loss:.4e}, Avg Val (All): {get_colored_str(avg_val_loss, prev_val_loss)}")
        print(f"  Target Losses -> Last4: {get_colored_str(avg_target_loss, prev_target_loss)}, "
              f"Last8: {get_colored_str(avg_target2_loss, prev_target2_loss)}, "
              f"Last16: {get_colored_str(avg_target3_loss, prev_target3_loss)}")
        print(f"  Queries -> Even Steps: {log_dict['val_loss_even_steps']:.4e}, Last Step: {log_dict['val_loss_last_step']:.4e}")
        
        # --- Creative Autoregressive Results Display ---
        if do_staircase:
            print("\n\t" + "╔══════════════════════════════════════════════════════════╗")
            print("\t" + "║      AUTOREGRESSIVE T8 PREDICTION (STAIRCASE EVAL)       ║")
            print("\t" + "╟──────────────────────────────────────────────────────────╢")
            for i, k in enumerate(ar_scenarios):
                indent = "  " * i
                label = f"Given T1-{k}:"
                color_val = get_colored_str(avg_ar_losses[k], prev_ar_losses[k], ".4e")
                # Calculate how many spaces to add to reach the right border
                # label is 12 chars, indent is 2*i chars, color_val visible is 10 chars.
                # Total visible: 2*i + 12 + 1 + 10 = 23 + 2*i.
                # Max i=6 -> 23 + 12 = 35. 
                # We want to fill up to 58 (width of box inside).
                padding = " " * (58 - (len(indent) + len(label) + 1 + 10))
                print(f"\t║ {indent}{label} {color_val}{padding} ║")
            print("\t" + "╚══════════════════════════════════════════════════════════╝\n")
        
        # Update previous values for next epoch trend comparison
        prev_val_loss = avg_val_loss
        prev_target_loss = avg_target_loss
        prev_target2_loss = avg_target2_loss
        prev_target3_loss = avg_target3_loss
        if do_staircase:
            for k in ar_scenarios:
                prev_ar_losses[k] = avg_ar_losses[k]

        # Management of top 5 checkpoints
        if len(best_checkpoints) < Config.MAX_CHECKPOINTS or avg_val_loss < best_checkpoints[-1]['val_loss']:
            new_path = os.path.join(Config.CHECKPOINT_DIR, f"{Config.CHECKPOINT_BASE_NAME}_epoch_{epoch+1}.pt")
            
            # Create a comprehensive checkpoint (The "Standard" format)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': avg_val_loss,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)},
                'model': model # Embedding the model object itself for maximum compatibility
            }
            
            torch.save(checkpoint, new_path)
            wandb.save(new_path)
            
            best_checkpoints.append({
                'val_loss': avg_val_loss,
                'epoch': epoch + 1,
                'path': new_path
            })
            best_checkpoints.sort(key=lambda x: x['val_loss'])
            
            # Purge the worst if it fell out of the top 5
            if len(best_checkpoints) > Config.MAX_CHECKPOINTS:
                worst = best_checkpoints.pop()
                if os.path.exists(worst['path']):
                    os.remove(worst['path'])
                    # Try to purge from wandb
                    try:
                        api = wandb.Api()
                        # wandb.run might be None if not initialized, but here it is
                        run_path = f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}"
                        run = api.run(run_path)
                        filename = os.path.basename(worst['path'])
                        file = run.file(filename)
                        file.delete()
                    except Exception as e:
                        # Silently fail if wandb deletion fails (e.g. file not yet uploaded)
                        pass
                print(f"Purged checkpoint rank {Config.MAX_CHECKPOINTS+1}: {worst['path']}")

            # If this is the new absolute best (Rank 1), also update the standard checkpoint file
            if best_checkpoints[0]['val_loss'] == avg_val_loss:
                best_val_loss = avg_val_loss
                shutil.copy2(new_path, model_save_path)
                wandb.save(model_save_path) # Update the main best checkpoint on wandb too
                print_rainbow(f"New Rank 1 Best Val Loss: {best_val_loss:.4e}. Saved to {model_save_path}")
            else:
                print(f"Saved new Rank {best_checkpoints.index(next(cp for cp in best_checkpoints if cp['epoch'] == epoch+1)) + 1} checkpoint to {new_path}")

    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser(description="Train the ordered transformer.")
    add_config_arg(parser)
    parser.add_argument("--train_h5", "--train-h5", dest="train_h5", default=None, help="Training HDF5 path.")
    parser.add_argument("--val_h5", "--val-h5", dest="val_h5", default=None, help="Validation HDF5 path.")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", dest="checkpoint_dir", default=None, help="Directory for checkpoints.")
    parser.add_argument("--checkpoint_base_name", "--checkpoint-base-name", dest="checkpoint_base_name", default=None, help="Checkpoint filename stem.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_checkpoints", "--max-checkpoints", dest="max_checkpoints", type=int, default=None)
    parser.add_argument("--staircase_eval_freq", "--staircase-eval-freq", dest="staircase_eval_freq", type=int, default=None)
    parser.add_argument("--limit_samples", "--limit-samples", dest="limit_samples", default=None, help="Limit samples for quick runs. Use none/all/0 for full dataset.")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--num_time", "--num-time", dest="num_time", type=int, default=None, help="Number of time steps per sample.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--wandb_project", "--wandb-project", dest="wandb_project", default=None)
    parser.add_argument("--wandb_mode", "--wandb-mode", dest="wandb_mode", default=None, help="wandb mode, e.g. online, offline, disabled.")
    return parser.parse_args()

if __name__ == "__main__":
    configure(parse_args())
    train()
