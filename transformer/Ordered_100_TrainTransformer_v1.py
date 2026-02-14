import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
import wandb

# Add current directory to path to import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformer_model_v1 import OrderedTransformerV1

# --- Configuration ---
class Config:
    # Data paths
    TRAIN_H5 = "/Users/kkreth/PycharmProjects/data/training_data.h5"
    VAL_H5 = "/Users/kkreth/PycharmProjects/data/validation_data.h5"
    
    # Model architecture
    LATENT_DIM = 47
    NUM_X = 26
    NUM_TIME = 8
    SEQ_LEN = NUM_X * NUM_TIME # 208
    
    INPUT_DIM = 52 # 47 latents + 3 xyz + 1 rel_time + 1 param
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    BIAS = True
    
    # Training
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Fast experimentation
    LIMIT_SAMPLES = 100000 # Set to an integer (e.g. 10000) to shrink the dataset
    
    # Target positions for focused evaluation (Last 4, 8, 16 positions in the last time period)
    TARGET_POSITIONS = list(range(SEQ_LEN - 4, SEQ_LEN)) 
    TARGET_POSITIONS_2 = list(range(SEQ_LEN - 8, SEQ_LEN))
    TARGET_POSITIONS_3 = list(range(SEQ_LEN - 16, SEQ_LEN))

# --- Dataset ---
class TransformerDataset(Dataset):
    def __init__(self, h5_path, max_samples=None):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_available = f['data'].shape[0]
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
        return torch.from_numpy(data).float()

# --- Training Setup ---
def train():
    """
    Main training loop.
    
    LOSS EXPLANATION:
    The "latent loss" (MSE) measures the model's ability to predict the next 47 latent features 
    in the sequence. Because we have flattened the 8x26 grid into a 208-token sequence,
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
        project="transformer_prepared_cubes",
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
    print(f"Loading training data from: {Config.TRAIN_H5}")
    print(f"Loading validation data from: {Config.VAL_H5}")
    
    # Datasets and Loaders
    try:
        train_dataset = TransformerDataset(Config.TRAIN_H5, max_samples=Config.LIMIT_SAMPLES)
        val_dataset = TransformerDataset(Config.VAL_H5, max_samples=Config.LIMIT_SAMPLES // 10 if Config.LIMIT_SAMPLES else None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = OrderedTransformerV1(Config).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    start_epoch = 0
    
    # --- Load Checkpoint ---
    model_save_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(model_save_dir, "best_ordered_transformer_v1.pt")
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
                print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch} (Best Val Loss: {best_val_loss:.6f})")
            else:
                # Fallback for legacy state_dict only files
                model.load_state_dict(checkpoint)
                print("Legacy state_dict loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    else:
        print("No existing checkpoint found. Starting from scratch.")
    
    # Track previous losses for coloring (Epoch level trend)
    prev_val_loss = None
    prev_target_loss = None
    prev_target2_loss = None
    prev_target3_loss = None
    
    # Helper for coloring based on trend
    def get_colored_str(val, prev_val, fmt=".6f"):
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
                'AvgAll': f"{loss.item():.5f}",
                'L4': get_colored_str(l4, last_batch_l4, ".5f"),
                'L8': get_colored_str(l8, last_batch_l8, ".5f"),
                'L16': get_colored_str(l16, last_batch_l16, ".5f")
            })
            
            last_batch_l4, last_batch_l8, last_batch_l16 = l4, l8, l16
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        target_pos_loss = 0 
        target_pos2_loss = 0
        target_pos3_loss = 0
        
        # Track loss per time step (1 to 8)
        # Note: predicting time t requires information from at least the start of time t or end of t-1
        time_step_losses = np.zeros(Config.NUM_TIME)
        
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
                # Time step t (0-7) corresponds to positions [t*26, (t+1)*26)
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
                
        avg_val_loss = val_loss / len(val_loader)
        avg_target_loss = target_pos_loss / len(val_loader)
        avg_target2_loss = target_pos2_loss / len(val_loader)
        avg_target3_loss = target_pos3_loss / len(val_loader)
        avg_time_step_losses = time_step_losses / len(val_loader)
        
        # Log to wandb
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "target_pos_loss": avg_target_loss,
            "target_pos2_loss": avg_target2_loss,
            "target_pos3_loss": avg_target3_loss
        }
        for t in range(Config.NUM_TIME):
            log_dict[f"val_loss_time_step_{t+1}"] = avg_time_step_losses[t]
            
        wandb.log(log_dict)
        
        print(f"Epoch {epoch+1}: Avg Train (All): {avg_train_loss:.6f}, Avg Val (All): {get_colored_str(avg_val_loss, prev_val_loss)}")
        print(f"  Target Losses -> Last4: {get_colored_str(avg_target_loss, prev_target_loss)}, "
              f"Last8: {get_colored_str(avg_target2_loss, prev_target2_loss)}, "
              f"Last16: {get_colored_str(avg_target3_loss, prev_target3_loss)}")
        
        # Update previous values for next epoch trend comparison
        prev_val_loss = avg_val_loss
        prev_target_loss = avg_target_loss
        prev_target2_loss = avg_target2_loss
        prev_target3_loss = avg_target3_loss

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_dir = os.path.dirname(os.path.abspath(__file__))
            model_save_path = os.path.join(model_save_dir, "best_ordered_transformer_v1.pt")
            
            # Create a comprehensive checkpoint (The "Standard" format)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)},
                'model': model # Embedding the model object itself for maximum compatibility
            }
            
            torch.save(checkpoint, model_save_path)
            wandb.save(model_save_path) # Also save model weights to wandb
            print(f"Saved best model checkpoint to {model_save_path}")

    wandb.finish()

if __name__ == "__main__":
    train()
