import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import sys
import time
from model_variants import get_model

# --- Configuration ---
class Config:
    # Data paths
    TRAIN_H5 = os.path.join(os.path.dirname(__file__), "data/train_40.h5")
    VAL_H5 = os.path.join(os.path.dirname(__file__), "data/val_40.h5")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
    
    # Model architecture
    LATENT_DIM = 47
    NUM_X = 26
    NUM_TIME = 40
    SEQ_LEN = NUM_X * NUM_TIME # 1040
    
    INPUT_DIM = 52
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    BIAS = True
    VARIANT = 'base' # 'base' or 'swiglu'
    
    # Training
    BATCH_SIZE = 2 # Keeping small for search
    LEARNING_RATE = 1e-3 # Keeping high for search
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Robustness techniques
    NOISE_STD = 5e-4   # Noise injected to latent portion of input
    AR_ROLLOUT_STEPS = 10 # Predict next 10 tokens (roughly 1/3 of a time step)
    AR_LOSS_WEIGHT = 0.05
    
    # Evaluation
    VAL_CONTEXT_STEPS = 12
    VAL_ROLLOUT_STEPS = 26 * (40 - VAL_CONTEXT_STEPS) # Predict remaining steps (28 * 26 = 728 tokens)

    # Architecture search space - Exploring diverse architectural inductive biases
    SEARCH_SPACE = [
        # 0: Baseline - Standard capacity
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "base"},
        
        # 1: Expressive Gating - SwiGLU activation (Llama-style)
        # Better for complex non-linear transitions in fluid dynamics
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "swiglu"},
        
        # 2: Efficiency - Multi-Query Attention (MQA)
        # Uses shared Key/Value across heads; focuses attention on global features
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "mqa"},
        
        # 3: Hybrid - Convolutional-Transformer
        # Uses 1D Conv before attention to capture local spatial/temporal correlations
        # Ideal for fluid dynamics where local neighborhood matters
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "conv"},
        
        # 4: Large Capacity Baseline
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 8, "VARIANT": "base"},
        
        # 5: Hybrid Wide - Conv + SwiGLU
        # Combining local inductive bias with advanced gating
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 4, "VARIANT": "conv", "USE_SWIGLU": True},
    ]

    # --- Runtime logic ---
    MAX_RUNTIME_PER_CANDIDATE = 60 # seconds

def l2_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

class TransformerDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['data'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        data = self._file['data'][idx]
        data = data.reshape(Config.SEQ_LEN, Config.INPUT_DIM)
        return torch.from_numpy(data).float()

def train(variant_idx=None):
    start_time = time.time()
    max_runtime = 60 # 60 seconds limit for candidate check
    if variant_idx is not None:
        variant_cfg = Config.SEARCH_SPACE[variant_idx]
        Config.EMBED_SIZE = variant_cfg["EMBED_SIZE"]
        Config.N_HEADS = variant_cfg["N_HEADS"]
        Config.N_LAYERS = variant_cfg["N_LAYERS"]
        Config.VARIANT = variant_cfg["VARIANT"]
    
    variant = Config.VARIANT
    run_name = f"production_{variant}_E{Config.EMBED_SIZE}_L{Config.N_LAYERS}_{int(time.time())}"
    wandb.init(project="transformer_neurIPS_production", name=run_name, config={
        "variant": variant,
        "embed_size": Config.EMBED_SIZE,
        "n_layers": Config.N_LAYERS,
        "n_heads": Config.N_HEADS,
        "noise_std": Config.NOISE_STD,
        "ar_steps": Config.AR_ROLLOUT_STEPS,
        "ar_weight": Config.AR_LOSS_WEIGHT,
        "val_context_steps": Config.VAL_CONTEXT_STEPS,
        "val_rollout_steps": Config.VAL_ROLLOUT_STEPS
    })
    
    train_dataset = TransformerDataset(Config.TRAIN_H5)
    val_dataset = TransformerDataset(Config.VAL_H5)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = get_model(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    best_rollout_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch in pbar:
            if time.time() - start_time > max_runtime:
                print(f"\nReached {max_runtime}s limit. Moving to next candidate.")
                wandb.finish()
                return

            batch = batch.to(Config.DEVICE)
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :Config.LATENT_DIM]
            
            # Noise injection
            if Config.NOISE_STD > 0:
                noise = torch.zeros_like(inputs)
                noise[:, :, :Config.LATENT_DIM] = torch.randn_like(inputs[:, :, :Config.LATENT_DIM]) * Config.NOISE_STD
                inputs = inputs + noise
            
            # Primary Teacher-Forced Pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l2_loss(outputs, targets)
            
            # --- Optional Short AR Rollout Loss ---
            if Config.AR_ROLLOUT_STEPS > 0 and Config.AR_LOSS_WEIGHT > 0:
                # We do a very short rollout to penalize error accumulation
                # To keep it differentiable and fast, we only do a few tokens
                # but following the pattern in the v2 script:
                # Using a shorter context for the AR pass to save memory/time
                context_len = min(inputs.shape[1], 256) 
                ar_context = inputs[:, :context_len, :].clone()
                ar_targets = targets[:, context_len : context_len + Config.AR_ROLLOUT_STEPS, :]
                
                curr = ar_context
                ar_preds = []
                for _ in range(Config.AR_ROLLOUT_STEPS):
                    # We must NOT use no_grad here because we want to backprop through the rollout
                    out = model(curr)
                    next_lat = out[:, -1:, :]
                    ar_preds.append(next_lat)
                    
                    # Construct next input token (pred latent + true metadata from next step)
                    # Use absolute index in 'inputs' to get metadata
                    next_idx = curr.shape[1]
                    if next_idx >= inputs.shape[1]: break
                    
                    next_tok = inputs[:, next_idx : next_idx + 1, :].clone()
                    next_tok[:, :, :Config.LATENT_DIM] = next_lat
                    curr = torch.cat([curr, next_tok], dim=1)
                
                if len(ar_preds) == Config.AR_ROLLOUT_STEPS:
                    ar_preds = torch.cat(ar_preds, dim=1)
                    ar_loss = l2_loss(ar_preds, ar_targets)
                    loss = loss + Config.AR_LOSS_WEIGHT * ar_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'l2': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(Config.DEVICE)
                inputs = batch[:, :-1, :]
                targets = batch[:, 1:, :Config.LATENT_DIM]
                outputs = model(inputs)
                val_loss += l2_loss(outputs, targets).item()
        val_loss /= len(val_loader)
        
        # --- Multi-step Rollout Evaluation ---
        model.eval()
        rollout_loss = 0
        persistence_loss = 0
        rollout_count = 0
        with torch.no_grad():
            # Use a smaller subset for speed if needed, or full val set
            for i, batch in enumerate(val_loader):
                if i >= 10: break # Only check first 10 batches (80 sequences) for speed
                batch = batch.to(Config.DEVICE)
                
                # Context is first 12 full time steps
                context_len = 26 * Config.VAL_CONTEXT_STEPS
                inputs = batch[:, :context_len, :]
                targets = batch[:, context_len : context_len + Config.VAL_ROLLOUT_STEPS, :Config.LATENT_DIM]
                
                # --- Persistence Baseline ---
                # Take the last frame of the context (the 12th time step)
                # Each time step has 26 tokens (one for each x-coordinate)
                last_frame = inputs[:, -26:, :Config.LATENT_DIM] # shape (B, 26, 47)
                # Repeat this frame to match the target length
                num_repeats = Config.VAL_ROLLOUT_STEPS // 26
                persistence_preds = last_frame.repeat(1, num_repeats, 1)
                persistence_loss += l2_loss(persistence_preds, targets).item()

                # --- Model Rollout ---
                curr = inputs
                preds = []
                for _ in range(Config.VAL_ROLLOUT_STEPS):
                    out = model(curr)
                    next_lat = out[:, -1:, :]
                    preds.append(next_lat)
                    
                    next_idx = curr.shape[1]
                    if next_idx >= batch.shape[1]: break
                    
                    next_tok = batch[:, next_idx : next_idx + 1, :].clone()
                    next_tok[:, :, :Config.LATENT_DIM] = next_lat
                    curr = torch.cat([curr, next_tok], dim=1)
                
                if len(preds) == Config.VAL_ROLLOUT_STEPS:
                    preds = torch.cat(preds, dim=1)
                    rollout_loss += l2_loss(preds, targets).item()
                    rollout_count += 1
        
        if rollout_count > 0:
            rollout_loss /= rollout_count
            persistence_loss /= rollout_count
        
        persistence_improvement = (persistence_loss - rollout_loss) / (persistence_loss + 1e-8) * 100

        wandb.log({
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "rollout_l2": rollout_loss,
            "persistence_l2": persistence_loss,
            "persistence_improvement_pct": persistence_improvement,
            "epoch": epoch
        })
        print(f"Epoch {epoch+1}: Train L2={train_loss:.6f}, Val L2={val_loss:.6f}")
        print(f"         Rollout L2={rollout_loss:.6f}, Persistence L2={persistence_loss:.6f} ({persistence_improvement:.1f}% better)")
        
        # Save best model based on rollout performance as it's the "most concerning metric"
        if rollout_loss < best_rollout_loss and rollout_loss > 0:
            best_rollout_loss = rollout_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_rollout_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best rollout model!")

        # --- Early exit for search ---
        if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
            print(f"60s limit reached for {run_name}. Moving to next candidate.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pt")
            torch.save(model.state_dict(), save_path)
            
    wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "search":
            print("Starting architecture search with 60s per candidate...")
            for i in range(len(Config.SEARCH_SPACE)):
                print(f"\n--- Testing Candidate {i} ---")
                train(i)
        elif sys.argv[1].isdigit():
            train(int(sys.argv[1]))
        else:
            Config.VARIANT = sys.argv[1]
            train()
    else:
        # Default to searching through all candidates
        print("Starting architecture search with 60s per candidate...")
        for i in range(len(Config.SEARCH_SPACE)):
            print(f"\n--- Testing Candidate {i} ---")
            train(i)
