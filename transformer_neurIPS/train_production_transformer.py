"""
Transformer Training Production Script for NeurIPS
=================================================
This script handles the training and architecture search for fluid dynamics latent sequences.

KEY METRICS TO MONITOR:
1. Rollout L2: Multi-step autoregressive prediction error (28 full time steps).
2. Persistence L2: The baseline "do-nothing" error (assuming t=12 is the result for all future t).
3. Persistence Improvement %: How much BETTER the model is than doing nothing. 
   Goal: MUST be > 0% and ideally > 50%.

WHERE TO LOOK:
- Search results: Console table at the end of the run.
- Telemetry: W&B project "transformer_neurIPS_production".
- Best Models: Saved in 'saved_models/' with '_rollout_best.pt' suffix.
"""

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
    """
    Global configuration for training and model parameters.
    Modify SEARCH_SPACE to test new architectures.
    """
    # Data paths - Absolute paths derived from file location
    TRAIN_H5 = os.path.join(os.path.dirname(__file__), "data/train_40.h5")
    VAL_H5 = os.path.join(os.path.dirname(__file__), "data/val_40.h5")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
    
    # Model architecture constants
    LATENT_DIM = 47 # Latent features from encoder
    NUM_X = 26      # Spatial locations per time step
    NUM_TIME = 40   # Total time steps in sequence
    SEQ_LEN = NUM_X * NUM_TIME # 1040 tokens total
    
    INPUT_DIM = 52  # 47 (latents) + 1 (time) + 1 (x) + 3 (params)
    
    # Defaults for single run (overwritten by SEARCH_SPACE if searching)
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.01
    BIAS = True
    VARIANT = 'base' 
    
    # Training Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    BATCH_SIZE = 128 if DEVICE == "cuda" else 32 # Higher on CUDA, safer on MPS
    ACCUMULATION_STEPS = 1 if DEVICE == "cuda" else 4 # Micro-batching for memory-constrained devices
    LEARNING_RATE = 2e-3 # Slightly higher LR to match larger batch size
    EPOCHS = 10
    NUM_WORKERS = 8 if DEVICE == "cuda" else 0 # Multi-threaded only on CUDA
    PIN_MEMORY = True if DEVICE == "cuda" else False

    # Stability & Robustness Techniques
    # Micro-batching setup: BATCH_SIZE is the per-step batch, 
    # effective batch size = BATCH_SIZE * ACCUMULATION_STEPS.
    # To "back out" or disable micro-batches, set ACCUMULATION_STEPS = 1.
    NOISE_STD = 5e-4       # Gaussian noise on inputs to fight AR drift
    AR_ROLLOUT_STEPS = 10 if DEVICE == "cuda" else 0 # Disabled on MPS to speed up training
    AR_LOSS_WEIGHT = 0.05
    
    # Persistence Baseline Evaluation Parameters
    VAL_CONTEXT_STEPS = 12 # Feed first 12 steps as context
    VAL_ROLLOUT_STEPS = 26 * (40 - VAL_CONTEXT_STEPS) # Predict remaining 28 steps (728 tokens)
    
    # Validation settings for speed on MPS
    VAL_INTERVAL = 1 if DEVICE == "cuda" else 999999 # Validate every epoch on CUDA, disabled on MPS
    
    # METRIC DEFINITION:
    # Persistence MSE = MSE(Target_Steps_13_to_40, Step_12_Repeated)
    # Model MSE = MSE(Target_Steps_13_to_40, Model_Predictions_13_to_40)
    # Goal: Model MSE < Persistence MSE

    # Architecture search space - Exploring diverse architectural inductive biases
    SEARCH_SPACE = [
        # 0: Baseline - Standard capacity (Strongest early performer)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "base"},
        
        # 1: Expressive Gating - SwiGLU activation (Llama-style, High throughput)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "swiglu"},
        
        # 2: Efficiency - Multi-Query Attention (MQA)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "mqa"},
        
        # 3: Hybrid - Convolutional-Transformer (Local spatial inductive bias)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "conv"},
        
        # 4: Deep Capacity Baseline
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 8, "VARIANT": "base"},
        
        # 5: Hybrid Wide - Conv + SwiGLU (Combining local bias with advanced gating)
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 4, "VARIANT": "conv", "USE_SWIGLU": True},
    ]

    # --- Runtime logic ---
    MAX_RUNTIME_PER_CANDIDATE = 1800 # 30 minutes for candidates (speedup should allow more training)
    CHECKPOINT_INTERVAL = 60 # Save every 60 seconds
    
    # H200 Optimization Recommendations:
    # 1. Use torch.compile(model) for Blackwell/Hopper speedups.
    # 2. Use torch.cuda.amp.autocast() or FP8 TransformerEngine if available.
    # 3. Increase BATCH_SIZE to 512+ to saturate HBM3e bandwidth.
    # 4. Use FlashAttention-3 kernels.

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def l2_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

class TransformerDataset(Dataset):
    def __init__(self, h5_path, subset_ratio=1.0):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_length = f['data'].shape[0]
            self.length = int(total_length * subset_ratio)
            if self.length == 0 and total_length > 0:
                self.length = 1

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
    if variant_idx is not None:
        variant_cfg = Config.SEARCH_SPACE[variant_idx]
        Config.EMBED_SIZE = variant_cfg["EMBED_SIZE"]
        Config.N_HEADS = variant_cfg["N_HEADS"]
        Config.N_LAYERS = variant_cfg["N_LAYERS"]
        Config.VARIANT = variant_cfg["VARIANT"]
    
    variant = Config.VARIANT
    # Stable run name for resumption (excludes timestamp if searching or top3)
    if variant_idx is not None:
        run_name = f"production_{variant}_E{Config.EMBED_SIZE}_L{Config.N_LAYERS}"
    else:
        run_name = f"production_{variant}_E{Config.EMBED_SIZE}_L{Config.N_LAYERS}_{int(time.time())}"
    
    wandb.init(project="transformer_neurIPS_production", name=run_name, id=run_name, resume="allow", config={
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
    
    train_dataset = TransformerDataset(Config.TRAIN_H5, subset_ratio=0.01)
    val_dataset = TransformerDataset(Config.VAL_H5, subset_ratio=0.01)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=Config.PIN_MEMORY
    )
    
    model = get_model(Config).to(Config.DEVICE)
    # Use torch.compile for faster execution on supported devices
    if hasattr(torch, "compile") and Config.DEVICE == "cuda":
        try:
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    best_val_loss = float('inf')
    best_train_l2 = float('inf')
    best_rollout_loss = float('inf')
    best_improvement = -float('inf')
    last_checkpoint_time = time.time()
    start_epoch = 0

    # --- Checkpoint Resumption ---
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    latest_cp_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")
    if os.path.exists(latest_cp_path):
        print(f"Resuming from checkpoint: {latest_cp_path}")
        try:
            checkpoint = torch.load(latest_cp_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            # If the checkpoint was saved at the end of an epoch, we start from the next one.
            if 'batch_idx' not in checkpoint or checkpoint['batch_idx'] >= len(train_loader) - 1:
                start_epoch += 1

            print(f"  -> Resuming from Epoch {start_epoch + 1}")
            
            # Restore best metrics if available in checkpoint
            if 'val_l2' in checkpoint: best_val_loss = checkpoint['val_l2']
            if 'rollout_mse' in checkpoint: best_rollout_loss = checkpoint['rollout_mse']
            if 'improvement' in checkpoint: best_improvement = checkpoint['improvement']
        except Exception as e:
            print(f"  -> Warning: Could not load checkpoint: {e}")

    if start_epoch >= Config.EPOCHS:
        print(f"Training already completed up to {start_epoch} epochs. Target is {Config.EPOCHS}. Skipping training.")
        wandb.finish()
        return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

    # OneCycleLR for faster convergence
    # Adjust steps_per_epoch for gradient accumulation
    effective_steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    if len(train_loader) % Config.ACCUMULATION_STEPS != 0:
        effective_steps_per_epoch += 1
    
    # Ensure total_steps is at least 10 to avoid ZeroDivisionError with small pct_start
    total_steps = effective_steps_per_epoch * Config.EPOCHS
    if total_steps < 10:
        pct_start = 0.3 # Higher pct_start for very short runs
    else:
        pct_start = 0.1

    try:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=Config.LEARNING_RATE, 
            steps_per_epoch=effective_steps_per_epoch, 
            epochs=Config.EPOCHS,
            pct_start=pct_start,
            last_epoch=start_epoch * effective_steps_per_epoch - 1 if start_epoch > 0 else -1
        )
    except Exception as e:
        print(f"  -> Warning: Could not initialize OneCycleLR: {e}. Falling back to constant LR.")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(device='cuda', enabled=(Config.DEVICE == "cuda"))

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0
        val_loss = float('inf') # Initialize to avoid UnboundLocalError when validation is skipped
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
                print(f"\nReached {Config.MAX_RUNTIME_PER_CANDIDATE}s limit. Moving to next candidate.")
                wandb.finish()
                return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

            batch = batch.to(Config.DEVICE)
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :Config.LATENT_DIM]
            
            # Noise injection
            if Config.NOISE_STD > 0:
                noise = torch.zeros_like(inputs)
                noise[:, :, :Config.LATENT_DIM] = torch.randn_like(inputs[:, :, :Config.LATENT_DIM]) * Config.NOISE_STD
                inputs = inputs + noise
            
            # Primary Teacher-Forced Pass
            with torch.amp.autocast(device_type=('cuda' if 'cuda' in Config.DEVICE else 'cpu'), enabled=('cuda' in Config.DEVICE or 'mps' in Config.DEVICE)):
                outputs = model(inputs)
                loss = l2_loss(outputs, targets)
                
                # --- Step-Level Baseline Check ---
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        prev_frame = inputs[:, -1:, :Config.LATENT_DIM]
                        step_target = targets[:, -1:, :]
                        step_persistence_mse = mse_loss(prev_frame, step_target).item()
                        step_model_mse = mse_loss(outputs[:, -1:, :], step_target).item()

                # --- Optional Short AR Rollout Loss ---
                if Config.AR_ROLLOUT_STEPS > 0 and Config.AR_LOSS_WEIGHT > 0:
                    context_len = min(inputs.shape[1], 256) 
                    ar_context = inputs[:, :context_len, :].clone()
                    ar_targets = targets[:, context_len : context_len + Config.AR_ROLLOUT_STEPS, :]
                    
                    curr = ar_context
                    ar_preds = []
                    for _ in range(Config.AR_ROLLOUT_STEPS):
                        out = model(curr)
                        next_lat = out[:, -1:, :]
                        ar_preds.append(next_lat)
                        
                        next_idx = curr.shape[1]
                        if next_idx >= inputs.shape[1]: break
                        
                        next_tok = inputs[:, next_idx : next_idx + 1, :].clone()
                        next_tok[:, :, :Config.LATENT_DIM] = next_lat
                        curr = torch.cat([curr, next_tok], dim=1)
                    
                    if len(ar_preds) == Config.AR_ROLLOUT_STEPS:
                        ar_preds = torch.cat(ar_preds, dim=1)
                        ar_loss = l2_loss(ar_preds, ar_targets)
                        loss = loss + Config.AR_LOSS_WEIGHT * ar_loss
                
                # Normalize loss for accumulation
                loss = loss / Config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * Config.ACCUMULATION_STEPS

            # Logging step-level metrics
            pbar.set_postfix({'l2': loss.item()})
            if batch_idx % 50 == 0:
                wandb.log({
                    "step_train_l2": loss.item(),
                    "step_persistence_mse": step_persistence_mse,
                    "step_model_mse": step_model_mse,
                    "step_improvement_pct": (step_persistence_mse - step_model_mse) / (step_persistence_mse + 1e-8) * 100,
                    "lr": scheduler.get_last_lr()[0]
                })
            
            # --- Periodic Checkpointing (Every Minute) ---
            if time.time() - last_checkpoint_time > Config.CHECKPOINT_INTERVAL:
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                cp_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_l2': best_val_loss,
                    'rollout_mse': best_rollout_loss,
                    'improvement': best_improvement,
                    'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
                }, cp_path)
                
                # Robust checkpointing: save scripted/traced model for portability
                try:
                    # We attempt to script a clean version of the model.
                    if hasattr(model, "_orig_mod"):
                        model_to_save = model._orig_mod
                    else:
                        model_to_save = model
                    
                    # Use trace as primary for robust architecture embedding since it handles
                    # most standard Transformer patterns better than scripting.
                    try:
                        # Use CPU for tracing to avoid device-specific issues in the trace
                        model_cpu = model_to_save.to('cpu')
                        model_cpu.eval() # Ensure eval mode for tracing
                        dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                        with torch.no_grad():
                            traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                        torch.jit.save(traced_model, cp_path.replace(".pt", "_scripted.pt"))
                        # Move back to original device and train mode
                        model_to_save.to(Config.DEVICE)
                        model_to_save.train()
                    except Exception as trace_err:
                        # Fallback to scripting if tracing fails
                        scripted_model = torch.jit.script(model_to_save)
                        torch.jit.save(scripted_model, cp_path.replace(".pt", "_scripted.pt"))
                except Exception as e:
                    # Don't fail the whole run if scripting/tracing fails, but log it
                    print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                
                last_checkpoint_time = time.time()
                # Also log to wandb that we saved a checkpoint
                wandb.log({"checkpoint_saved": 1}, commit=False)
        
        train_loss /= len(train_loader)
        
        # Save best model based on training L2 performance
        if train_loss < best_train_l2:
            best_train_l2 = train_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            # Existing convention: saved_models/[run_name]_train_best.pt
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_train_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_l2': train_loss,
                'val_l2': best_val_loss,
                'rollout_mse': best_rollout_loss,
                'improvement': best_improvement,
                'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
            }, save_path)
            
            # Robust checkpointing: save scripted/traced model
            try:
                if hasattr(model, "_orig_mod"):
                    model_to_save = model._orig_mod
                else:
                    model_to_save = model
                
                try:
                    model_cpu = model_to_save.to('cpu')
                    model_cpu.eval()
                    dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                    torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                    model_to_save.to(Config.DEVICE)
                    model_to_save.train()
                except:
                    scripted_model = torch.jit.script(model_to_save)
                    torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
            except Exception as e:
                print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                
            print(f"  --> Saved new best training model (L2={train_loss:.6f})")
            print(f"\033[94m{save_path}\033[0m")

        if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
            print(f"\nReached {Config.MAX_RUNTIME_PER_CANDIDATE}s limit. Moving to next candidate.")
            wandb.finish()
            return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

        # Validation
        if (epoch + 1) % Config.VAL_INTERVAL == 0 or (epoch == Config.EPOCHS - 1 and Config.DEVICE == "cuda"):
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
            
            # --- Multi-step Rollout Evaluation (The "Most Concerning Metric") ---
            model.eval()
            rollout_mse = 0
            persistence_mse = 0
            rollout_count = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 10: break 
                    batch = batch.to(Config.DEVICE)
                    
                    context_len = 26 * Config.VAL_CONTEXT_STEPS
                    inputs = batch[:, :context_len, :]
                    targets = batch[:, context_len : context_len + Config.VAL_ROLLOUT_STEPS, :Config.LATENT_DIM]
                    
                    # --- Persistence Baseline (Static Step 12) ---
                    # Take the last frame of the context (the 12th time step)
                    last_frame = inputs[:, -26:, :Config.LATENT_DIM] # shape (B, 26, 47)
                    num_repeats = Config.VAL_ROLLOUT_STEPS // 26
                    persistence_preds = last_frame.repeat(1, num_repeats, 1)
                    persistence_mse += mse_loss(persistence_preds, targets).item()
    
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
                        rollout_mse += mse_loss(preds, targets).item()
                        rollout_count += 1
            
            if rollout_count > 0:
                rollout_mse /= rollout_count
                persistence_mse /= rollout_count
            
            persistence_improvement = (persistence_mse - rollout_mse) / (persistence_mse + 1e-8) * 100
            best_improvement = max(best_improvement, persistence_improvement)
    
            wandb.log({
                "train_loss": train_loss, 
                "val_loss": val_loss, 
                "rollout_mse": rollout_mse,
                "persistence_mse": persistence_mse,
                "persistence_improvement_pct": persistence_improvement,
                "baseline_red_line": persistence_mse, # Red line on rollout_mse plot
                "epoch": epoch
            })
            print(f"Epoch {epoch+1}: Train L2={train_loss:.6f}, Val L2={val_loss:.6f}")
            print(f"         Rollout MSE={rollout_mse:.6f}, Persistence MSE={persistence_mse:.6f} ({persistence_improvement:.1f}% better)")
            
            # Save best model based on rollout performance
            if rollout_mse < best_rollout_loss and rollout_mse > 0:
                best_rollout_loss = rollout_mse
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_rollout_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'rollout_mse': rollout_mse,
                    'persistence_improvement': persistence_improvement,
                    'val_l2': best_val_loss,
                    'improvement': best_improvement,
                    'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
                }, save_path)
                
                # Robust checkpointing: save scripted/traced model
                try:
                    if hasattr(model, "_orig_mod"):
                        model_to_save = model._orig_mod
                    else:
                        model_to_save = model
                        
                    try:
                        model_cpu = model_to_save.to('cpu')
                        model_cpu.eval()
                        dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                        with torch.no_grad():
                            traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                        torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                        model_to_save.to(Config.DEVICE)
                        model_to_save.train()
                    except:
                        scripted_model = torch.jit.script(model_to_save)
                        torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
                except Exception as e:
                    print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                    
                print(f"  --> Saved new best rollout model!")
                print(f"\033[94m{save_path}\033[0m")
            
            wandb.run.summary["best_rollout_mse"] = best_rollout_loss
            wandb.run.summary["best_improvement"] = best_improvement
        else:
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}: Train L2={train_loss:.6f} (Validation skipped)")

        # --- Early exit for search ---
        if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
            print(f"60s limit reached for {run_name}. Moving to next candidate.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_l2': val_loss,
                'rollout_mse': best_rollout_loss,
                'improvement': best_improvement,
                'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
            }, save_path)
            print(f"  --> Saved new best validation model (L2={val_loss:.6f})")
            print(f"\033[94m{save_path}\033[0m")
            
            # Robust checkpointing: save scripted/traced model
            try:
                if hasattr(model, "_orig_mod"):
                    model_to_save = model._orig_mod
                else:
                    model_to_save = model
                
                try:
                    model_cpu = model_to_save.to('cpu')
                    model_cpu.eval()
                    dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                    torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                    model_to_save.to(Config.DEVICE)
                    model_to_save.train()
                except:
                    scripted_model = torch.jit.script(model_to_save)
                    torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
            except Exception as e:
                print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
            
    wandb.finish()
    return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

if __name__ == "__main__":
    results = []
    # TOP 3 CANDIDATES: 0, 1, 5
    TOP_CANDIDATES = [0, 1, 5]
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "search":
            print(f"Starting architecture search with {Config.MAX_RUNTIME_PER_CANDIDATE}s per candidate...")
            for i in range(len(Config.SEARCH_SPACE)):
                print(f"\n--- Testing Candidate {i} ---")
                res = train(i)
                results.append((i, res))
        elif sys.argv[1] == "top3":
            print(f"Running TOP 3 candidates for {Config.MAX_RUNTIME_PER_CANDIDATE}s each...")
            for i in TOP_CANDIDATES:
                print(f"\n--- Testing Top Candidate {i} ---")
                res = train(i)
                results.append((i, res))
        elif sys.argv[1].isdigit():
            res = train(int(sys.argv[1]))
            results.append((int(sys.argv[1]), res))
        else:
            Config.VARIANT = sys.argv[1]
            res = train()
            results.append((Config.VARIANT, res))
    else:
        # Default to searching through all candidates
        print("Starting architecture search with 60s per candidate...")
        for i in range(len(Config.SEARCH_SPACE)):
            print(f"\n--- Testing Candidate {i} ---")
            res = train(i)
            results.append((i, res))

    if results:
        print("\n" + "="*50)
        print("ARCHITECTURE SEARCH LEADERBOARD")
        print("="*50)
        print(f"{'ID':<5} | {'Val L2':<10} | {'Rollout MSE':<12} | {'Improv %':<10}")
        print("-" * 50)
        for cand_id, res in results:
            if res:
                print(f"{str(cand_id):<5} | {res['val_l2']:<10.6f} | {res['rollout_l2']:<12.6f} | {res['improvement']:<10.2f}%")
        print("="*50)
