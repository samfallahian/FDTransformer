#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Training Script for Model 5 (AttentionSE)
===================================================

This script trains the Model_GEN3_05_AttentionSE architecture for 1000 epochs
using 100% of the available data.

INSTRUCTIONS FOR LOADING MODELS:
--------------------------------
The checkpoints saved by this script contain the model state, optimizer state, and 
the model architecture is also saved using TorchScript (_scripted.pt).

To load a model for inference:
1. Using PyTorch (if classes are available):
   checkpoint = torch.load('path/to/model.pt')
   from encoder.autoencoderGEN3.models import get_model_by_index
   model = get_model_by_index(4) # 4 is the index for Model 5
   model.load_state_dict(checkpoint['model_state_dict'])

2. Truly single-file loading:
   model = torch.jit.load('path/to/model_scripted.pt')
"""

import os
import sys
import time
import logging
import pickle
import argparse
import glob
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb

# Add project root to path to allow imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index, ORIGINAL_DIM, LATENT_DIM
from Ordered_001_Initialize import HostPreferences

# Configuration
MODEL_IDX = 4  # Model 5 (AttentionSE)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 1000
DATA_PERCENTAGE = 100
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models_production')
GEN3_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'encoder', 'autoencoderGEN3', 'saved_models', 'Model_GEN3_05_AttentionSE_best.pt')
BASE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'encoder', 'saved_models', 'Model_09_Residual_AE_epoch_500.pt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def cleanup_wandb():
    """Purges any existing wandb artifacts in the current directory."""
    wandb_dir = os.path.join(os.getcwd(), 'wandb')
    if os.path.exists(wandb_dir):
        print_rainbow(f"--- PURGING WANDB ARTIFACTS AT {wandb_dir} ---")
        try:
            shutil.rmtree(wandb_dir)
            logger.info("Successfully purged wandb directory.")
        except Exception as e:
            logger.error(f"Failed to purge wandb directory: {e}")

def accelerator_report():
    print_rainbow("--- SEARCHING FOR BEST AVAILABLE ACCELERATOR ---")
    if torch.cuda.is_available():
        print_rainbow("TRYING CUDA (NVIDIA)... SUCCESS!")
        device = torch.device('cuda')
    else:
        print_rainbow("TRYING CUDA (NVIDIA)... NOT FOUND.")
        if torch.backends.mps.is_available():
            print_rainbow("TRYING MPS (APPLE SILICON)... SUCCESS!")
            device = torch.device('mps')
        else:
            print_rainbow("TRYING MPS (APPLE SILICON)... NOT FOUND.")
            print_rainbow("FALLING BACK TO CPU... SUCCESS!")
            device = torch.device('cpu')
    print_rainbow(f"*** FINAL SELECTION: {device.type.upper()} IS ACTIVE ***")
    return device

def load_data(sample_percentage=100):
    preferences_path = os.path.join(PROJECT_ROOT, "experiment.preferences")
    prefs = HostPreferences(filename=preferences_path)
    root_dir = getattr(prefs, 'root_path', os.getcwd())
    
    train_path = os.path.join(root_dir, 'training_auto_encoder.pkl')
    val_path = os.path.join(root_dir, 'validation_auto_encoder.pkl')
    
    def load_file(path):
        with open(path, 'rb') as f:
            arr = pickle.load(f).astype(np.float32)
        if sample_percentage < 100:
            arr = arr[:int(len(arr) * sample_percentage / 100.0)]
        return arr

    logger.info(f"Loading {sample_percentage}% of data...")
    train_np = load_file(train_path)
    val_np = load_file(val_path)
    return train_np, val_np

def purge_old_checkpoints(model_name, keep=5):
    """Keeps only the last N checkpoints for the given model."""
    # Find all .pt files (excluding scripted versions for sorting)
    pattern = os.path.join(SAVE_DIR, f"{model_name}_epoch_*.pt")
    ckpts = [f for f in glob.glob(pattern) if not f.endswith("_scripted.pt")]
    
    # Sort by epoch number extracted from filename
    def get_epoch(filename):
        try:
            return int(filename.split('_epoch_')[-1].split('.pt')[0])
        except:
            return 0
            
    ckpts.sort(key=get_epoch)
    
    if len(ckpts) > keep:
        to_remove = ckpts[:-keep]
        for f in to_remove:
            try:
                os.remove(f)
                # Also remove associated scripted model
                scripted = f.replace(".pt", "_scripted.pt")
                if os.path.exists(scripted):
                    os.remove(scripted)
                logger.info(f"Purged old checkpoint: {os.path.basename(f)}")
            except Exception as e:
                logger.error(f"Failed to remove {f}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Model 5 Production Training")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--data_percentage', type=int, default=DATA_PERCENTAGE)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # Initial cleanup
    cleanup_wandb()
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = accelerator_report()
    train_np, val_np = load_data(sample_percentage=args.data_percentage)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=args.batch_size, shuffle=False)
    
    model = get_model_by_index(MODEL_IDX).to(device)
    model_name = type(model).__name__
    
    # Load base weights
    loaded_any = False
    
    # Priority 1: Specific GEN3 Model 5 checkpoint
    if os.path.exists(GEN3_CHECKPOINT_PATH):
        print_rainbow(f"*** DETECTED GEN3 MODEL 5 CHECKPOINT AT {GEN3_CHECKPOINT_PATH} ***")
        print_rainbow("--- INITIATING MYRIAD LOADING SEQUENCE ---")
        try:
            checkpoint = torch.load(GEN3_CHECKPOINT_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            matched_keys = [k for k in state_dict.keys() if k in model.state_dict()]
            
            print_rainbow(f"SUCCESS: MATCHED {len(matched_keys)} TENSORS FROM GEN3 CHECKPOINT!")
            if not missing_keys and not unexpected_keys:
                print_rainbow("PERFECT MATCH: 100% OF ARCHITECTURE WEIGHTS RECOVERED!")
            else:
                if missing_keys:
                    print_rainbow(f"INFO: {len(missing_keys)} KEYS MISSING (STILL INITIALIZING FROM DEFAULTS)")
                if unexpected_keys:
                    print_rainbow(f"INFO: {len(unexpected_keys)} UNEXPECTED KEYS IGNORED")
            
            loaded_any = True
        except Exception as e:
            print_rainbow(f"FAILED TO LOAD GEN3 CHECKPOINT: {e}")

    # Priority 2: Fallback to Model 09 weights if nothing loaded yet
    if not loaded_any and os.path.exists(BASE_WEIGHTS_PATH):
        print_rainbow(f"*** GEN3 CHECKPOINT NOT FOUND, FALLING BACK TO MODEL 09 BASE WEIGHTS ***")
        try:
            checkpoint = torch.load(BASE_WEIGHTS_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print_rainbow("SUCCESS: BASE WEIGHTS FROM MODEL 09 INJECTED!")
            loaded_any = True
        except Exception as e:
            print_rainbow(f"FAILED TO LOAD MODEL 09 WEIGHTS: {e}")
            
    if not loaded_any:
        print_rainbow("!!! WARNING: NO PRE-TRAINED WEIGHTS LOADED. STARTING FROM RANDOM INITIALIZATION !!!")
    else:
        print_rainbow("--- MODEL WEIGHTS SEEDED SUCCESSFULLY ---")
    
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    
    if not args.no_wandb:
        wandb.init(project="autoencoder-GEN3-production", name=f"{model_name}-1000epochs", config=vars(args))

    best_val_rmse = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_sse, train_elements = 0.0, 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, z = model(x)
            loss, _, _, _ = model.loss_function(recon_x, x, z)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_sse += torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                train_elements += x.numel()
        
        train_rmse = np.sqrt(train_sse / train_elements)
        
        model.eval()
        val_sse, val_elements = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, z = model(x)
                val_sse += torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                val_elements += x.numel()
        
        val_rmse = np.sqrt(val_sse / val_elements)
        
        if not args.no_wandb:
            wandb.log({"epoch": epoch+1, "train_rmse": train_rmse, "val_rmse": val_rmse})
            
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train RMSE={train_rmse:.6f}, Val RMSE={val_rmse:.6f}")
        
        # Save every epoch
        save_path = os.path.join(SAVE_DIR, f"{model_name}_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_rmse': val_rmse,
        }, save_path)
        
        # Save scripted version
        try:
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
        except:
            pass
            
        # Purge all but last 5
        purge_old_checkpoints(model_name, keep=5)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Also maintain a 'best' symlink or file if desired, but user just asked for last 5
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_name}_absolute_best.pt"))

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
