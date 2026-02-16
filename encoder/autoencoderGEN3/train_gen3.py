#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEN3 Autoencoder Training Script
================================

This script trains 10 different autoencoder architectures to compare their performance.

INSTRUCTIONS FOR LOADING MODELS:
--------------------------------
The checkpoints saved by this script contain the model state, optimizer state, and 
crucially, the model architecture is also saved using TorchScript where possible,
or by embedding the model class name and configuration.

To load a model for inference:
1. Using PyTorch (if classes are available):
   checkpoint = torch.load('path/to/model.pt')
   model = get_model_by_index(index)
   model.load_state_dict(checkpoint['model_state_dict'])

2. Truly single-file loading (Experimental):
   If the model was saved with torch.jit.save():
   model = torch.jit.load('path/to/model_scripted.pt')

Note: This script also embeds the model source code in the 'model_source' field 
of the checkpoint for future reference.
"""

import os
import sys
import time
import logging
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb

# Add project root to path to allow imports
# __file__ is encoder/autoencoderGEN3/train_gen3.py
# dirname(__file__) is encoder/autoencoderGEN3
# dirname(dirname(__file__)) is encoder
# dirname(dirname(dirname(__file__))) is project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index, ORIGINAL_DIM, LATENT_DIM
from Ordered_001_Initialize import HostPreferences

# Configuration
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 100
DATA_PERCENTAGE = 10
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
BASE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'encoder', 'autoencoderGEN3', 'saved_models_production', 'Model_GEN3_05_AttentionSE_absolute_best.pt')

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
    logger.info(f"Selected device: {device}")
    return device

def load_data(sample_percentage=10):
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

def train_one_model(model_idx, train_loader, val_loader, device, num_epochs=100, use_wandb=True):
    model = get_model_by_index(model_idx).to(device)
    model_name = type(model).__name__
    logger.info(f"Training model {model_idx+1}/10: {model_name}")

    # Load base weights if available
    if os.path.exists(BASE_WEIGHTS_PATH):
        print_rainbow(f"*** LOADING BASE WEIGHTS FROM MODEL 09 INTO {model_name} ***")
        try:
            # Using weights_only=False because the checkpoint contains custom classes/numpy types
            checkpoint = torch.load(BASE_WEIGHTS_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Filter out incompatible keys if necessary, though strict=False handles most cases
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Log some info about what was loaded
            matched_keys = [k for k in state_dict.keys() if k in model.state_dict()]
            logger.info(f"Successfully matched {len(matched_keys)} weight tensors.")
            if missing_keys:
                logger.info(f"Missing {len(missing_keys)} keys (expected for non-baseline architectures).")
        except Exception as e:
            logger.error(f"Failed to load base weights: {e}")
    else:
        logger.warning(f"Base weights NOT found at {BASE_WEIGHTS_PATH}")
    
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    
    if use_wandb:
        wandb.init(project="autoencoder-GEN3-comparison", name=model_name, config={
            "model_idx": model_idx,
            "model_name": model_name,
            "epochs": num_epochs,
            "batch_size": DEFAULT_BATCH_SIZE,
            "lr": DEFAULT_LR,
            "data_percentage": DATA_PERCENTAGE
        })

    best_val_rmse = float('inf')
    
    for epoch in range(num_epochs):
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
        
        if use_wandb:
            wandb.log({"epoch": epoch+1, "train_rmse": train_rmse, "val_rmse": val_rmse})
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"[{model_name}] Epoch {epoch+1}/{num_epochs}: Train RMSE={train_rmse:.6f}, Val RMSE={val_rmse:.6f}")
            
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save best model with all info embedded
            save_path = os.path.join(SAVE_DIR, f"{model_name}_best.pt")
            
            # Get model source code
            try:
                import inspect
                model_source = inspect.getsource(type(model))
                # Also get base classes if needed, but for simplicity just get the whole file
                with open(os.path.join(os.path.dirname(__file__), 'models.py'), 'r') as f:
                    full_models_source = f.read()
            except:
                full_models_source = "Source not available"

            torch.save({
                'model_idx': model_idx,
                'model_class': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'epoch': epoch + 1,
                'model_source': full_models_source
            }, save_path)
            
            # Attempt TorchScript for single-file loading if possible
            try:
                # Some models like the Skip connection one might need careful scripting
                # We'll try to script a wrapper that doesn't rely on internal state if possible
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, os.path.join(SAVE_DIR, f"{model_name}_scripted.pt"))
            except Exception as e:
                # logger.warning(f"Could not save {model_name} as TorchScript: {e}")
                pass

    logger.info(f"Finished {model_name}. Best Val RMSE: {best_val_rmse:.6f}")
    if use_wandb:
        wandb.finish()
    return best_val_rmse

def main():
    parser = argparse.ArgumentParser(description="GEN3 Autoencoder Comparison")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--data_percentage', type=int, default=DATA_PERCENTAGE)
    parser.add_argument('--model_idx', type=int, default=None, help="Index of model to train (0-9). If None, trains all.")
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = accelerator_report()
    
    train_np, val_np = load_data(sample_percentage=args.data_percentage)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=args.batch_size, shuffle=False)
    
    results = {}
    model_indices = [args.model_idx] if args.model_idx is not None else range(10)
    
    for i in model_indices:
        try:
            best_rmse = train_one_model(i, train_loader, val_loader, device, 
                                      num_epochs=args.epochs, use_wandb=not args.no_wandb)
            results[f"Model_{i+1}"] = best_rmse
        except Exception as e:
            logger.error(f"Failed to train model {i+1}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("="*40)
    logger.info("COMPARISON RESULTS")
    logger.info("="*40)
    for name, rmse in results.items():
        logger.info(f"{name}: {rmse:.6f}")
    logger.info("="*40)

if __name__ == "__main__":
    main()
