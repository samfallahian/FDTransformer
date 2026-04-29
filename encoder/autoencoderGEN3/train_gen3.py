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
try:
    from .config import (
        add_config_argument,
        choose_path,
        config_get,
        configured_path,
        load_config,
        optional_path,
    )
except ImportError:
    from config import (
        add_config_argument,
        choose_path,
        config_get,
        configured_path,
        load_config,
        optional_path,
    )

# Configuration
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 100
DATA_PERCENTAGE = 10
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
DEFAULT_BASE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'encoder', 'autoencoderGEN3', 'saved_models_production', 'Model_GEN3_05_AttentionSE_absolute_best.pt')

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

def resolve_training_paths(args, config):
    train_path = optional_path(args.train_path, base_dir=os.getcwd()) or configured_path(config, "data.train_path")
    val_path = optional_path(args.val_path, base_dir=os.getcwd()) or configured_path(config, "data.val_path")
    if train_path and val_path:
        return train_path, val_path

    data_root = optional_path(args.data_root, base_dir=os.getcwd()) or configured_path(config, "data.data_root")
    train_filename = config_get(config, "data.train_filename", "training_auto_encoder.pkl")
    val_filename = config_get(config, "data.val_filename", "validation_auto_encoder.pkl")
    if data_root:
        return (
            train_path or os.path.join(data_root, train_filename),
            val_path or os.path.join(data_root, val_filename),
        )

    raise ValueError(
        "Training and validation data paths are required. Set data.train_path and "
        "data.val_path in autoencoderGEN3/config.json, set data.data_root, or pass "
        "--train_path/--val_path on the command line."
    )


def load_data(train_path, val_path, sample_percentage=10):
    def load_file(path):
        with open(path, 'rb') as f:
            arr = pickle.load(f).astype(np.float32)
        if sample_percentage < 100:
            arr = arr[:int(len(arr) * sample_percentage / 100.0)]
        return arr

    logger.info(f"Loading {sample_percentage}% of data...")
    logger.info(f"Training data: {train_path}")
    logger.info(f"Validation data: {val_path}")
    train_np = load_file(train_path)
    val_np = load_file(val_path)
    return train_np, val_np

def train_one_model(
    model_idx,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LR,
    data_percentage=DATA_PERCENTAGE,
    save_dir=DEFAULT_SAVE_DIR,
    base_weights_path=DEFAULT_BASE_WEIGHTS_PATH,
    use_wandb=True,
):
    model = get_model_by_index(model_idx).to(device)
    model_name = type(model).__name__
    logger.info(f"Training model {model_idx+1}/10: {model_name}")

    # Load base weights if available
    if os.path.exists(base_weights_path):
        print_rainbow(f"*** LOADING BASE WEIGHTS FROM MODEL 09 INTO {model_name} ***")
        try:
            # Using weights_only=False because the checkpoint contains custom classes/numpy types
            checkpoint = torch.load(base_weights_path, map_location=device, weights_only=False)
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
        logger.warning(f"Base weights NOT found at {base_weights_path}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if use_wandb:
        wandb.init(project="autoencoder-GEN3-comparison", name=model_name, config={
            "model_idx": model_idx,
            "model_name": model_name,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "data_percentage": data_percentage
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
            save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            
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
                torch.jit.save(scripted_model, os.path.join(save_dir, f"{model_name}_scripted.pt"))
            except Exception as e:
                # logger.warning(f"Could not save {model_name} as TorchScript: {e}")
                pass

    logger.info(f"Finished {model_name}. Best Val RMSE: {best_val_rmse:.6f}")
    if use_wandb:
        wandb.finish()
    return best_val_rmse

def main():
    parser = argparse.ArgumentParser(description="GEN3 Autoencoder Comparison")
    add_config_argument(parser)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--data_percentage', type=int, default=None)
    parser.add_argument('--model_idx', type=int, default=None, help="Index of model to train (0-9). If None, trains all.")
    parser.add_argument('--train_path', type=str, default=None, help="Path to training_auto_encoder.pkl")
    parser.add_argument('--val_path', type=str, default=None, help="Path to validation_auto_encoder.pkl")
    parser.add_argument('--data_root', type=str, default=None, help="Directory containing training and validation pickle files")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory for comparison checkpoints")
    parser.add_argument('--base_weights_path', type=str, default=None, help="Optional checkpoint used to seed comparison models")
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    epochs = args.epochs if args.epochs is not None else int(config_get(config, "training.epochs", DEFAULT_EPOCHS))
    batch_size = args.batch_size if args.batch_size is not None else int(config_get(config, "training.batch_size", DEFAULT_BATCH_SIZE))
    lr = args.lr if args.lr is not None else float(config_get(config, "training.learning_rate", DEFAULT_LR))
    data_percentage = args.data_percentage if args.data_percentage is not None else int(config_get(config, "training.data_percentage", DATA_PERCENTAGE))
    save_dir = choose_path(args.save_dir, config, "paths.comparison_save_dir", DEFAULT_SAVE_DIR)
    base_weights_path = choose_path(args.base_weights_path, config, "paths.comparison_base_weights_path", DEFAULT_BASE_WEIGHTS_PATH)
    train_path, val_path = resolve_training_paths(args, config)

    os.makedirs(save_dir, exist_ok=True)
    device = accelerator_report()
    
    train_np, val_np = load_data(train_path, val_path, sample_percentage=data_percentage)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=batch_size, shuffle=False)
    
    results = {}
    model_indices = [args.model_idx] if args.model_idx is not None else range(10)
    
    for i in model_indices:
        try:
            best_rmse = train_one_model(i, train_loader, val_loader, device, 
                                      num_epochs=epochs,
                                      batch_size=batch_size,
                                      lr=lr,
                                      data_percentage=data_percentage,
                                      save_dir=save_dir,
                                      base_weights_path=base_weights_path,
                                      use_wandb=not args.no_wandb)
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
