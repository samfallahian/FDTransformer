#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic training script for permutation models
Adapted from train_WAE_01_cached.py for use with multiple autoencoder architectures
"""

import os
import sys
import time
import logging
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from Ordered_001_Initialize import HostPreferences

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def accelerator_report():
    """Detect CUDA/MPS/CPU and return device"""
    has_cuda = torch.cuda.is_available()
    mps_built = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    mps_avail = mps_built and torch.backends.mps.is_available()
    device = torch.device('cuda') if has_cuda else (torch.device('mps') if mps_avail else torch.device('cpu'))
    logger.info(f"Selected device: {device}")
    return device


def load_cached_array(file_path, limit=None, sample_percentage=None):
    """Load pickled NumPy array with optional sampling"""
    t0 = time.time()
    with open(file_path, 'rb') as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"File {file_path} does not contain a NumPy array")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # Apply percentage sampling first (faster)
    if sample_percentage is not None and sample_percentage < 100:
        target_size = int(len(arr) * sample_percentage / 100.0)
        arr = arr[:target_size]
        logger.info(f"Sampling {sample_percentage}% of data: {target_size} rows")

    if limit is not None:
        arr = arr[:int(limit)]

    t1 = time.time()
    logger.info(f"Loaded {os.path.basename(file_path)}: shape={arr.shape} dtype={arr.dtype} in {t1-t0:.2f}s")
    return arr


def make_loader(arr, batch_size, shuffle, device):
    """Create DataLoader from NumPy array"""
    x_tensor = torch.from_numpy(arr)
    dataset = TensorDataset(x_tensor)

    # Only use pin_memory on CUDA (not MPS, causes warnings)
    use_pin_memory = (device.type == 'cuda')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_pin_memory,
        drop_last=False,
        num_workers=0
    )
    return loader


def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4, model_name="model", save_dir=None):
    """
    Train the model and return RMSE history

    Returns:
        dict with 'train_rmse' and 'val_rmse' lists
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_rmse_history = []
    val_rmse_history = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Reset timer for each epoch

        # Training
        model.train()
        train_loss = 0.0
        train_sse = 0.0
        train_elements = 0

        for batch in train_loader:
            x_cpu = batch[0]
            x = x_cpu.to(device, non_blocking=(device.type == 'cuda'))

            optimizer.zero_grad()

            # Forward pass - handle different model signatures
            output = model(x)
            if len(output) == 2:
                recon_x, z = output
                loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, x, z)
            elif len(output) == 3:
                recon_x, z, extra = output
                loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, x, z, extra)
            else:
                raise ValueError(f"Unexpected model output length: {len(output)}")

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

            # Compute RMSE
            with torch.no_grad():
                sse = torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                train_sse += sse
                train_elements += x.numel()

        train_rmse = np.sqrt(train_sse / train_elements) if train_elements > 0 else float('nan')
        train_rmse_history.append(train_rmse)

        # Validation
        model.eval()
        val_loss = 0.0
        val_sse = 0.0
        val_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                x_cpu = batch[0]
                x = x_cpu.to(device, non_blocking=(device.type == 'cuda'))

                output = model(x)
                if len(output) == 2:
                    recon_x, z = output
                    loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, x, z)
                elif len(output) == 3:
                    recon_x, z, extra = output
                    loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, x, z, extra)

                val_loss += float(loss.item())

                sse = torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                val_sse += sse
                val_elements += x.numel()

        val_rmse = np.sqrt(val_sse / val_elements) if val_elements > 0 else float('nan')
        val_rmse_history.append(val_rmse)

        # Calculate time for this epoch and estimate remaining time
        epoch_time = time.time() - epoch_start_time
        epochs_remaining = num_epochs - (epoch + 1)
        eta_minutes = (epoch_time * epochs_remaining) / 60.0

        # Log progress more frequently (every 5 epochs)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"{model_name} - Epoch {epoch+1}/{num_epochs}: "
                       f"train_RMSE={train_rmse:.6f} val_RMSE={val_rmse:.6f} "
                       f"epoch_time={epoch_time:.1f}s [ETA: {eta_minutes:.1f}min]")

        # Save checkpoint every 10 epochs
        if save_dir and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pt")
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_rmse_history': train_rmse_history,
                    'val_rmse_history': val_rmse_history,
                }, checkpoint_path)
                logger.info(f"  → Saved checkpoint: {os.path.basename(checkpoint_path)}")
            except Exception as e:
                logger.warning(f"  → Failed to save checkpoint: {e}")

    return {
        'train_rmse': train_rmse_history,
        'val_rmse': val_rmse_history,
        'final_train_rmse': train_rmse_history[-1] if train_rmse_history else float('nan'),
        'final_val_rmse': val_rmse_history[-1] if val_rmse_history else float('nan')
    }


def main(model_class, model_name, num_epochs=100, batch_size=128, lr=1e-4, sample_percentage=100, save_dir=None):
    """Main training function"""
    logger.info(f"Starting training for {model_name}")

    # Load preferences
    preferences_path = os.path.join(PARENT_DIR, "experiment.preferences")
    prefs = HostPreferences(filename=preferences_path)
    root_dir = getattr(prefs, 'root_path', os.getcwd())

    # Data paths
    train_path = os.path.join(root_dir, 'training_auto_encoder.pkl')
    val_path = os.path.join(root_dir, 'validation_auto_encoder.pkl')

    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    # Device
    device = accelerator_report()

    # Load data with sampling
    if sample_percentage < 100:
        logger.info(f"Using {sample_percentage}% of data for training")

    train_np = load_cached_array(train_path, sample_percentage=sample_percentage)
    val_np = load_cached_array(val_path, sample_percentage=sample_percentage)

    train_loader = make_loader(train_np, batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_np, batch_size, shuffle=False, device=device)

    del train_np, val_np

    # Initialize model
    model = model_class()

    # Train
    results = train_model(model, train_loader, val_loader, device,
                         num_epochs=num_epochs, lr=lr, model_name=model_name,
                         save_dir=save_dir)

    logger.info(f"Training complete for {model_name}")
    logger.info(f"Final validation RMSE: {results['final_val_rmse']:.6f}")

    return results


if __name__ == "__main__":
    print("This is a generic training module. Import and call main() with your model class.")
