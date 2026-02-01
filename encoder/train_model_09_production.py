#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production training script for Model_09_Residual_AE

Continues training from checkpoint with full dataset and wandb logging.
This is the winning model from the 10-model comparison experiment.
"""

import os
import sys
import time
import logging
import pickle
import argparse
import glob
import re
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from encoder.permutations.model_09_residual_ae import ResidualAE
from Ordered_001_Initialize import HostPreferences

# Import wandb
import wandb

# Configuration
MODEL_NAME = "Model_09_Residual_AE"
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 500

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


def load_cached_array(file_path, sample_percentage=100):
    """Load pickled NumPy array with optional sampling"""
    t0 = time.time()
    with open(file_path, 'rb') as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"File {file_path} does not contain a NumPy array")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    # Apply percentage sampling
    if sample_percentage < 100:
        target_size = int(len(arr) * sample_percentage / 100.0)
        arr = arr[:target_size]
        logger.info(f"Sampling {sample_percentage}% of data: {target_size} rows")

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


def prune_old_checkpoints(save_dir, model_name, keep_last=5):
    """Keep only the last N checkpoints, delete older ones"""
    try:
        # Find all epoch checkpoints (not best or final)
        pattern = os.path.join(save_dir, f"{model_name}_epoch_*.pt")
        checkpoints = glob.glob(pattern)

        # Extract epoch numbers and sort
        checkpoint_epochs = []
        for ckpt in checkpoints:
            match = re.search(r'epoch_(\d+)\.pt$', ckpt)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_epochs.append((epoch_num, ckpt))

        # Sort by epoch number
        checkpoint_epochs.sort(key=lambda x: x[0])

        # Delete all but the last N
        if len(checkpoint_epochs) > keep_last:
            to_delete = checkpoint_epochs[:-keep_last]
            for epoch_num, ckpt_path in to_delete:
                try:
                    os.remove(ckpt_path)
                    logger.info(f"  → Pruned old checkpoint: epoch_{epoch_num}")
                except Exception as e:
                    logger.warning(f"  → Failed to delete {ckpt_path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to prune checkpoints: {e}")


def train_model(model, train_loader, val_loader, device, optimizer,
                start_epoch=0, num_epochs=500, save_dir=None, use_wandb=True, keep_checkpoints=5):
    """
    Train the model from a checkpoint

    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        optimizer: Optimizer (already loaded with state)
        start_epoch: Starting epoch number
        num_epochs: Total number of epochs to train
        save_dir: Directory to save checkpoints
        use_wandb: Whether to log to wandb
        keep_checkpoints: Number of recent checkpoints to keep

    Returns:
        dict with training history
    """
    model = model.to(device)

    train_rmse_history = []
    val_rmse_history = []

    best_val_rmse = float('inf')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_sse = 0.0
        train_elements = 0

        for batch in train_loader:
            x_cpu = batch[0]
            x = x_cpu.to(device, non_blocking=(device.type == 'cuda'))

            optimizer.zero_grad()

            # Forward pass
            recon_x, z = model(x)
            loss, recon_loss, l2_reg, _ = model.loss_function(recon_x, x, z)

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
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_sse = 0.0
        val_elements = 0

        with torch.no_grad():
            for batch in val_loader:
                x_cpu = batch[0]
                x = x_cpu.to(device, non_blocking=(device.type == 'cuda'))

                recon_x, z = model(x)
                loss, recon_loss, l2_reg, _ = model.loss_function(recon_x, x, z)

                val_loss += float(loss.item())

                sse = torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                val_sse += sse
                val_elements += x.numel()

        val_rmse = np.sqrt(val_sse / val_elements) if val_elements > 0 else float('nan')
        val_rmse_history.append(val_rmse)
        avg_val_loss = val_loss / len(val_loader)

        # Calculate time and ETA
        epoch_time = time.time() - epoch_start_time
        epochs_remaining = num_epochs - (epoch + 1)
        eta_minutes = (epoch_time * epochs_remaining) / 60.0

        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_rmse': train_rmse,
                'val_loss': avg_val_loss,
                'val_rmse': val_rmse,
                'epoch_time_seconds': epoch_time,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

        # Console logging
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"train_RMSE={train_rmse:.6f} val_RMSE={val_rmse:.6f} "
                       f"epoch_time={epoch_time:.1f}s [ETA: {eta_minutes:.1f}min]")

        # Save checkpoint EVERY epoch
        if save_dir:
            checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{epoch+1}.pt")
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'train_rmse_history': train_rmse_history,
                    'val_rmse_history': val_rmse_history,
                    'best_val_rmse': best_val_rmse,
                }, checkpoint_path)

                # Only log saves every 5 epochs to reduce clutter
                if (epoch + 1) % 5 == 0 or epoch == start_epoch or epoch == num_epochs - 1:
                    logger.info(f"  → Saved checkpoint: {os.path.basename(checkpoint_path)}")

                # Save to wandb only every 10 epochs
                if use_wandb and (epoch + 1) % 10 == 0:
                    wandb.save(checkpoint_path)

            except Exception as e:
                logger.warning(f"  → Failed to save checkpoint: {e}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if save_dir:
                best_path = os.path.join(save_dir, f"{MODEL_NAME}_best.pt")
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_rmse': train_rmse,
                        'val_rmse': val_rmse,
                        'train_rmse_history': train_rmse_history,
                        'val_rmse_history': val_rmse_history,
                        'best_val_rmse': best_val_rmse,
                    }, best_path)
                    logger.info(f"  → New best model! Val RMSE: {val_rmse:.6f}")
                    if use_wandb:
                        wandb.save(best_path)
                except Exception as e:
                    logger.warning(f"  → Failed to save best model: {e}")

        # Prune old checkpoints every 5 epochs
        if save_dir and (epoch + 1) % 5 == 0:
            prune_old_checkpoints(save_dir, MODEL_NAME, keep_last=keep_checkpoints)

    return {
        'train_rmse': train_rmse_history,
        'val_rmse': val_rmse_history,
        'final_train_rmse': train_rmse_history[-1] if train_rmse_history else float('nan'),
        'final_val_rmse': val_rmse_history[-1] if val_rmse_history else float('nan'),
        'best_val_rmse': best_val_rmse
    }


def main():
    parser = argparse.ArgumentParser(description="Train Model_09_Residual_AE with full dataset")
    parser.add_argument('--resume_checkpoint', type=str,
                       default=None,  # Set below
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Total number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--data_percentage', type=int, default=100,
                       help='Percentage of data to use (100 = full dataset)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='fluid-dynamics-ae',
                       help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (default: Model_09_Residual_AE_production)')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                       help='Number of recent checkpoints to keep (default: 5)')
    parser.add_argument('--no_resume', action='store_true',
                       help='Start from scratch, do not load checkpoint')

    args = parser.parse_args()

    # Set default checkpoint path if not provided and not disabled
    if args.resume_checkpoint is None and not args.no_resume:
        args.resume_checkpoint = os.path.join(
            PARENT_DIR,
            'encoder/permutations/checkpoints/Model_09_Residual_AE/Model_09_Residual_AE_epoch_100.pt'
        )

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

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        # Read model source code to embed in wandb
        model_source_path = os.path.join(PARENT_DIR, 'encoder', 'permutations', 'model_09_residual_ae.py')
        try:
            with open(model_source_path, 'r') as f:
                model_source_code = f.read()
        except Exception as e:
            logger.warning(f"Could not read model source: {e}")
            model_source_code = "Could not load model source"

        wandb_name = args.wandb_name or f"{MODEL_NAME}_production"

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            notes=model_source_code,  # Embed model definition in notes
            config={
                'model': MODEL_NAME,
                'architecture': 'Residual Autoencoder with skip connections',
                'input_dim': 375,
                'latent_dim': 47,
                'hidden_dims': [250, 150, 100],
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'data_percentage': args.data_percentage,
                'optimizer': 'Adam',
                'loss': 'MSE + L2 regularization',
                'resume_from': args.resume_checkpoint,
                'train_path': train_path,
                'val_path': val_path,
                'keep_checkpoints': args.keep_checkpoints,
            }
        )
        logger.info(f"Initialized wandb: project={args.wandb_project}, name={wandb_name}")

    # Load data
    logger.info(f"Loading data: {args.data_percentage}% of full dataset")
    train_np = load_cached_array(train_path, sample_percentage=args.data_percentage)
    val_np = load_cached_array(val_path, sample_percentage=args.data_percentage)

    train_loader = make_loader(train_np, args.batch_size, shuffle=True, device=device)
    val_loader = make_loader(val_np, args.batch_size, shuffle=False, device=device)

    del train_np, val_np

    # Initialize model
    logger.info("Initializing model...")
    model = ResidualAE()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load checkpoint
    start_epoch = 0
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        logger.info(f"Loading checkpoint from: {args.resume_checkpoint}")
        try:
            # weights_only=False is safe for our own checkpoints (PyTorch 2.6+ requirement)
            checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)

            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

            prev_val_rmse = checkpoint.get('val_rmse', 'N/A')
            logger.info(f"✓ Successfully resumed from epoch {start_epoch}")
            logger.info(f"  Previous validation RMSE: {prev_val_rmse}")

            if use_wandb:
                wandb.config.update({
                    'resumed_from_epoch': start_epoch,
                    'resumed_val_rmse': checkpoint.get('val_rmse', None)
                })

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Starting from scratch...")
            start_epoch = 0
    elif args.no_resume:
        logger.info("--no_resume specified, starting from scratch")
    else:
        if args.resume_checkpoint:
            logger.warning(f"Checkpoint not found: {args.resume_checkpoint}")
        logger.info("Starting from scratch (no checkpoint loaded)")

    # Create save directory
    save_dir = os.path.join(PARENT_DIR, 'encoder', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    # Train
    logger.info("="*80)
    logger.info(f"Starting training from epoch {start_epoch + 1} to {args.epochs}")
    logger.info(f"Saving checkpoints to: {save_dir}")
    logger.info(f"Keeping last {args.keep_checkpoints} checkpoints")
    logger.info("="*80)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        start_epoch=start_epoch,
        num_epochs=args.epochs,
        save_dir=save_dir,
        use_wandb=use_wandb,
        keep_checkpoints=args.keep_checkpoints
    )

    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Final training RMSE: {results['final_train_rmse']:.6f}")
    logger.info(f"Final validation RMSE: {results['final_val_rmse']:.6f}")
    logger.info(f"Best validation RMSE: {results['best_val_rmse']:.6f}")
    logger.info("="*80)

    # Finish wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
