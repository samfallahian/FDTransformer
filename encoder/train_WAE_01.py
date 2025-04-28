#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train script for WAE model 01.
This script trains the WAE model using either Apple Silicon or NVIDIA GPU if available.
Supports loading from a saved model checkpoint with the --resume_checkpoint argument.
"""

import os
import sys
import logging
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse  # <---- NEW

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our custom modules
from encoder.model_WAE_01 import WAE
from EfficientDataLoader import EfficientDataLoader

# Figure out the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preferences_path = os.path.join(project_root, "experiment.preferences")
from Ordered_001_Initialize import HostPreferences

# When creating preferences, pass the resolved path:
preferences = HostPreferences(filename=preferences_path)

# Training configuration constants
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-6
NUM_WORKERS = 250
SAVE_INTERVAL = 100
BATCHES_PER_EPOCH = 100
CACHE_SIZE = 2500
MODEL_NAME = "WAE_01"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set logging level from preferences
if hasattr(preferences, 'logging_level'):
    level = getattr(logging, preferences.logging_level.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        logger.info(f"Set logging level to {preferences.logging_level.upper()}")

def setup_device():
    """Set up the computing device (GPU/CPU) for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using NVIDIA GPU: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected, using CPU for training (this will be slow)")
    return device

import wandb

def main():
    # Parse command-line arguments for model checkpoint resuming (<---- NEW)
    parser = argparse.ArgumentParser(description="Train WAE model 01 (with checkpoint resume support)")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to checkpoint (.pt file) to resume training from (optional).")
    args = parser.parse_args()

    # Read model file for notes (adjust path as needed)
    model_filepath = os.path.join(parent_dir, 'encoder', 'model_WAE_01.py')
    notes_content = ""
    try:
        with open(model_filepath, 'r') as file:
            notes_content = file.read()
    except Exception as e:
        notes_content = f"Failed to load model file: {e}"

    # Start wandb run, use MODEL_NAME as project, add notes
    wandb_run = wandb.init(
        project=MODEL_NAME,
        name=f"{MODEL_NAME}_run",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model": MODEL_NAME,
        },
        notes=notes_content
    )

    # Setup device (GPU/CPU)
    device = setup_device()
    
    # Create save directories in current directory
    save_dir = './saved_models'
    log_dir = './logs'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize model and optimizer
    model = WAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch = 0  # Default start at first epoch
    
    # === Load checkpoint if provided ===  (<---- NEW)
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint
    else:
        # Convenience: If wae_final.pt exists and not specified, can default to that (optional)
        default_final = os.path.join(save_dir, f"wae_final.pt")
        resume_path = default_final if os.path.isfile(default_final) else None

    if resume_path and os.path.isfile(resume_path):
        try:
            logger.info(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']  # Resume from next epoch
                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                logger.info("Checkpoint does not contain epoch info, starting from epoch 0")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error("Training will start from scratch.")
    else:
        if args.resume_checkpoint is not None:
            logger.warning(f"Checkpoint {args.resume_checkpoint} does not exist, will start from scratch.")

    logger.info(f"Initialized WAE model with latent dimension {model.fc4.out_features}")
    
    # Initialize data loader
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        cache_size=CACHE_SIZE,
        shuffle=True
    )
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")
    
    # Training loop
    start_time = time.time()
    global_step = 0
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_mmd_loss = 0
        epoch_triplet_loss = 0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for _ in range(BATCHES_PER_EPOCH):
            # Get batch from dataloader
            batch = dataloader.get_batch(NUMBER_OF_ROWS=BATCH_SIZE)
            velocity_data = batch['velocity_data']
            
            # Convert numpy array to tensor and move to device
            x = torch.tensor(velocity_data, dtype=torch.float32).to(device)
            
            # Forward pass
            recon_x, z = model(x)
            
            # Calculate loss
            loss, recon_loss, mmd_loss, triplet_loss = model.loss_function(recon_x, x, z)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_mmd_loss += mmd_loss.item()
            epoch_triplet_loss += triplet_loss.item()
            num_batches += 1
            global_step += 1
            
            # Log to tensorboard
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/recon_step', recon_loss.item(), global_step)
            writer.add_scalar('Loss/mmd_step', mmd_loss.item(), global_step)
            writer.add_scalar('Loss/triplet_step', triplet_loss.item(), global_step)

            # Log metrics to wandb
            wandb.log({
                "Loss/train_step": loss.item(),
                "Loss/recon_step": recon_loss.item(),
                "Loss/mmd_step": mmd_loss.item(),
                "Loss/triplet_step": triplet_loss.item(),
                "global_step": global_step
            })
        
        # Compute epoch average losses
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_mmd_loss = epoch_mmd_loss / num_batches
        avg_triplet_loss = epoch_triplet_loss / num_batches
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, " 
                   f"Loss: {avg_loss:.6f}, "
                   f"Recon: {avg_recon_loss:.6f}, "
                   f"MMD: {avg_mmd_loss:.6f}, "
                   f"Triplet: {avg_triplet_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch)
        writer.add_scalar('Loss/mmd_epoch', avg_mmd_loss, epoch)
        writer.add_scalar('Loss/triplet_epoch', avg_triplet_loss, epoch)

        # Epoch summary to wandb
        wandb.log({
            "Loss/train_epoch": avg_loss,
            "Loss/recon_epoch": avg_recon_loss,
            "Loss/mmd_epoch": avg_mmd_loss,
            "Loss/triplet_epoch": avg_triplet_loss,
            "epoch": epoch
        })
        
        # Save model checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{epoch+1}.pt")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
            wandb.save(checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{MODEL_NAME}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    wandb.save(final_model_path)
    
    # Training summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Final loss: {avg_loss:.6f}")
    
    writer.close()
    wandb.finish()

if __name__ == "__main__":
    main()