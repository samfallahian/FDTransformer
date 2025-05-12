#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train script for VAE_01 model.
This script trains the SpatialAwareVAE model using either Apple Silicon or NVIDIA GPU if available.
Supports loading from a saved model checkpoint with the --resume_checkpoint argument.
"""

import os
import sys
import logging
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import wandb
from tqdm import tqdm

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our custom modules
from encoder.abandoned_models.model_VAE_01 import SpatialAwareVAE
from EfficientDataLoader import EfficientDataLoader

# Figure out the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preferences_path = os.path.join(project_root, "experiment.preferences")
from Ordered_001_Initialize import HostPreferences

# When creating preferences, pass the resolved path:
preferences = HostPreferences(filename=preferences_path)

# Training configuration constants
BATCH_SIZE = 768
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
NUM_WORKERS = 32
SAVE_INTERVAL = 10
BATCHES_PER_EPOCH = 5
CACHE_SIZE = 150
MODEL_NAME = "VAE_01"
ENHANCED_VAE = True  # Changed to False to use the basic model for first run

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


# Define a debug wrapper to track tensor shapes during forward passes
def log_tensor_shape(tensor, name):
    """Log the shape of a tensor for debugging"""
    if tensor is not None:
        logger.debug(f"TENSOR SHAPE - {name}: {tensor.shape}")
    else:
        logger.debug(f"TENSOR SHAPE - {name}: None")
    return tensor


# Wrapper around SpatialAwareVAE to add debug logging
class DebugSpatialVAE(SpatialAwareVAE):
    def __init__(self, *args, **kwargs):
        super(DebugSpatialVAE, self).__init__(*args, **kwargs)
        logger.debug(f"Initialized DebugSpatialVAE with args: {args}, kwargs: {kwargs}")

    def forward(self, x):
        logger.debug("=== Starting DebugSpatialVAE Forward Pass ===")
        log_tensor_shape(x, "Input")

        try:
            # Encode the input
            logger.debug("--- Encoding Input ---")
            mu, logvar = self.encode(x)
            log_tensor_shape(mu, "Mu")
            log_tensor_shape(logvar, "LogVar")

            # Sample from the latent distribution
            logger.debug("--- Sampling from Latent Distribution ---")
            z = self.reparameterize(mu, logvar)
            log_tensor_shape(z, "Latent Z")

            # Decode the latent representation
            logger.debug("--- Decoding Latent Representation ---")
            reconstruction = self.decode(z)
            log_tensor_shape(reconstruction, "Reconstruction")

            logger.debug("=== Completed DebugSpatialVAE Forward Pass ===")
            return reconstruction, mu, logvar

        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    # Parse command-line arguments for model checkpoint resuming
    parser = argparse.ArgumentParser(description="Train VAE_01 model with SpatialAwareVAE")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to checkpoint (.pt file) to resume training from (optional).")
    parser.add_argument('--enhanced', action='store_true', default=ENHANCED_VAE,
                        help="Use the enhanced spatial VAE model with graph convolutions and attention.")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help="Learning rate for the optimizer.")
    args = parser.parse_args()

    # Read model file for notes
    model_filepath = os.path.join(parent_dir, 'encoder', 'model_VAE_01.py')
    notes_content = ""
    try:
        with open(model_filepath, 'r') as file:
            notes_content = file.read()
    except Exception as e:
        notes_content = f"Failed to load model file: {e}"

    # Start wandb run
    wandb_run = wandb.init(
        project=MODEL_NAME,
        name=f"{MODEL_NAME}_{'enhanced' if args.enhanced else 'basic'}_run",
        config={
            "batch_size": args.batch_size,
            "epochs": NUM_EPOCHS,
            "learning_rate": args.learning_rate,
            "model": MODEL_NAME,
            "enhanced": args.enhanced,
            "latent_dim": 47,
            "input_dim": 375
        },
        notes=notes_content
    )

    # Setup device (GPU/CPU)
    device = setup_device()

    # Create save directories in current directory
    save_dir = '../saved_models'
    log_dir = '../logs'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize model and optimizer
    logger.debug("Initializing model and optimizer")

    # For initial testing, only use the basic model
    model_class = DebugSpatialVAE if logger.level <= logging.DEBUG else SpatialAwareVAE
    model = model_class(
        input_dim=375,
        latent_dim=47,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        enhanced=args.enhanced  # Add this line to pass the enhanced flag
    ).to(device)
    logger.info(f"Using {'Enhanced' if args.enhanced else 'Basic'} Spatial VAE model")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0  # Default start at first epoch

    # === Load checkpoint if provided ===
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint
    else:
        # Convenience: If model_final.pt exists and not specified, can default to that
        default_final = os.path.join(save_dir, f"{MODEL_NAME}_final.pt")
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

    # Initialize data loader
    logger.debug("Initializing data loader")
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        cache_size=CACHE_SIZE,
        pin_memory=True,  # Add this line to enable pinned memory
        shuffle=True
    )
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")

    # Training loop
    start_time = time.time()
    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()  # Start the epoch timer
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        num_batches = 0

        # Initialize tqdm progress bar for batches
        with tqdm(total=BATCHES_PER_EPOCH, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch") as pbar:
            for batch_idx in range(BATCHES_PER_EPOCH):
                # Get batch from dataloader
                batch = dataloader.get_batch(NUMBER_OF_ROWS=args.batch_size)
                velocity_data = batch['velocity_data']

                # Convert to tensor and move to device
                x = torch.tensor(velocity_data, dtype=torch.float32).to(device)

                try:
                    # Forward pass and loss calculation
                    recon_x, mu, logvar = model(x)
                    total_loss, recon_loss, kld_loss = model.loss_function(
                        recon_x, x, mu, logvar, kld_weight=1.0
                    )

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    # Update metrics
                    epoch_loss += total_loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kld_loss += kld_loss.item()
                    num_batches += 1
                    global_step += 1

                    # Update tqdm bar
                    pbar.set_postfix({
                        "Batch Loss": f"{total_loss.item():.6f}",
                        "Recon Loss": f"{recon_loss.item():.6f}",
                        "KLD Loss": f"{kld_loss.item():.6f}"
                    })
                    pbar.update(1)

                    # Log to tensorboard and wandb
                    writer.add_scalar('Loss/train_step', total_loss.item(), global_step)
                    writer.add_scalar('Loss/recon_step', recon_loss.item(), global_step)
                    writer.add_scalar('Loss/kld_step', kld_loss.item(), global_step)

                    wandb.log({
                        "Loss/train_step": total_loss.item(),
                        "Loss/recon_step": recon_loss.item(),
                        "Loss/kld_step": kld_loss.item(),
                        "global_step": global_step
                    })

                    if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                                    f"Batch {batch_idx + 1}/{BATCHES_PER_EPOCH}, "
                                    f"Loss: {total_loss.item():.6f}, Recon: {recon_loss.item():.6f}, KLD: {kld_loss.item():.6f}")

                except Exception as e:
                    logger.error(f"Error during training loop: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise

        # Epoch average stats
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kld_loss = epoch_kld_loss / num_batches
        epoch_time = time.time() - epoch_start_time

        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Recon: {avg_recon_loss:.6f}, "
                    f"KLD: {avg_kld_loss:.6f}, "
                    f"Time: {epoch_time:.2f}s")

        # Tensorboard and wandb logging
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch)
        writer.add_scalar('Loss/kld_epoch', avg_kld_loss, epoch)

        wandb.log({
            "Loss/train_epoch": avg_loss,
            "Loss/recon_epoch": avg_recon_loss,
            "Loss/kld_epoch": avg_kld_loss,
            "epoch": epoch
        })

        # Save model checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{epoch + 1}.pt")
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