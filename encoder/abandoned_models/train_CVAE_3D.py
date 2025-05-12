#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train script for CVAE_3D model with adapter for flat input data.
This script trains the CVAE_3D model using either Apple Silicon or NVIDIA GPU if available.
Supports loading from a saved model checkpoint with the --resume_checkpoint argument.
"""

import os
import sys
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our custom modules
from encoder.abandoned_models.model_CVAE_3D_01 import CVAE_3D
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
LEARNING_RATE = 1e-5
NUM_WORKERS = 300
SAVE_INTERVAL = 100
BATCHES_PER_EPOCH = 10
CACHE_SIZE = 1500
MODEL_NAME = "CVAE_3D_01"

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


# Wrapper around CVAE_3D to add debug logging
class DebugCVAE_3D(CVAE_3D):
    def __init__(self, *args, **kwargs):
        super(DebugCVAE_3D, self).__init__(*args, **kwargs)
        logger.debug(f"Initialized DebugCVAE_3D with args: {args}, kwargs: {kwargs}")
        
        # Log model architecture details
        logger.debug("=== CVAE_3D Encoder Architecture ===")
        for i, layer in enumerate(self.encoder):
            logger.debug(f"Encoder Layer {i}: {layer}")
        
        logger.debug("=== CVAE_3D Decoder Architecture ===")
        for i, layer in enumerate(self.decoder):
            logger.debug(f"Decoder Layer {i}: {layer}")
            
    def forward(self, x):
        logger.debug("=== Starting DebugCVAE_3D Forward Pass ===")
        log_tensor_shape(x, "CVAE_3D Input")
        
        # Track progress through encoder
        logger.debug("--- Encoder Processing ---")
        encoder_x = x
        for i, layer in enumerate(self.encoder):
            try:
                logger.debug(f"Running encoder layer {i}: {layer}")
                encoder_x = layer(encoder_x)
                log_tensor_shape(encoder_x, f"After encoder layer {i}")
            except Exception as e:
                logger.error(f"Error in encoder layer {i}: {str(e)}")
                logger.error(f"Layer details: {layer}")
                logger.error(f"Input tensor shape: {encoder_x.shape if hasattr(encoder_x, 'shape') else 'unknown'}")
                raise
        
        # Get representation directly
        logger.debug("--- Computing Representation ---")
        try:
            z_representation = self.representation(x)
            log_tensor_shape(z_representation, "Z Representation")
        except Exception as e:
            logger.error(f"Error computing representation: {str(e)}")
            raise
        
        # Full encode 
        logger.debug("--- Bottleneck Processing ---")
        try:
            z, mu, logvar = self.encode(x)
            log_tensor_shape(z, "Z (before FC3)")
            log_tensor_shape(mu, "Mu")
            log_tensor_shape(logvar, "LogVar")
            
            z = self.fc3(z)
            log_tensor_shape(z, "Z (after FC3)")
        except Exception as e:
            logger.error(f"Error in encode/bottleneck: {str(e)}")
            raise
        
        # Decode
        logger.debug("--- Decoder Processing ---")
        try:
            recon_x = self.decode(z)
            log_tensor_shape(recon_x, "Reconstructed Output")
        except Exception as e:
            logger.error(f"Error in decode: {str(e)}")
            raise
            
        logger.debug("=== Completed DebugCVAE_3D Forward Pass ===")
        return recon_x, mu, logvar, z_representation
    
    def encode(self, x):
        logger.debug("--- CVAE_3D Encode Function ---")
        try:
            h = log_tensor_shape(self.encoder(x), "Encoder Output")
            logger.debug("Running bottleneck")
            outputs = self.bottleneck(h)
            mu, logvar = log_tensor_shape(outputs[1], "Mu from bottleneck"), log_tensor_shape(outputs[2], "LogVar from bottleneck")
            z = log_tensor_shape(outputs[0], "Z from bottleneck")
            return z, mu, logvar
        except Exception as e:
            logger.error(f"Error in encode function: {str(e)}")
            raise
    
    def representation(self, x):
        logger.debug("--- CVAE_3D Representation Function ---")
        try:
            encoder_output = log_tensor_shape(self.encoder(x), "Encoder Output (for representation)")
            bottleneck_output = log_tensor_shape(self.bottleneck(encoder_output)[0], "Bottleneck Output (for representation)")
            return bottleneck_output
        except Exception as e:
            logger.error(f"Error in representation function: {str(e)}")
            # Add more detailed error logging
            logger.error(f"Input tensor shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
            if hasattr(x, 'shape'):
                logger.error(f"Input tensor stats - min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
            raise
    
    def decode(self, z):
        logger.debug("--- CVAE_3D Decode Function ---")
        try:
            result = super().decode(z)
            return log_tensor_shape(result, "Decoder Result")
        except Exception as e:
            logger.error(f"Error in decode function: {str(e)}")
            raise


# Define the adapter class for CVAE_3D
class CVAE_3D_Adapter(nn.Module):
    def __init__(self):
        super(CVAE_3D_Adapter, self).__init__()

        # Set constants to match WAE model
        self.original_dim = 375
        self.latent_dim = 47

        logger.info(f"Initializing CVAE_3D_Adapter with latent dimension {self.latent_dim}")

        # Create the core CVAE model with modified latent dimension
        # Use h_dim=125 as in the original code
        self.cvae = DebugCVAE_3D(image_channels=3, h_dim=125, z_dim=self.latent_dim)

        # Add input and output adapters
        # This transforms flat 375-dim vector to 3D tensor format
        # Assuming the 375 values represent 125 points with xyz values
        # We'll reshape to a larger grid with 3 channels to prevent dimension issues
        self.grid_size = 8  # Increase from 5 to 8 to allow for multiple convolutions
        self.input_adapter = nn.Linear(self.original_dim, self.grid_size * self.grid_size * self.grid_size * 3)
        self.output_adapter = nn.Linear(self.grid_size * self.grid_size * self.grid_size * 3, self.original_dim)
        
        logger.debug(f"ADAPTER CONFIG - Grid size: {self.grid_size}x{self.grid_size}x{self.grid_size}, Channels: 3")
        logger.debug(f"ADAPTER CONFIG - Input adapter: {self.original_dim} → {self.grid_size**3 * 3}")
        logger.debug(f"ADAPTER CONFIG - Output adapter: {self.grid_size**3 * 3} → {self.original_dim}")

    def forward(self, x):
        # Log input tensor shape
        log_tensor_shape(x, "Adapter Input")
        
        # Ensure input is treated as flat vector
        x = x.view(-1, self.original_dim)
        log_tensor_shape(x, "Adapter Input Reshaped")

        # Transform input to 3D format
        x_3d = self.input_adapter(x)
        log_tensor_shape(x_3d, "After Input Adapter (Flat)")
        
        # Reshape to 3D
        x_3d = x_3d.view(-1, 3, self.grid_size, self.grid_size, self.grid_size)  # [batch, channels, D, H, W]
        log_tensor_shape(x_3d, "After Reshape to 3D")

        try:
            # Process with original CVAE model
            logger.debug("Starting CVAE forward pass...")
            recon_x_3d, mu, logvar, z_representation = self.cvae(x_3d)
            logger.debug("Completed CVAE forward pass")
            
            # Log output shapes
            log_tensor_shape(recon_x_3d, "Reconstructed 3D")
            log_tensor_shape(mu, "Mu")
            log_tensor_shape(logvar, "LogVar")
            log_tensor_shape(z_representation, "Z Representation")

            # Convert output back to flat format
            recon_x_flat = recon_x_3d.view(-1, self.grid_size * self.grid_size * self.grid_size * 3)
            log_tensor_shape(recon_x_flat, "Recon Flattened")
            
            recon_x = self.output_adapter(recon_x_flat)
            log_tensor_shape(recon_x, "Final Output")

            return recon_x, mu, logvar, z_representation
            
        except Exception as e:
            logger.error(f"Error in CVAE forward pass: {str(e)}")
            # Print more detailed error information
            import traceback
            logger.error(traceback.format_exc())
            raise

    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss with components:
        1. Reconstruction loss (using log_cosh similar to WAE)
        2. KL divergence (standard VAE term)
        3. Triplet loss for specific point (similar to WAE)
        """
        # Debug shapes
        log_tensor_shape(recon_x, "Loss Input - recon_x")
        log_tensor_shape(x, "Loss Input - x")
        
        # Reshape for triplet loss
        recon_x_grouped = recon_x.view(-1, int(recon_x.nelement() / 3), 3)
        x_grouped = x.view(-1, int(x.nelement() / 3), 3)
        
        log_tensor_shape(recon_x_grouped, "Grouped recon_x")
        log_tensor_shape(x_grouped, "Grouped x")

        # Reconstruction loss using log cosh (matches WAE implementation)
        prediction_error = x_grouped - recon_x_grouped
        scaled_error = 10 * prediction_error  # Same scaling as WAE
        log_cosh_loss = torch.log(torch.cosh(scaled_error))
        recon_loss = torch.mean(torch.sum(log_cosh_loss, dim=1))

        # KL divergence for VAE
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Triplet loss for the 63rd point (same as in WAE)
        random_index = torch.randint(len(x_grouped), size=(1,)).item()
        triplet_loss = F.mse_loss(recon_x_grouped[random_index, 62], x_grouped[random_index, 62])

        # Total loss (similar structure to WAE but with KL instead of MMD)
        total_loss = recon_loss + kl_loss + triplet_loss
        
        logger.debug(f"LOSS - Recon: {recon_loss.item():.6f}, KL: {kl_loss.item():.6f}, Triplet: {triplet_loss.item():.6f}, Total: {total_loss.item():.6f}")

        return total_loss, recon_loss, kl_loss, triplet_loss


import wandb


def main():
    # Parse command-line arguments for model checkpoint resuming
    parser = argparse.ArgumentParser(description="Train CVAE_3D model with flat input adapter")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to checkpoint (.pt file) to resume training from (optional).")
    args = parser.parse_args()

    # Read model file for notes
    model_filepath = os.path.join(parent_dir, 'encoder', 'model_CVAE_3D_01.py')
    notes_content = ""
    try:
        with open(model_filepath, 'r') as file:
            notes_content = file.read()
    except Exception as e:
        notes_content = f"Failed to load model file: {e}"

    # Start wandb run
    wandb_run = wandb.init(
        project=MODEL_NAME,
        name=f"{MODEL_NAME}_run",
        config={
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model": MODEL_NAME,
            "latent_dim": 47,  # Explicitly track latent dimension
            "grid_size": 8  # Track the 3D grid size
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
    model = CVAE_3D_Adapter().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
        epoch_kl_loss = 0  # KL divergence instead of MMD
        epoch_triplet_loss = 0
        num_batches = 0

        epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx in range(BATCHES_PER_EPOCH):
            logger.debug(f"Processing batch {batch_idx + 1}/{BATCHES_PER_EPOCH} in epoch {epoch + 1}")
            
            # Get batch from dataloader
            batch = dataloader.get_batch(NUMBER_OF_ROWS=BATCH_SIZE)
            velocity_data = batch['velocity_data']
            
            # Log batch statistics
            logger.debug(f"Batch shape: {velocity_data.shape}")
            
            # Convert numpy array to tensor and move to device
            x = torch.tensor(velocity_data, dtype=torch.float32).to(device)
            logger.debug(f"Input tensor moved to device: {device}")

            try:
                # Forward pass
                logger.debug("Starting forward pass")
                recon_x, mu, logvar, z_representation = model(x)
                logger.debug("Completed forward pass")

                # Calculate loss
                logger.debug("Calculating loss")
                loss, recon_loss, kl_loss, triplet_loss = model.loss_function(recon_x, x, mu, logvar)
                logger.debug(f"Loss calculated: {loss.item()}")

                # Backward pass and optimize
                logger.debug("Starting backward pass")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.debug("Completed backward pass and optimization step")

                # Update metrics
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_triplet_loss += triplet_loss.item()
                num_batches += 1
                global_step += 1

                # Log to tensorboard
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                writer.add_scalar('Loss/recon_step', recon_loss.item(), global_step)
                writer.add_scalar('Loss/kl_step', kl_loss.item(), global_step)
                writer.add_scalar('Loss/triplet_step', triplet_loss.item(), global_step)

                # Log metrics to wandb
                wandb.log({
                    "Loss/train_step": loss.item(),
                    "Loss/recon_step": recon_loss.item(),
                    "Loss/kl_step": kl_loss.item(),
                    "Loss/triplet_step": triplet_loss.item(),
                    "global_step": global_step
                })
                
                # Periodic batch status
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{BATCHES_PER_EPOCH}, Loss: {loss.item():.6f}")
                
            except Exception as e:
                logger.error(f"Error during training loop: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise

        # Compute epoch average losses
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        avg_triplet_loss = epoch_triplet_loss / num_batches

        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Recon: {avg_recon_loss:.6f}, "
                    f"KL: {avg_kl_loss:.6f}, "
                    f"Triplet: {avg_triplet_loss:.6f}, "
                    f"Time: {epoch_time:.2f}s")

        # Log to tensorboard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch)
        writer.add_scalar('Loss/kl_epoch', avg_kl_loss, epoch)
        writer.add_scalar('Loss/triplet_epoch', avg_triplet_loss, epoch)

        # Epoch summary to wandb
        wandb.log({
            "Loss/train_epoch": avg_loss,
            "Loss/recon_epoch": avg_recon_loss,
            "Loss/kl_epoch": avg_kl_loss,
            "Loss/triplet_epoch": avg_triplet_loss,
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