#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation script for WAE model 01.
This script validates a trained WAE model by loading a specified checkpoint file.
It samples 10 random batches, displays a representative example from each batch,
and prints input/output data and reconstruction errors.
"""

import os
import sys
import logging
import torch
import numpy as np
import argparse
from tabulate import tabulate

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

# Validation configuration constants
BATCH_SIZE = 20
NUM_BATCHES = 10

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
    """Set up the computing device (GPU/CPU) for validation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using NVIDIA GPU: {device_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected, using CPU for validation")
    return device

def load_model(model_path, device):
    """Load a trained WAE model from the specified checkpoint file."""
    logger.info(f"Loading model from {model_path}")
    
    # Initialize the model
    model = WAE().to(device)
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract the model state dict from the checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model from state_dict")
        else:
            # Fallback if the structure is different
            model.load_state_dict(checkpoint)
            logger.info("Loaded model directly from checkpoint")
            
        logger.info(f"Successfully loaded model with latent dimension {model.fc4.out_features}")
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def calculate_statistics(original, reconstructed):
    """Calculate and return statistics about the reconstruction error."""
    # Reshape to group x,y,z components together
    original_grouped = original.view(-1, int(original.nelement()/3), 3)
    recon_grouped = reconstructed.view(-1, int(reconstructed.nelement()/3), 3)
    
    # Calculate absolute error
    abs_error = torch.abs(original_grouped - recon_grouped)
    
    # Calculate squared error
    squared_error = (original_grouped - recon_grouped) ** 2
    
    # Calculate squared Euclidean distances for each sample
    squared_euclidean_per_point = torch.sum(squared_error, dim=2)  # Sum across x,y,z dimensions
    squared_euclidean_per_sample = torch.mean(squared_euclidean_per_point, dim=1)  # Mean across points
    
    # Calculate statistics
    stats = {
        'mean_abs_error': torch.mean(abs_error).item(),
        'max_abs_error': torch.max(abs_error).item(),
        'mean_squared_error': torch.mean(squared_error).item(),
        'max_point_error': torch.max(torch.sum(squared_error, dim=2)).sqrt().item(),
        'log_cosh_loss': WAE.log_cosh_loss(original_grouped, recon_grouped).item(),
        'squared_euclidean_distances': squared_euclidean_per_sample.cpu().tolist()
    }
    
    return stats

def format_point(point):
    """Format a 3D point (x,y,z) for display."""
    return f"({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})"

def visualize_sample(original, reconstructed, index=0, max_points=5):
    """Visualize and compare original and reconstructed data for a single sample."""
    # Get batch size to ensure we don't try to access an out-of-bounds index
    batch_size = original.size(0)
    
    # Check if the index is valid
    if index >= batch_size:
        logger.warning(f"Index {index} is out of bounds for batch size {batch_size}. Using index 0 instead.")
        index = 0
    
    # Extract the sample at the given index first, then reshape
    original_sample = original[index]
    recon_sample = reconstructed[index]
    
    # Reshape to group x,y,z components together
    points_per_sample = original_sample.nelement() // 3
    original_grouped = original_sample.view(points_per_sample, 3)
    recon_grouped = recon_sample.view(points_per_sample, 3)
    
    # Calculate error
    error = torch.abs(original_grouped - recon_grouped)
    
    # Create a table with formatted data
    table_data = []
    for i in range(min(max_points, len(original_grouped))):
        table_data.append([
            i, 
            format_point(original_grouped[i]),
            format_point(recon_grouped[i]),
            format_point(error[i])
        ])
    
    # Add the 62nd point (referenced in the loss function)
    if len(original_grouped) > 62:
        table_data.append(["...", "...", "...", "..."])
        table_data.append([
            62, 
            format_point(original_grouped[62]),
            format_point(recon_grouped[62]),
            format_point(error[62])
        ])
    
    headers = ["Point", "Original", "Reconstructed", "Absolute Error"]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate WAE model 01")
    parser.add_argument('model_path', type=str, 
                        help="Path to model checkpoint (.pt file) to validate.")
    args = parser.parse_args()
    
    # Log the model path we're going to use
    logger.info(f"Validating model: {args.model_path}")
    
    # Setup device (GPU/CPU)
    device = setup_device()
    
    # Load the model
    model = load_model(args.model_path, device)
    
    # Initialize data loader
    logger.info(f"Initializing data loader from {preferences.training_data_path}")
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=BATCH_SIZE,
        num_workers=4,  # Fewer workers for validation
        cache_size=5,   # Smaller cache for validation
        shuffle=True
    )
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")
    
    # Sample batches and evaluate
    all_stats = []
    
    print("\n" + "="*80)
    print(f"VALIDATION RESULTS FOR MODEL: {args.model_path}")
    print("="*80)
    
    for batch_idx in range(NUM_BATCHES):
        logger.info(f"Processing batch {batch_idx+1}/{NUM_BATCHES}")
        
        # Get batch from dataloader
        batch = dataloader.get_batch(NUMBER_OF_ROWS=BATCH_SIZE)
        velocity_data = batch['velocity_data']
        
        # Convert numpy array to tensor and move to device
        x = torch.tensor(velocity_data, dtype=torch.float32).to(device)
        
        # Forward pass (no need to track gradients during validation)
        with torch.no_grad():
            recon_x, z = model(x)
        
        # Calculate statistics
        batch_stats = calculate_statistics(x, recon_x)
        all_stats.append(batch_stats)
        
        # Select a random sample from the batch to visualize
        sample_idx = np.random.randint(0, x.shape[0])
        
        # Print batch summary
        print(f"\nBatch {batch_idx+1}/{NUM_BATCHES}:")
        print(f"  Mean Absolute Error: {batch_stats['mean_abs_error']:.6f}")
        print(f"  Max Absolute Error: {batch_stats['max_abs_error']:.6f}")
        print(f"  Mean Squared Error: {batch_stats['mean_squared_error']:.6f}")
        print(f"  Max Point Error: {batch_stats['max_point_error']:.6f}")
        print(f"  Log-Cosh Loss: {batch_stats['log_cosh_loss']:.6f}")
        
        # Print squared Euclidean distances in a nicely formatted table
        print("\n  Squared Euclidean Distances for each sample:")
        distances = batch_stats['squared_euclidean_distances']
        
        # Create a formatted table for squared Euclidean distances
        table_data = []
        for i in range(0, len(distances), 4):  # Display 4 values per row
            row = [f"Sample {i+j+1}: {distances[i+j]:.6f}" for j in range(min(4, len(distances)-i))]
            table_data.append(row)
        
        headers = [f"Batch {batch_idx+1} Distances"] + [""] * (len(table_data[0])-1) if table_data else []
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Visualize a random sample from this batch
        print(f"\nSample {sample_idx} from Batch {batch_idx+1}:")
        visualization = visualize_sample(x, recon_x, index=sample_idx)
        print(visualization)
        
        # Print latent space representation for this sample
        latent_repr = z[sample_idx].cpu().numpy()
        print(f"\nLatent Representation (dimension {len(latent_repr)}):")
        latent_str = ", ".join([f"{val:.4f}" for val in latent_repr[:10]])
        print(f"  [{latent_str}, ...]")
        
        print("\n" + "-"*80)
    
    # Calculate and print overall statistics
    print("\nOVERALL STATISTICS:")
    overall_stats = {
        'mean_abs_error': np.mean([s['mean_abs_error'] for s in all_stats]),
        'max_abs_error': np.max([s['max_abs_error'] for s in all_stats]),
        'mean_squared_error': np.mean([s['mean_squared_error'] for s in all_stats]),
        'max_point_error': np.max([s['max_point_error'] for s in all_stats]),
        'log_cosh_loss': np.mean([s['log_cosh_loss'] for s in all_stats])
    }
    
    print(f"  Mean Absolute Error: {overall_stats['mean_abs_error']:.6f}")
    print(f"  Max Absolute Error: {overall_stats['max_abs_error']:.6f}")
    print(f"  Mean Squared Error: {overall_stats['mean_squared_error']:.6f}")
    print(f"  Max Point Error: {overall_stats['max_point_error']:.6f}")
    print(f"  Log-Cosh Loss: {overall_stats['log_cosh_loss']:.6f}")
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()