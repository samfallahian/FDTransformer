#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position Error Analysis script for a pre-trained VAE model.
This script tracks absolute errors at each position across a large number of samples.
It processes 100,000 encode-decode operations and creates a whisker plot
of the error distributions at each position.
"""

import os
import sys
import logging
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our custom modules
from encoder.model_VAE_01 import SpatialAwareVAE, EnhancedSpatialVAE
from EfficientDataLoader import EfficientDataLoader

# Figure out the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preferences_path = os.path.join(project_root, "experiment.preferences")
from Ordered_001_Initialize import HostPreferences

# When creating preferences, pass the resolved path:
preferences = HostPreferences(filename=preferences_path)

# Validation configuration constants
BATCH_SIZE = 100  # Larger batch size for efficiency
NUM_SAMPLES = 10000  # Total samples to process
POINTS_PER_SAMPLE = 125  # Each point has x,y,z, so 375 total values
ERROR_THRESHOLD = 0.02  # Threshold for counting errors

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
    """Load a pre-trained VAE model from the specified checkpoint file."""
    logger.info(f"Loading model from {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Detect if the model is enhanced based on the keys in the state_dict
        is_enhanced = any(key.startswith("positional_encoding") or key.startswith("self_attention")
                          for key in checkpoint["model_state_dict"].keys())
        model_type = "EnhancedSpatialVAE" if is_enhanced else "SpatialAwareVAE"
        logger.info(f"Detected model architecture: {model_type}")

        # Initialize the appropriate model
        if is_enhanced:
            model = EnhancedSpatialVAE().to(device)
        else:
            model = SpatialAwareVAE().to(device)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Successfully loaded model state_dict.")

        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)


def compute_errors_by_position(model, dataloader, device, num_samples):
    """
    Process samples and track the absolute error at each position.

    Returns:
        errors_by_position: A numpy array of shape [num_positions, num_samples] containing
                           the absolute error at each position across all samples.
    """
    logger.info(f"Computing errors across {num_samples} samples...")

    # Initialize array to store errors
    num_positions = POINTS_PER_SAMPLE * 3  # x, y, z for each point
    errors_by_position = np.zeros((num_positions, num_samples))

    samples_processed = 0
    batch_count = 0

    progress_bar = tqdm(total=num_samples, desc="Processing samples")

    while samples_processed < num_samples:
        batch = dataloader.get_batch(NUMBER_OF_ROWS=BATCH_SIZE)
        velocity_data = batch['velocity_data']

        # Convert to tensor and move to device
        x = torch.tensor(velocity_data, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstruction, mu, logvar = model(x)

        # Absolute error
        abs_error = torch.abs(x - reconstruction).cpu().numpy()

        # Store errors
        batch_size = abs_error.shape[0]
        samples_to_add = min(batch_size, num_samples - samples_processed)

        for i in range(samples_to_add):
            errors_by_position[:, samples_processed + i] = abs_error[i, :]

        samples_processed += samples_to_add
        batch_count += 1
        progress_bar.update(samples_to_add)

    progress_bar.close()
    logger.info(f"Processed {batch_count} batches for {samples_processed} samples")

    return errors_by_position


def count_threshold_errors(errors_by_position, threshold=ERROR_THRESHOLD):
    """
    Count the number of times the absolute error exceeds the threshold for each position.
    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples].
        threshold: The error threshold to count.
    Returns:
        A numpy array of shape [num_positions] with counts of exceedances.
    """
    return np.sum(errors_by_position > threshold, axis=1)


def create_whisker_plots(errors_by_position, output_path):
    """
    Create whisker plots showing error distributions for each position.
    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples].
        output_path: Output directory for plots and data.
    """
    logger.info("Creating whisker plots...")

    # Number of positions
    num_positions = errors_by_position.shape[0]

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Overview plot for all positions
    plt.figure(figsize=(20, 10))
    plt.boxplot(errors_by_position.T, showfliers=True, flierprops={'markersize': 1.5})
    plt.xlabel('Position Index')
    plt.ylabel('Absolute Error')
    plt.title('Error Distribution Across All Positions')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'error_distribution_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw error data
    np.save(os.path.join(output_path, 'errors_by_position.npy'), errors_by_position)
    logger.info(f"Whisker plots and error data saved to {output_path}")


def display_threshold_errors(errors_by_position, threshold=ERROR_THRESHOLD):
    """
    Display a table showing how often errors exceed the threshold for each position.
    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples].
        threshold: The error threshold for counting.
    """
    logger.info(f"Counting threshold exceedances for threshold={threshold}")

    threshold_counts = count_threshold_errors(errors_by_position, threshold)
    total_samples = errors_by_position.shape[1]

    # Calculate percentages
    threshold_percentages = (threshold_counts / total_samples) * 100

    # Number of points and components
    num_positions = errors_by_position.shape[0]
    points_per_sample = num_positions // 3
    components = ['X', 'Y', 'Z']

    # Tabulate results
    table_data = []
    for point_idx in range(points_per_sample):
        row = [point_idx]
        for comp_idx, comp in enumerate(components):
            pos_idx = point_idx * 3 + comp_idx
            count = threshold_counts[pos_idx]
            percentage = threshold_percentages[pos_idx]
            row.append(f"{count} ({percentage:.2f}%)")
        table_data.append(row)

    headers = ["Point"] + components
    print(f"\n{'=' * 80}\nERROR COUNTS ABOVE THRESHOLD ({threshold:.2f})\n{'=' * 80}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Display summary
    total_errors = np.sum(threshold_counts)
    overall_error_rate = (total_errors / (num_positions * total_samples)) * 100
    print(f"\nOverall error rate: {overall_error_rate:.4f}% ({total_errors}/{num_positions * total_samples})")


def main():
    """
    Main function for position-wise error analysis.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze position-wise errors in VAE model.")
    parser.add_argument('model_path', type=str, help="Path to the model checkpoint file.")
    parser.add_argument('--output', type=str, default='vae_position_analysis', help="Output directory for results.")
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES, help=f"Number of samples to process.")
    parser.add_argument('--threshold', type=float, default=ERROR_THRESHOLD, help=f"Error threshold for analysis.")
    args = parser.parse_args()

    # Log start
    logger.info(f"Analyzing model: {args.model_path}")

    # Set up device
    device = setup_device()

    # Load model
    model = load_model(args.model_path, device)

    # Set up DataLoader
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=BATCH_SIZE,
        num_workers=4,
        cache_size=5,
        shuffle=True
    )

    logger.info(f"Processing {args.samples} samples for error analysis.")
    errors_by_position = compute_errors_by_position(model, dataloader, device, args.samples)

    # Generate visualizations and statistics
    create_whisker_plots(errors_by_position, args.output)
    display_threshold_errors(errors_by_position, args.threshold)

    # Print summary
    overall_mean_error = np.mean(errors_by_position)
    overall_median_error = np.median(errors_by_position)
    overall_max_error = np.max(errors_by_position)
    print(f"\n{'=' * 80}\nANALYSIS SUMMARY\n{'=' * 80}")
    print(f"Mean Error: {overall_mean_error:.6f}")
    print(f"Median Error: {overall_median_error:.6f}")
    print(f"Max Error: {overall_max_error:.6f}")
    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()