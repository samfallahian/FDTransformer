#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Position Error Analysis script for WAE model 01.
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
from encoder.model_WAE_01 import WAE
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


def compute_errors_by_position(model, dataloader, device, num_samples):
    """
    Process samples and track the absolute error at each position.

    Returns:
        errors_by_position: A numpy array of shape [num_positions, num_samples] containing
                           the absolute error at each position across all samples.
    """
    logger.info(f"Computing errors across {num_samples} samples...")

    # Initialize array to store errors by position
    num_positions = POINTS_PER_SAMPLE * 3  # x, y, z for each point
    errors_by_position = np.zeros((num_positions, num_samples))

    samples_processed = 0
    batch_count = 0

    # Use tqdm for progress bar
    progress_bar = tqdm(total=num_samples, desc="Processing samples")

    while samples_processed < num_samples:
        # Get batch from dataloader
        batch = dataloader.get_batch(NUMBER_OF_ROWS=BATCH_SIZE)
        velocity_data = batch['velocity_data']

        # Convert numpy array to tensor and move to device
        x = torch.tensor(velocity_data, dtype=torch.float32).to(device)

        # Forward pass (no need to track gradients during validation)
        with torch.no_grad():
            recon_x, z = model(x)

        # Calculate absolute error for each position
        abs_error = torch.abs(x - recon_x)

        # Move error tensor to CPU and convert to numpy
        abs_error_np = abs_error.cpu().numpy()

        # Add errors to our tracking array
        batch_size = abs_error_np.shape[0]
        samples_to_add = min(batch_size, num_samples - samples_processed)

        # For each sample in the batch
        for i in range(samples_to_add):
            # For each position, store the error
            errors_by_position[:, samples_processed + i] = abs_error_np[i, :]

        samples_processed += samples_to_add
        batch_count += 1

        # Update progress bar
        progress_bar.update(samples_to_add)

    progress_bar.close()
    logger.info(f"Processed {batch_count} batches to reach {samples_processed} samples")

    return errors_by_position


def count_threshold_errors(errors_by_position, threshold=ERROR_THRESHOLD):
    """
    Count how many times the absolute error exceeds the threshold for each position.

    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples]
        threshold: The error threshold to count

    Returns:
        A numpy array of shape [num_positions] with counts of threshold exceedances
    """
    return np.sum(errors_by_position > threshold, axis=1)


def create_whisker_plots(errors_by_position, output_path):
    """
    Create whisker plots showing error distributions for each position.

    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples]
        output_path: Where to save the plot images
    """
    logger.info("Creating whisker plots...")

    # Total number of positions
    num_positions = errors_by_position.shape[0]

    # Since 375 positions is a lot for one plot, let's create multiple plots
    # We'll create separate plots for x, y, z components
    components = ['X', 'Y', 'Z']
    points_per_sample = num_positions // 3

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Create one big figure for an overview of all positions
    plt.figure(figsize=(20, 10))
    plt.boxplot(errors_by_position.T, showfliers=True, flierprops={'markersize': 1})  # Show outliers
    plt.xlabel('Position Index')
    plt.ylabel('Absolute Error')
    plt.title(f'Error Distribution Across All {num_positions} Positions')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_path, 'all_positions_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate plots for each component (X, Y, Z) with 25 points per plot
    points_per_plot = 25
    for component_idx, component in enumerate(components):
        num_plots = (points_per_sample + points_per_plot - 1) // points_per_plot

        for plot_idx in range(num_plots):
            start_point = plot_idx * points_per_plot
            end_point = min(start_point + points_per_plot, points_per_sample)

            if start_point >= points_per_sample:
                break

            # Calculate position indices for this component and range
            position_indices = []
            for i in range(start_point, end_point):
                position_indices.append(i * 3 + component_idx)

            # Extract data for these positions
            plot_data = errors_by_position[position_indices, :].T

            # Create the plot
            plt.figure(figsize=(15, 8))
            # Use showfliers=True to display outlier dots
            sns.boxplot(data=plot_data, showfliers=True, flierprops={'markersize': 1.5})

            # Set labels for the x-axis showing actual point indices
            plt.xticks(range(len(position_indices)),
                       [f"{i // 3}" for i in range(start_point * 3, end_point * 3, 3)],
                       rotation=90)

            plt.xlabel('Point Index')
            plt.ylabel('Absolute Error')
            plt.title(f'{component} Component Errors - Points {start_point} to {end_point - 1}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(output_path, f'{component}_points_{start_point}_to_{end_point - 1}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    # Create heatmap of median errors for a more compact visualization
    median_errors = np.median(errors_by_position, axis=1).reshape(3, -1)

    plt.figure(figsize=(20, 6))
    sns.heatmap(median_errors, cmap='viridis', annot=False)
    plt.xlabel('Point Index')
    plt.ylabel('Component (X=0, Y=1, Z=2)')
    plt.title('Median Absolute Error by Position')
    plt.savefig(os.path.join(output_path, 'median_error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save the raw error data for future analysis
    np.save(os.path.join(output_path, 'errors_by_position.npy'), errors_by_position)

    logger.info(f"Saved plots and data to {output_path}")


def display_threshold_errors(errors_by_position, threshold=ERROR_THRESHOLD):
    """
    Display a table showing how many times errors exceed the threshold for each position.

    Args:
        errors_by_position: A numpy array of shape [num_positions, num_samples]
        threshold: The error threshold to count
    """
    # Count threshold exceedances for each position
    threshold_counts = count_threshold_errors(errors_by_position, threshold)
    total_samples = errors_by_position.shape[1]

    # Calculate percentages
    threshold_percentages = (threshold_counts / total_samples) * 100

    # Total number of positions
    num_positions = errors_by_position.shape[0]
    points_per_sample = num_positions // 3
    components = ['X', 'Y', 'Z']

    # Create table data
    table_data = []

    # Format table with components and points
    for point_idx in range(points_per_sample):
        row = [point_idx]
        for component_idx, component in enumerate(components):
            position_idx = point_idx * 3 + component_idx
            count = threshold_counts[position_idx]
            percentage = threshold_percentages[position_idx]
            row.append(f"{count} ({percentage:.2f}%)")
        table_data.append(row)

    headers = ["Point", "X Errors", "Y Errors", "Z Errors"]

    print("\n" + "=" * 80)
    print(f"POSITIONS EXCEEDING ERROR THRESHOLD {threshold:.4f}")
    print("=" * 80)
    print(f"Total samples analyzed: {total_samples}")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Find top 10 positions with highest error rates
    top_indices = np.argsort(threshold_counts)[-10:][::-1]

    print("\nTop 10 Positions with Highest Error Rates:")
    top_table = []
    for pos_idx in top_indices:
        point_idx = pos_idx // 3
        component_idx = pos_idx % 3
        component = components[component_idx]
        count = threshold_counts[pos_idx]
        percentage = threshold_percentages[pos_idx]
        top_table.append([f"Point {point_idx}", component, count, f"{percentage:.2f}%"])

    top_headers = ["Position", "Component", "Error Count", "Percentage"]
    print(tabulate(top_table, headers=top_headers, tablefmt="grid"))

    # Summary statistics
    total_errors = np.sum(threshold_counts)
    total_positions = num_positions * total_samples
    overall_error_rate = (total_errors / total_positions) * 100

    print(
        f"\nOverall Error Rate: {total_errors} errors out of {total_positions} position measurements ({overall_error_rate:.4f}%)")

    # Component-wise statistics
    for component_idx, component in enumerate(components):
        component_positions = np.arange(component_idx, num_positions, 3)
        component_counts = threshold_counts[component_positions]
        component_total = np.sum(component_counts)
        component_rate = (component_total / (points_per_sample * total_samples)) * 100
        print(f"{component} Component Error Rate: {component_total} errors ({component_rate:.4f}%)")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze position-wise errors in WAE model")
    parser.add_argument('model_path', type=str,
                        help="Path to model checkpoint (.pt file) to validate.")
    parser.add_argument('--output', type=str, default='position_error_analysis',
                        help="Directory to save output plots and data")
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES,
                        help=f"Number of samples to process (default: {NUM_SAMPLES})")
    parser.add_argument('--threshold', type=float, default=ERROR_THRESHOLD,
                        help=f"Error threshold for counting errors (default: {ERROR_THRESHOLD})")
    args = parser.parse_args()

    # Log the model path we're going to use
    logger.info(f"Analyzing model: {args.model_path}")

    # Update number of samples if specified
    num_samples = args.samples
    logger.info(f"Will process {num_samples} samples")

    # Setup device (GPU/CPU)
    device = setup_device()

    # Load the model
    model = load_model(args.model_path, device)

    # Initialize data loader
    logger.info(f"Initializing data loader from {preferences.training_data_path}")
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=BATCH_SIZE,
        num_workers=4,
        cache_size=5,
        shuffle=True
    )
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")

    # Compute errors by position
    errors_by_position = compute_errors_by_position(model, dataloader, device, num_samples)

    # Create whisker plots
    create_whisker_plots(errors_by_position, args.output)

    # Display threshold errors table
    display_threshold_errors(errors_by_position, args.threshold)

    # Print summary statistics
    print("\n" + "=" * 80)
    print(f"POSITION ERROR ANALYSIS RESULTS FOR MODEL: {args.model_path}")
    print("=" * 80)

    # Calculate overall statistics
    mean_error = np.mean(errors_by_position)
    median_error = np.median(errors_by_position)
    max_error = np.max(errors_by_position)

    print(f"Overall Mean Absolute Error: {mean_error:.6f}")
    print(f"Overall Median Absolute Error: {median_error:.6f}")
    print(f"Overall Maximum Absolute Error: {max_error:.6f}")

    # Calculate per-component statistics
    components = ['X', 'Y', 'Z']
    points_per_sample = errors_by_position.shape[0] // 3

    for i, component in enumerate(components):
        component_errors = errors_by_position[i::3, :]
        component_mean = np.mean(component_errors)
        component_median = np.median(component_errors)
        component_max = np.max(component_errors)

        print(f"\n{component} Component Statistics:")
        print(f"  Mean Absolute Error: {component_mean:.6f}")
        print(f"  Median Absolute Error: {component_median:.6f}")
        print(f"  Maximum Absolute Error: {component_max:.6f}")

    # Find positions with highest errors
    median_by_position = np.median(errors_by_position, axis=1)
    top_positions = np.argsort(median_by_position)[-10:][::-1]

    print("\nTop 10 Positions with Highest Median Error:")
    for pos in top_positions:
        point_idx = pos // 3
        component_idx = pos % 3
        component = components[component_idx]
        median_error = median_by_position[pos]
        print(f"  Point {point_idx}, {component} component: {median_error:.6f}")

    print(f"\nDetailed plots saved to: {args.output}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()