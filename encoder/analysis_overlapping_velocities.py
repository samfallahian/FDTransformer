#!/usr/bin/env python3
"""
analysis_overlapping_velocities.py - Analyze velocity distributions at overlapping coordinates.

PURPOSE:
This script analyzes how different velocity components (vx, vy, vz) are distributed when
the same spatial coordinate appears in multiple 5x5x5 cubes. It reads the output files
from generate_decoded_velocity_analysis.py and creates histograms showing the velocity
distributions for each unique spatial position.

PROCESS:
1. Loads the decoded velocity and position mapping files
2. For each unique spatial coordinate (x, y, z):
   - Finds all occurrences of that coordinate across all cubes
   - Gathers the corresponding vx, vy, vz values
   - Creates a histogram with vx (red), vy (green), vz (blue)

OUTPUT:
Box/whisker plots showing velocity distributions (25th, 50th, 75th percentiles with outliers)
for overlapping coordinates, with vx (red), vy (green), vz (blue).

USAGE:
    python encoder/analysis_overlapping_velocities.py --dataset 7p2 --time 1000
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
from collections import defaultdict

# Configuration
OUTPUT_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"
FIGURE_OUTPUT_DIR = "/Users/kkreth/PycharmProjects/cgan/encoder/velocity_overlap_analysis"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(dataset_name, time, output_dir):
    """
    Load the decoded velocity and position mapping files.

    Args:
        dataset_name: Dataset name (e.g., "7p2")
        time: Time step
        output_dir: Base directory for data files

    Returns:
        Tuple of (velocity_df, position_df)
    """
    data_path = Path(output_dir) / dataset_name

    velocity_file = data_path / f"df_decoded_velocity_{time:04d}.pkl.gz"
    position_file = data_path / f"df_position_mapping_{time:04d}.pkl.gz"

    if not velocity_file.exists():
        logger.error(f"Velocity file not found: {velocity_file}")
        return None, None

    if not position_file.exists():
        logger.error(f"Position file not found: {position_file}")
        return None, None

    logger.info(f"Loading velocity data from {velocity_file}")
    velocity_df = pd.read_pickle(velocity_file, compression='gzip')

    logger.info(f"Loading position mapping from {position_file}")
    position_df = pd.read_pickle(position_file, compression='gzip')

    logger.info(f"Loaded velocity data: {velocity_df.shape}")
    logger.info(f"Loaded position data: {position_df.shape}")

    return velocity_df, position_df


def gather_velocities_by_position(velocity_df, position_df):
    """
    Gather all velocities for each unique spatial position.

    Args:
        velocity_df: DataFrame with decoded velocities
        position_df: DataFrame with position mappings

    Returns:
        Dictionary mapping (x, y, z) -> {'vx': list, 'vy': list, 'vz': list}
    """
    logger.info("Gathering velocities by spatial position...")

    # Dictionary to store velocities for each position
    position_velocities = defaultdict(lambda: {'vx': [], 'vy': [], 'vz': []})

    # Iterate through each row (each cube)
    for idx in tqdm(range(len(velocity_df)), desc="Processing cubes"):
        vel_row = velocity_df.iloc[idx]
        pos_row = position_df.iloc[idx]

        # Process each of the 125 points in the cube
        for i in range(1, 126):
            x = pos_row[f'x_{i}']
            y = pos_row[f'y_{i}']
            z = pos_row[f'z_{i}']

            vx = vel_row[f'vx_{i}']
            vy = vel_row[f'vy_{i}']
            vz = vel_row[f'vz_{i}']

            position_key = (int(x), int(y), int(z))
            position_velocities[position_key]['vx'].append(vx)
            position_velocities[position_key]['vy'].append(vy)
            position_velocities[position_key]['vz'].append(vz)

    logger.info(f"Found {len(position_velocities)} unique spatial positions")

    return position_velocities


def plot_velocity_histograms(position_velocities, dataset_name, time, output_dir):
    """
    Create box/whisker plots of velocity distributions for each position.
    Splits into multiple figures with 1000 positions each.

    Args:
        position_velocities: Dictionary of position -> velocities
        dataset_name: Dataset name
        time: Time step
        output_dir: Directory to save figures
    """
    logger.info("Creating velocity box plots...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort positions by a consistent order (x, then y, then z)
    sorted_positions = sorted(position_velocities.keys())

    logger.info(f"Creating box plots for {len(sorted_positions)} positions")

    # Split into batches of 1000 positions
    batch_size = 1000
    num_batches = (len(sorted_positions) + batch_size - 1) // batch_size

    logger.info(f"Will create {num_batches} figure(s) with up to {batch_size} positions each")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sorted_positions))
        batch_positions = sorted_positions[start_idx:end_idx]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_positions)} positions)")

        # Prepare data for box plots
        # Each position gets 3 boxes (vx, vy, vz)
        data_to_plot = []
        position_labels = []
        colors = []

        for pos in batch_positions:
            vx_vals = position_velocities[pos]['vx']
            vy_vals = position_velocities[pos]['vy']
            vz_vals = position_velocities[pos]['vz']

            data_to_plot.extend([vx_vals, vy_vals, vz_vals])
            position_labels.extend([f'{pos}\nvx', f'{pos}\nvy', f'{pos}\nvz'])
            colors.extend(['red', 'green', 'blue'])

        # Create figure
        fig, ax = plt.subplots(figsize=(max(16, len(batch_positions) * 0.6), 8))

        # Create box plots
        bp = ax.boxplot(data_to_plot,
                        labels=position_labels,
                        patch_artist=True,
                        showfliers=True,
                        whis=[25, 75])

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Rotate x-axis labels for readability
        ax.set_xlabel('Position and Velocity Component')
        ax.set_ylabel('Velocity Value')
        ax.set_title(f'Velocity Distributions by Position - {dataset_name} t={time}\n' +
                     f'Positions {start_idx + 1}-{end_idx} (25th, 50th, 75th percentiles with outliers)')
        plt.xticks(rotation=90, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save figure with batch number
        output_file = output_path / f"velocity_boxplots_{dataset_name}_{time:04d}_batch_{batch_idx + 1:03d}.png"
        logger.info(f"Saving figure to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Batch {batch_idx + 1} saved successfully")

    logger.info(f"All {num_batches} box plot(s) saved successfully")


def main():
    """Main processing function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze overlapping velocity data')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., "7p2")')
    parser.add_argument('--time', type=int, required=True, help='Time step (e.g., 1000)')
    args = parser.parse_args()

    # Load data
    velocity_df, position_df = load_data(args.dataset, args.time, OUTPUT_DIR)

    if velocity_df is None or position_df is None:
        logger.error("Failed to load data")
        return 1

    # Gather velocities by position
    position_velocities = gather_velocities_by_position(velocity_df, position_df)

    # Plot histograms
    plot_velocity_histograms(position_velocities, args.dataset, args.time, FIGURE_OUTPUT_DIR)

    logger.info("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
