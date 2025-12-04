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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

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


def compute_rmse_statistics(position_velocities):
    """
    Compute RMSE statistics for each position and velocity component.

    For each position that appears in multiple cubes, compute the RMSE
    (Root Mean Square Error) of the decoded velocities relative to their mean.
    This measures the consistency/variation in how the autoencoder decodes
    the same spatial position when it appears in different contexts.

    Args:
        position_velocities: Dictionary of position -> velocities

    Returns:
        Dictionary with RMSE statistics and per-position RMSE values
    """
    logger.info("Computing RMSE statistics for overlapping positions...")

    rmse_vx_list = []
    rmse_vy_list = []
    rmse_vz_list = []

    positions_with_rmse = []

    for pos, velocities in position_velocities.items():
        vx_vals = np.array(velocities['vx'])
        vy_vals = np.array(velocities['vy'])
        vz_vals = np.array(velocities['vz'])

        # Only compute RMSE for positions that appear multiple times
        if len(vx_vals) > 1:
            # Compute mean for each component
            vx_mean = np.mean(vx_vals)
            vy_mean = np.mean(vy_vals)
            vz_mean = np.mean(vz_vals)

            # Compute RMSE: sqrt(mean((values - mean)^2))
            rmse_vx = np.sqrt(np.mean((vx_vals - vx_mean) ** 2))
            rmse_vy = np.sqrt(np.mean((vy_vals - vy_mean) ** 2))
            rmse_vz = np.sqrt(np.mean((vz_vals - vz_mean) ** 2))

            rmse_vx_list.append(rmse_vx)
            rmse_vy_list.append(rmse_vy)
            rmse_vz_list.append(rmse_vz)

            positions_with_rmse.append({
                'position': pos,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'rmse_vx': rmse_vx,
                'rmse_vy': rmse_vy,
                'rmse_vz': rmse_vz,
                'sample_count': len(vx_vals),
                'mean_vx': vx_mean,
                'mean_vy': vy_mean,
                'mean_vz': vz_mean
            })

    # Convert to arrays for statistics
    rmse_vx_array = np.array(rmse_vx_list)
    rmse_vy_array = np.array(rmse_vy_list)
    rmse_vz_array = np.array(rmse_vz_list)

    # Compute overall statistics
    stats = {
        'vx': {
            'mean_rmse': np.mean(rmse_vx_array),
            'median_rmse': np.median(rmse_vx_array),
            'std_rmse': np.std(rmse_vx_array),
            'min_rmse': np.min(rmse_vx_array),
            'max_rmse': np.max(rmse_vx_array),
            'q25_rmse': np.percentile(rmse_vx_array, 25),
            'q75_rmse': np.percentile(rmse_vx_array, 75)
        },
        'vy': {
            'mean_rmse': np.mean(rmse_vy_array),
            'median_rmse': np.median(rmse_vy_array),
            'std_rmse': np.std(rmse_vy_array),
            'min_rmse': np.min(rmse_vy_array),
            'max_rmse': np.max(rmse_vy_array),
            'q25_rmse': np.percentile(rmse_vy_array, 25),
            'q75_rmse': np.percentile(rmse_vy_array, 75)
        },
        'vz': {
            'mean_rmse': np.mean(rmse_vz_array),
            'median_rmse': np.median(rmse_vz_array),
            'std_rmse': np.std(rmse_vz_array),
            'min_rmse': np.min(rmse_vz_array),
            'max_rmse': np.max(rmse_vz_array),
            'q25_rmse': np.percentile(rmse_vz_array, 25),
            'q75_rmse': np.percentile(rmse_vz_array, 75)
        },
        'total_positions': len(position_velocities),
        'positions_with_overlaps': len(positions_with_rmse)
    }

    logger.info(f"Computed RMSE for {len(positions_with_rmse)} positions with overlaps")
    logger.info(f"Mean RMSE - Vx: {stats['vx']['mean_rmse']:.6f}, "
                f"Vy: {stats['vy']['mean_rmse']:.6f}, "
                f"Vz: {stats['vz']['mean_rmse']:.6f}")

    return stats, positions_with_rmse


def plot_interactive_3d(position_velocities, dataset_name, time, output_dir):
    """
    Create interactive 3D scatter plots using Plotly showing RMSE (deltas from mean)
    for overlapping coordinates. Creates one HTML file with three subplots (vx, vy, vz).

    Args:
        position_velocities: Dictionary of position -> velocities
        dataset_name: Dataset name
        time: Time step
        output_dir: Directory to save figures
    """
    logger.info("Creating interactive 3D scatter plots...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Sort positions by a consistent order (x, then y, then z)
    sorted_positions = sorted(position_velocities.keys())

    logger.info(f"Creating 3D scatter for {len(sorted_positions)} positions")

    # Prepare data for plotting
    # For each position, compute RMSE (delta from mean)
    positions_x = []
    positions_y = []
    positions_z = []
    rmse_vx = []
    rmse_vy = []
    rmse_vz = []
    mean_vx = []
    mean_vy = []
    mean_vz = []
    sample_counts = []

    for pos in sorted_positions:
        vx_vals = np.array(position_velocities[pos]['vx'])
        vy_vals = np.array(position_velocities[pos]['vy'])
        vz_vals = np.array(position_velocities[pos]['vz'])

        positions_x.append(pos[0])
        positions_y.append(pos[1])
        positions_z.append(pos[2])

        # Compute mean (centroid value)
        vx_mean = np.mean(vx_vals)
        vy_mean = np.mean(vy_vals)
        vz_mean = np.mean(vz_vals)

        mean_vx.append(vx_mean)
        mean_vy.append(vy_mean)
        mean_vz.append(vz_mean)

        # Compute RMSE (delta from centroid)
        rmse_vx.append(np.sqrt(np.mean((vx_vals - vx_mean) ** 2)))
        rmse_vy.append(np.sqrt(np.mean((vy_vals - vy_mean) ** 2)))
        rmse_vz.append(np.sqrt(np.mean((vz_vals - vz_mean) ** 2)))

        sample_counts.append(len(vx_vals))

    # Create subplots
    subplot_titles = ('Vx RMSE (red)', 'Vy RMSE (green)', 'Vz RMSE (blue)')
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )

    # Vx scatter (red) - colored by RMSE
    trace_vx = go.Scatter3d(
        x=positions_x,
        y=positions_y,
        z=positions_z,
        mode='markers',
        marker=dict(
            size=4,
            color=rmse_vx,
            colorscale='Reds',
            opacity=0.6,
            colorbar=dict(x=0.28, len=0.9, title='RMSE Vx'),
            showscale=True
        ),
        text=[f'Pos: ({x}, {y}, {z})<br>Mean: {mv:.6f}<br>RMSE: {rmse:.6f}<br>Samples: {n}'
              for x, y, z, mv, rmse, n in zip(positions_x, positions_y, positions_z, mean_vx, rmse_vx, sample_counts)],
        hoverinfo='text',
        name='Vx'
    )

    # Vy scatter (green) - colored by RMSE
    trace_vy = go.Scatter3d(
        x=positions_x,
        y=positions_y,
        z=positions_z,
        mode='markers',
        marker=dict(
            size=4,
            color=rmse_vy,
            colorscale='Greens',
            opacity=0.6,
            colorbar=dict(x=0.63, len=0.9, title='RMSE Vy'),
            showscale=True
        ),
        text=[f'Pos: ({x}, {y}, {z})<br>Mean: {mv:.6f}<br>RMSE: {rmse:.6f}<br>Samples: {n}'
              for x, y, z, mv, rmse, n in zip(positions_x, positions_y, positions_z, mean_vy, rmse_vy, sample_counts)],
        hoverinfo='text',
        name='Vy'
    )

    # Vz scatter (blue) - colored by RMSE
    trace_vz = go.Scatter3d(
        x=positions_x,
        y=positions_y,
        z=positions_z,
        mode='markers',
        marker=dict(
            size=4,
            color=rmse_vz,
            colorscale='Blues',
            opacity=0.6,
            colorbar=dict(x=1.0, len=0.9, title='RMSE Vz'),
            showscale=True
        ),
        text=[f'Pos: ({x}, {y}, {z})<br>Mean: {mv:.6f}<br>RMSE: {rmse:.6f}<br>Samples: {n}'
              for x, y, z, mv, rmse, n in zip(positions_x, positions_y, positions_z, mean_vz, rmse_vz, sample_counts)],
        hoverinfo='text',
        name='Vz'
    )

    # Add traces to subplots
    fig.add_trace(trace_vx, row=1, col=1)
    fig.add_trace(trace_vy, row=1, col=2)
    fig.add_trace(trace_vz, row=1, col=3)

    # Update layout
    fig.update_layout(
        title_text=f'Velocity RMSE by Position - {dataset_name} t={time}<br>(RMSE = deviation from mean, interactive 3D)',
        width=1800,
        height=600,
        showlegend=False
    )

    # Update axes labels for all subplots
    for i in range(1, 4):
        fig.update_scenes(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            row=1, col=i
        )

    # Save as HTML
    output_file = output_path / f"velocity_3d_interactive_{dataset_name}_{time:04d}.html"
    logger.info(f"Saving interactive plot to {output_file}")
    pio.write_html(fig, str(output_file))

    logger.info(f"Interactive 3D plot saved successfully")


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
    parser.add_argument('--generate_png', action='store_true',
                        help='Generate PNG box plots (creates large files, disabled by default)')
    args = parser.parse_args()

    # Load data
    velocity_df, position_df = load_data(args.dataset, args.time, OUTPUT_DIR)

    if velocity_df is None or position_df is None:
        logger.error("Failed to load data")
        return 1

    # Gather velocities by position
    position_velocities = gather_velocities_by_position(velocity_df, position_df)

    # Compute RMSE statistics
    rmse_stats, positions_with_rmse = compute_rmse_statistics(position_velocities)

    # Save RMSE statistics to CSV
    output_path = Path(FIGURE_OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save overall statistics
    stats_file = output_path / f"rmse_statistics_{args.dataset}_{args.time:04d}.txt"
    with open(stats_file, 'w') as f:
        f.write(f"RMSE Statistics for {args.dataset} at time {args.time}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total unique positions: {rmse_stats['total_positions']}\n")
        f.write(f"Positions with overlaps: {rmse_stats['positions_with_overlaps']}\n\n")

        for component in ['vx', 'vy', 'vz']:
            f.write(f"\n{component.upper()} RMSE Statistics:\n")
            f.write("-" * 40 + "\n")
            for stat_name, stat_value in rmse_stats[component].items():
                f.write(f"  {stat_name:15s}: {stat_value:.8f}\n")

    logger.info(f"Saved RMSE statistics to {stats_file}")

    # Save per-position RMSE to CSV
    if positions_with_rmse:
        rmse_df = pd.DataFrame(positions_with_rmse)
        rmse_csv = output_path / f"rmse_per_position_{args.dataset}_{args.time:04d}.csv"
        rmse_df.to_csv(rmse_csv, index=False)
        logger.info(f"Saved per-position RMSE to {rmse_csv}")

    # Always generate interactive 3D plot (HTML)
    plot_interactive_3d(position_velocities, args.dataset, args.time, FIGURE_OUTPUT_DIR)

    # Optionally generate PNG box plots
    if args.generate_png:
        logger.info("Generating PNG box plots (this may take a while and create large files)...")
        plot_velocity_histograms(position_velocities, args.dataset, args.time, FIGURE_OUTPUT_DIR)
    else:
        logger.info("Skipping PNG generation (use --generate_png to enable)")

    logger.info("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
