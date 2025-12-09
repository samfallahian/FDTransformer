#!/usr/bin/env python3
"""
analysis_overlapping_velocities_mutual_information.py - Analyze correlation between RMSE and sample size.

PURPOSE:
This script analyzes whether higher RMSE values correlate with sample size (number of times
a position appears in multiple cubes). It uses mutual information to detect non-linear
relationships, along with traditional correlation metrics (Pearson and Spearman).

The analysis helps determine if positions that appear more frequently have systematically
higher or lower RMSE values, which could indicate encoding bias or sampling effects.

INPUTS:
- rmse_per_position_<dataset>_<time>.csv from analysis_overlapping_velocities.py
  Contains: x, y, z, rmse_vx, rmse_vy, rmse_vz, sample_count, mean_vx, mean_vy, mean_vz

OUTPUTS:
- Statistical analysis report (text file)
- Scatter plots with regression lines (RMSE vs sample_count)
- Mutual information scores
- Correlation coefficients (Pearson, Spearman)
- Binned analysis showing RMSE distribution across sample count ranges

USAGE:
    python encoder/analysis_overlapping_velocities_mutual_information.py --dataset 7p2 --time 1000
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from Ordered_001_Initialize import HostPreferences  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize host preferences to get correct paths
try:
    host_prefs = HostPreferences()
    metadata_path = Path(host_prefs.metadata_location)
    PROJECT_ROOT = metadata_path.parent.parent

    ANALYSIS_DIR = PROJECT_ROOT / "encoder" / "velocity_overlap_analysis"

    logger.info(f"Initialized paths from HostPreferences:")
    logger.info(f"  Analysis dir: {ANALYSIS_DIR}")
except Exception as e:
    logger.warning(f"Could not load HostPreferences, using default paths: {e}")
    ANALYSIS_DIR = "/Users/kkreth/PycharmProjects/cgan/encoder/velocity_overlap_analysis"


def load_rmse_data(dataset_name, time, analysis_dir):
    """
    Load the RMSE per position CSV file.

    Args:
        dataset_name: Dataset name (e.g., "7p2")
        time: Time step
        analysis_dir: Directory containing RMSE CSV files

    Returns:
        DataFrame with RMSE data
    """
    analysis_path = Path(analysis_dir)
    csv_file = analysis_path / f"rmse_per_position_{dataset_name}_{time:04d}.csv"

    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        return None

    logger.info(f"Loading RMSE data from {csv_file}")
    df = pd.read_csv(csv_file)

    logger.info(f"Loaded {len(df)} positions with RMSE data")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Sample count range: {df['sample_count'].min()} - {df['sample_count'].max()}")

    return df


def compute_mutual_information(rmse_values, sample_counts):
    """
    Compute mutual information between RMSE and sample count.

    Mutual information measures how much knowing the sample count reduces
    uncertainty about the RMSE value. Higher MI indicates stronger dependency.

    Args:
        rmse_values: Array of RMSE values
        sample_counts: Array of sample counts

    Returns:
        Mutual information score (in bits)
    """
    # Reshape for sklearn (expects 2D)
    X = sample_counts.reshape(-1, 1)
    y = rmse_values

    # Use mutual_info_regression for continuous target
    mi_score = mutual_info_regression(X, y, random_state=42)[0]

    return mi_score


def compute_correlations(df):
    """
    Compute correlation metrics between RMSE and sample_count.

    Args:
        df: DataFrame with RMSE data

    Returns:
        Dictionary with correlation results for each velocity component
    """
    logger.info("Computing correlation metrics...")

    results = {}

    for component in ['vx', 'vy', 'vz']:
        rmse_col = f'rmse_{component}'
        rmse_values = df[rmse_col].values
        sample_counts = df['sample_count'].values

        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = pearsonr(sample_counts, rmse_values)

        # Spearman correlation (monotonic relationship)
        spearman_r, spearman_p = spearmanr(sample_counts, rmse_values)

        # Mutual information (non-linear dependency)
        mi_score = compute_mutual_information(rmse_values, sample_counts)

        results[component] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mutual_info': mi_score,
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'rmse_median': np.median(rmse_values),
            'sample_count_mean': np.mean(sample_counts),
            'sample_count_std': np.std(sample_counts)
        }

        logger.info(f"{component.upper()} - Pearson r={pearson_r:.4f} (p={pearson_p:.4e}), "
                   f"Spearman r={spearman_r:.4f} (p={spearman_p:.4e}), "
                   f"MI={mi_score:.6f}")

    return results


def binned_analysis(df):
    """
    Analyze RMSE distribution across sample count bins.

    Groups positions by sample count ranges and computes RMSE statistics
    for each bin to visualize trends more clearly.

    Args:
        df: DataFrame with RMSE data

    Returns:
        Dictionary with binned statistics
    """
    logger.info("Performing binned analysis...")

    # Define bins based on sample count quartiles
    sample_counts = df['sample_count'].values
    quartiles = np.percentile(sample_counts, [0, 25, 50, 75, 100])

    logger.info(f"Sample count quartiles: {quartiles}")

    # Create bins - ensure uniqueness
    bins = np.unique([quartiles[0], quartiles[1], quartiles[2], quartiles[3], quartiles[4]])

    logger.info(f"Unique bins: {bins}")

    # Create labels for bins
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)]

    df['sample_bin'] = pd.cut(df['sample_count'], bins=bins, labels=bin_labels, include_lowest=True)

    binned_results = {}

    for component in ['vx', 'vy', 'vz']:
        rmse_col = f'rmse_{component}'

        bin_stats = df.groupby('sample_bin')[rmse_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max',
            ('q25', lambda x: np.percentile(x, 25)),
            ('q75', lambda x: np.percentile(x, 75))
        ])

        binned_results[component] = bin_stats

        logger.info(f"\n{component.upper()} RMSE by sample count bin:")
        logger.info(f"\n{bin_stats}")

    return binned_results, bin_labels


def plot_scatter_with_regression(df, dataset_name, time, output_dir):
    """
    Create scatter plots of RMSE vs sample_count with regression lines.

    Args:
        df: DataFrame with RMSE data
        dataset_name: Dataset name
        time: Time step
        output_dir: Directory to save figures
    """
    logger.info("Creating scatter plots with regression lines...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'vx': 'red', 'vy': 'green', 'vz': 'blue'}

    for idx, component in enumerate(['vx', 'vy', 'vz']):
        ax = axes[idx]
        rmse_col = f'rmse_{component}'

        x = df['sample_count'].values
        y = df[rmse_col].values

        # Scatter plot
        ax.scatter(x, y, alpha=0.3, s=10, color=colors[component])

        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = poly(x_line)

        ax.plot(x_line, y_line, 'k--', linewidth=2, label=f'Linear fit: y={coeffs[0]:.2e}x+{coeffs[1]:.2e}')

        ax.set_xlabel('Sample Count', fontsize=12)
        ax.set_ylabel(f'RMSE {component.upper()}', fontsize=12)
        ax.set_title(f'{component.upper()} RMSE vs Sample Count', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle(f'RMSE vs Sample Count - {dataset_name} t={time}', fontsize=16)
    plt.tight_layout()

    output_file = output_path / f"rmse_vs_sample_count_{dataset_name}_{time:04d}.png"
    logger.info(f"Saving scatter plot to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_binned_analysis(binned_results, bin_labels, dataset_name, time, output_dir):
    """
    Create bar plots showing RMSE statistics by sample count bins.

    Args:
        binned_results: Dictionary with binned statistics
        bin_labels: Labels for bins
        dataset_name: Dataset name
        time: Time step
        output_dir: Directory to save figures
    """
    logger.info("Creating binned analysis plots...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'vx': 'red', 'vy': 'green', 'vz': 'blue'}

    for idx, component in enumerate(['vx', 'vy', 'vz']):
        ax = axes[idx]
        stats = binned_results[component]

        x_pos = np.arange(len(bin_labels))

        # Plot mean with error bars (std)
        ax.bar(x_pos, stats['mean'], yerr=stats['std'],
               color=colors[component], alpha=0.6, capsize=5,
               label='Mean ± Std')

        # Add median line
        ax.plot(x_pos, stats['median'], 'ko-', linewidth=2, markersize=8, label='Median')

        ax.set_xlabel('Sample Count Bin', fontsize=12)
        ax.set_ylabel(f'RMSE {component.upper()}', fontsize=12)
        ax.set_title(f'{component.upper()} RMSE by Sample Count Bin', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)

    plt.suptitle(f'RMSE Distribution by Sample Count Bins - {dataset_name} t={time}', fontsize=16)
    plt.tight_layout()

    output_file = output_path / f"rmse_binned_analysis_{dataset_name}_{time:04d}.png"
    logger.info(f"Saving binned analysis plot to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def save_analysis_report(correlation_results, binned_results, bin_labels, dataset_name, time, output_dir):
    """
    Save a comprehensive analysis report to a text file.

    Args:
        correlation_results: Dictionary with correlation metrics
        binned_results: Dictionary with binned statistics
        bin_labels: Labels for bins
        dataset_name: Dataset name
        time: Time step
        output_dir: Directory to save report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"mi_correlation_report_{dataset_name}_{time:04d}.txt"

    logger.info(f"Saving analysis report to {report_file}")

    with open(report_file, 'w') as f:
        f.write(f"RMSE vs Sample Count Analysis - {dataset_name} t={time}\n")
        f.write("=" * 80 + "\n\n")

        f.write("OBJECTIVE:\n")
        f.write("Determine if higher RMSE correlates with sample size (number of times a\n")
        f.write("position appears in multiple cubes). This could indicate encoding bias or\n")
        f.write("systematic variation in how the autoencoder handles frequently-seen positions.\n\n")

        f.write("METRICS:\n")
        f.write("- Pearson r: Measures linear correlation (-1 to +1)\n")
        f.write("- Spearman r: Measures monotonic correlation (-1 to +1)\n")
        f.write("- Mutual Information: Measures non-linear dependency (0 = independent, higher = more dependent)\n")
        f.write("- p-value: Statistical significance (p < 0.05 typically considered significant)\n\n")

        f.write("=" * 80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for component in ['vx', 'vy', 'vz']:
            results = correlation_results[component]
            f.write(f"\n{component.upper()} Component:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  RMSE Statistics:\n")
            f.write(f"    Mean:   {results['rmse_mean']:.8f}\n")
            f.write(f"    Median: {results['rmse_median']:.8f}\n")
            f.write(f"    Std:    {results['rmse_std']:.8f}\n\n")

            f.write(f"  Sample Count Statistics:\n")
            f.write(f"    Mean: {results['sample_count_mean']:.2f}\n")
            f.write(f"    Std:  {results['sample_count_std']:.2f}\n\n")

            f.write(f"  Correlation Metrics:\n")
            f.write(f"    Pearson r:        {results['pearson_r']:+.6f}  (p={results['pearson_p']:.4e})\n")
            f.write(f"    Spearman r:       {results['spearman_r']:+.6f}  (p={results['spearman_p']:.4e})\n")
            f.write(f"    Mutual Info:      {results['mutual_info']:.6f}\n\n")

            # Interpretation
            f.write(f"  Interpretation:\n")
            if abs(results['pearson_r']) < 0.1:
                f.write(f"    - Very weak linear correlation\n")
            elif abs(results['pearson_r']) < 0.3:
                f.write(f"    - Weak linear correlation\n")
            elif abs(results['pearson_r']) < 0.5:
                f.write(f"    - Moderate linear correlation\n")
            else:
                f.write(f"    - Strong linear correlation\n")

            if results['pearson_p'] < 0.05:
                f.write(f"    - Statistically significant (p < 0.05)\n")
            else:
                f.write(f"    - Not statistically significant (p >= 0.05)\n")

            if results['pearson_r'] > 0:
                f.write(f"    - Positive correlation: RMSE tends to INCREASE with sample count\n")
            else:
                f.write(f"    - Negative correlation: RMSE tends to DECREASE with sample count\n")

            f.write(f"    - MI score of {results['mutual_info']:.6f} indicates ")
            if results['mutual_info'] < 0.01:
                f.write(f"very weak dependency\n")
            elif results['mutual_info'] < 0.05:
                f.write(f"weak dependency\n")
            elif results['mutual_info'] < 0.1:
                f.write(f"moderate dependency\n")
            else:
                f.write(f"strong dependency\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("BINNED ANALYSIS (Sample Count Quartiles)\n")
        f.write("=" * 80 + "\n\n")

        for component in ['vx', 'vy', 'vz']:
            stats = binned_results[component]
            f.write(f"\n{component.upper()} Component:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Bin':<15} {'Count':>8} {'Mean':>12} {'Median':>12} {'Std':>12} {'Q25':>12} {'Q75':>12}\n")
            f.write("-" * 80 + "\n")

            for bin_label in bin_labels:
                row = stats.loc[bin_label]
                f.write(f"{bin_label:<15} {int(row['count']):>8} "
                       f"{row['mean']:>12.8f} {row['median']:>12.8f} {row['std']:>12.8f} "
                       f"{row['q25']:>12.8f} {row['q75']:>12.8f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        f.write("Based on the correlation analysis and binned statistics:\n\n")

        for component in ['vx', 'vy', 'vz']:
            results = correlation_results[component]
            f.write(f"{component.upper()}: ")

            if results['pearson_p'] < 0.05 and abs(results['pearson_r']) > 0.1:
                direction = "increases" if results['pearson_r'] > 0 else "decreases"
                f.write(f"RMSE {direction} with sample count (r={results['pearson_r']:.3f}, p={results['pearson_p']:.4e}).\n")
            else:
                f.write(f"No significant correlation between RMSE and sample count.\n")

        f.write("\n")

    logger.info("Analysis report saved successfully")


def main():
    """Main processing function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze correlation between RMSE and sample count using mutual information')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., "7p2")')
    parser.add_argument('--time', type=int, required=True, help='Time step (e.g., 1000)')
    args = parser.parse_args()

    # Load RMSE data
    df = load_rmse_data(args.dataset, args.time, ANALYSIS_DIR)

    if df is None:
        logger.error("Failed to load RMSE data")
        return 1

    # Compute correlations
    correlation_results = compute_correlations(df)

    # Binned analysis
    binned_results, bin_labels = binned_analysis(df)

    # Create visualizations
    plot_scatter_with_regression(df, args.dataset, args.time, ANALYSIS_DIR)
    plot_binned_analysis(binned_results, bin_labels, args.dataset, args.time, ANALYSIS_DIR)

    # Save comprehensive report
    save_analysis_report(correlation_results, binned_results, bin_labels,
                        args.dataset, args.time, ANALYSIS_DIR)

    logger.info("Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
