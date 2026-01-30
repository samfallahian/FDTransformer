#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot training curves and comparison charts from experiment results
"""

import json
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


def load_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_training_curves(results_data, output_dir):
    """Plot training and validation RMSE curves for all models"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib required for plotting")
        return

    results = results_data['results']
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        print("No successful results to plot")
        return

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All validation curves
    for r in successful:
        epochs = range(1, len(r['val_rmse_history']) + 1)
        ax1.plot(epochs, r['val_rmse_history'], label=r['name'].replace('Model_', ''), alpha=0.7)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation RMSE')
    ax1.set_title('Validation RMSE vs Epoch (All Models)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Top 5 models (train + val)
    sorted_results = sorted(successful, key=lambda x: x['final_val_rmse'])[:5]

    for r in sorted_results:
        epochs = range(1, len(r['val_rmse_history']) + 1)
        label = r['name'].replace('Model_', '')
        ax2.plot(epochs, r['train_rmse_history'], '--', alpha=0.5, label=f"{label} (train)")
        ax2.plot(epochs, r['val_rmse_history'], '-', alpha=0.7, label=f"{label} (val)")

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Top 5 Models: Train vs Validation RMSE')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    plt.close()


def plot_comparison_bar(results_data, output_dir):
    """Create bar chart comparing final validation RMSE"""
    if not HAS_MATPLOTLIB:
        return

    results = results_data['results']
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        return

    # Sort by validation RMSE
    successful.sort(key=lambda x: x['final_val_rmse'])

    names = [r['name'].replace('Model_', '').replace('_', ' ') for r in successful]
    val_rmse = [r['final_val_rmse'] for r in successful]
    train_rmse = [r['final_train_rmse'] for r in successful]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_rmse, width, label='Training RMSE', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_rmse, width, label='Validation RMSE', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.set_title('Final RMSE Comparison (All Models)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison bar chart to: {output_path}")
    plt.close()


def plot_training_time(results_data, output_dir):
    """Plot training time vs RMSE"""
    if not HAS_MATPLOTLIB:
        return

    results = results_data['results']
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        return

    names = [r['name'].replace('Model_', '').replace('_', '\n') for r in successful]
    val_rmse = [r['final_val_rmse'] for r in successful]
    times = [r['training_time_seconds'] / 60 for r in successful]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(times, val_rmse, s=100, alpha=0.6, c=val_rmse, cmap='RdYlGn_r')

    for i, name in enumerate(names):
        ax.annotate(name, (times[i], val_rmse[i]), fontsize=7, alpha=0.7)

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel('Final Validation RMSE')
    ax.set_title('Training Time vs Performance')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, label='Validation RMSE')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'time_vs_rmse.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved time vs RMSE plot to: {output_path}")
    plt.close()


def print_summary_table(results_data):
    """Print a formatted summary table"""
    results = results_data['results']
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)

    config = results_data['configuration']
    print(f"Configuration: {config['num_epochs']} epochs, batch size {config['batch_size']}, "
          f"lr {config['learning_rate']}")
    print(f"Total time: {results_data['overall_time_seconds']/60:.2f} minutes\n")

    if successful:
        successful.sort(key=lambda x: x['final_val_rmse'])

        print(f"Successful Models ({len(successful)}):")
        print("-"*100)
        print(f"{'Rank':<6} {'Model':<30} {'Val RMSE':<12} {'Train RMSE':<12} {'Time (min)':<12} {'Epochs':<8}")
        print("-"*100)

        for rank, r in enumerate(successful, 1):
            print(f"{rank:<6} {r['name']:<30} {r['final_val_rmse']:<12.6f} "
                  f"{r['final_train_rmse']:<12.6f} {r['training_time_seconds']/60:<12.2f} "
                  f"{r['num_epochs']:<8}")

        print("-"*100)
        print(f"\n🏆 BEST MODEL: {successful[0]['name']}")
        print(f"   Validation RMSE: {successful[0]['final_val_rmse']:.6f}")
        print(f"   Training RMSE: {successful[0]['final_train_rmse']:.6f}")
        print(f"   Description: {successful[0]['description']}")

        # Calculate improvement percentages
        if len(successful) > 1:
            worst_rmse = successful[-1]['final_val_rmse']
            best_rmse = successful[0]['final_val_rmse']
            improvement = ((worst_rmse - best_rmse) / worst_rmse) * 100
            print(f"\n   Improvement over worst: {improvement:.2f}%")

    if failed:
        print(f"\n\nFailed Models ({len(failed)}):")
        print("-"*100)
        for r in failed:
            print(f"✗ {r['name']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Plot and analyze experiment results")
    parser.add_argument('json_file', type=str, nargs='?', help='Path to results JSON file')
    parser.add_argument('--latest', action='store_true', help='Use latest results file')
    parser.add_argument('--no-plots', action='store_true', help='Only print summary, no plots')
    args = parser.parse_args()

    # Find results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Determine which file to use
    if args.latest or args.json_file is None:
        json_files = sorted(results_dir.glob('experiment_results_*.json'))
        if not json_files:
            print(f"No results files found in {results_dir}")
            return
        json_path = json_files[-1]
        print(f"Using latest results: {json_path.name}")
    else:
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return

    # Load results
    results_data = load_results(json_path)

    # Print summary
    print_summary_table(results_data)

    # Create plots
    if not args.no_plots and HAS_MATPLOTLIB:
        output_dir = json_path.parent
        print(f"\nGenerating plots in {output_dir}...")
        plot_training_curves(results_data, output_dir)
        plot_comparison_bar(results_data, output_dir)
        plot_training_time(results_data, output_dir)
        print("\n✓ All plots generated successfully")
    elif args.no_plots:
        print("\nSkipping plots (--no-plots specified)")
    else:
        print("\nSkipping plots (matplotlib not available)")


if __name__ == "__main__":
    main()
