#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Model GEN3 05 (AttentionSE) on a random sample of data files
====================================================================

This script:
- Randomly selects N pickle files from a given directory (default: 10)
- Loads Model GEN3 05 (AttentionSE) with production weights
- Decodes latent vectors back to velocity space
- Computes rigorous error metrics per file (MAE/RMSE mean/median/std/p95, global RMSE)
- Saves a per-file summary CSV
- Additionally, for the FIRST processed file, saves a detailed CSV with
  validation columns and error columns similar to single-file validators
  (`average_error`, `rmse_error`).

Usage example:
python -m encoder.autoencoderGEN3.validate_model_05_production \
  --data_dir /Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent \
  --n_files 10 \
  --seed 42

"""

import os
import sys
import argparse
import glob
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Resolve project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index
from TransformLatent import FloatConverter

# === Accelerator detection & colorful report (consistent with project style) ===
CSI = "\033["
RESET = f"{CSI}0m"
COLORS = [31, 33, 32, 36, 34, 35]  # R, Y, G, C, B, M

def rainbow(msg: str) -> str:
    out = []
    k = 0
    for ch in msg:
        if ch.strip():
            out.append(f"{CSI}{COLORS[k % len(COLORS)]}m{ch}{RESET}")
            k += 1
        else:
            out.append(ch)
    return ''.join(out)

def accelerator_report():
    has_cuda = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available()

    if has_cuda:
        device = torch.device('cuda')
    elif mps_avail:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    lines = [
        f"Selected device: {device}",
        f"CUDA available: {has_cuda}",
        f"MPS available: {mps_avail}",
    ]

    print("\n" + "="*50)
    for ln in lines:
        print(rainbow(ln))
    print("="*50 + "\n")

    return device

# --- Helpers ---

def list_candidate_files(data_dir: str) -> List[str]:
    """Return .pkl and .pkl.gz files inside data_dir (recursive)."""
    patt1 = os.path.join(data_dir, "**", "*.pkl")
    patt2 = os.path.join(data_dir, "**", "*.pkl.gz")
    files = sorted(glob.glob(patt1, recursive=True) + glob.glob(patt2, recursive=True))
    return files


def pick_random_files(files: List[str], n: int, seed: int) -> List[str]:
    if not files:
        return []
    rng = np.random.default_rng(seed)
    if n >= len(files):
        return files
    idx = rng.choice(len(files), size=n, replace=False)
    return [files[i] for i in idx]


def load_dataframe(path: str) -> pd.DataFrame:
    """Load pickle DataFrame with pandas (compression auto-inferred)."""
    return pd.read_pickle(path, compression='infer')


def find_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Determine latent columns (latent_1..latent_47) and velocity columns (375 dims).
    Primary strategy: columns starting with 'velocity_'.
    Fallback: exclude known metadata + latent columns (as in Ordered_250_ValidateOneFile).
    """
    latent_cols = [f"latent_{i}" for i in range(1, 48)]
    if not all(col in df.columns for col in latent_cols):
        raise ValueError("Missing some latent columns in the input file.")

    # Preferred approach
    velocity_cols = [c for c in df.columns if c.startswith('velocity_')]

    if len(velocity_cols) != 375:
        # Fallback
        metadata_cols = [
            'x', 'y', 'z', 'vx', 'vy', 'vz', 'time', 'distance',
            'vx_original', 'vy_original', 'vz_original',
            'centroid_x', 'centroid_y', 'centroid_z',
            'original_vx', 'original_vy', 'original_vz'
        ]
        velocity_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('latent_')]
        if len(velocity_cols) != 375:
            raise ValueError(f"Expected 375 velocity columns, found {len(velocity_cols)}")

    return latent_cols, velocity_cols


def compute_file_metrics(original_np: np.ndarray, decoded_np: np.ndarray) -> dict:
    """Compute rigorous error metrics for a single file in both Normalized and Raw scales."""
    converter = FloatConverter()
    
    # 1. Normalized metrics
    diff_norm = decoded_np - original_np
    abs_diff_norm = np.abs(diff_norm)
    mae_rows_norm = abs_diff_norm.mean(axis=1)
    rmse_rows_norm = np.sqrt((diff_norm ** 2).mean(axis=1))
    global_rmse_norm = math.sqrt(np.mean(diff_norm ** 2))

    # 2. Raw metrics (Un-normalized)
    original_raw = converter.unconvert(original_np)
    decoded_raw = converter.unconvert(decoded_np)
    diff_raw = decoded_raw - original_raw
    abs_diff_raw = np.abs(diff_raw)
    mae_rows_raw = abs_diff_raw.mean(axis=1)
    rmse_rows_raw = np.sqrt((diff_raw ** 2).mean(axis=1))
    global_rmse_raw = math.sqrt(np.mean(diff_raw ** 2))

    def pct(arr, q):
        return float(np.percentile(arr, q))

    metrics = {
        'rows': int(original_np.shape[0]),
        'cols': int(original_np.shape[1]),
        # Norm
        'mae_mean_norm': float(mae_rows_norm.mean()),
        'mae_p95_norm': pct(mae_rows_norm, 95),
        'rmse_mean_norm': float(rmse_rows_norm.mean()),
        'rmse_global_norm': float(global_rmse_norm),
        # Raw
        'mae_mean_raw': float(mae_rows_raw.mean()),
        'mae_median_raw': float(np.median(mae_rows_raw)),
        'mae_std_raw': float(mae_rows_raw.std(ddof=0)),
        'mae_p95_raw': pct(mae_rows_raw, 95),
        'rmse_mean_raw': float(rmse_rows_raw.mean()),
        'rmse_median_raw': float(np.median(rmse_rows_raw)),
        'rmse_std_raw': float(rmse_rows_raw.std(ddof=0)),
        'rmse_p95_raw': pct(rmse_rows_raw, 95),
        'rmse_global_raw': float(global_rmse_raw),
        'max_abs_err_raw': float(abs_diff_raw.max()),
    }
    return metrics


def save_detailed_csv(df: pd.DataFrame, decoded_np: np.ndarray, velocity_cols: List[str], out_path: str):
    """Save a detailed CSV for one file with validation and error columns (Norm & Raw)."""
    converter = FloatConverter()
    validation_cols = [f"validation_{i}" for i in range(1, decoded_np.shape[1] + 1)]
    validation_df = pd.DataFrame(decoded_np, columns=validation_cols, index=df.index)

    original_np = df[velocity_cols].values.astype(np.float32)
    
    # Normalized errors
    diff_norm = decoded_np - original_np
    
    # Raw values and errors
    original_raw = converter.unconvert(original_np)
    decoded_raw = converter.unconvert(decoded_np)
    diff_raw = decoded_raw - original_raw

    df_out = df.copy()
    # Normalized
    df_out['average_error_norm'] = np.abs(diff_norm).mean(axis=1)
    df_out['rmse_error_norm'] = np.sqrt((diff_norm ** 2).mean(axis=1))
    
    # Raw
    df_out['average_error_raw'] = np.abs(diff_raw).mean(axis=1)
    df_out['rmse_error_raw'] = np.sqrt((diff_raw ** 2).mean(axis=1))

    df_out = pd.concat([df_out, validation_df], axis=1)
    df_out.to_csv(out_path, index=False)


def save_summary_figure(summary_df: pd.DataFrame, out_path: str):
    """Save a figure comparing Normalized vs Raw RMSE across processed files."""
    if summary_df.empty:
        return

    # Set up the figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    indices = np.arange(len(summary_df))
    width = 0.35

    # Plot 1: RMSE Comparison (Normalized)
    axes[0].bar(indices - width/2, summary_df['rmse_mean_norm'], width, label='RMSE Mean (Norm)', color='skyblue')
    axes[0].bar(indices + width/2, summary_df['rmse_global_norm'], width, label='RMSE Global (Norm)', color='steelblue')
    axes[0].set_title('Normalized RMSE per File')
    axes[0].set_ylabel('Error (Normalized 0-1)')
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels(summary_df['file'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: RMSE Comparison (Raw)
    axes[1].bar(indices - width/2, summary_df['rmse_mean_raw'], width, label='RMSE Mean (Raw)', color='salmon')
    axes[1].bar(indices + width/2, summary_df['rmse_global_raw'], width, label='RMSE Global (Raw)', color='darkred')
    axes[1].set_title('Raw (Un-normalized) RMSE per File')
    axes[1].set_ylabel('Error (Raw Units)')
    axes[1].set_xticks(indices)
    axes[1].set_xticklabels(summary_df['file'], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(rainbow(f"Saved summary figure to {out_path}"))


def main():
    parser = argparse.ArgumentParser(description="Validate Model GEN3 05 on random sample of files")
    parser.add_argument('--data_dir', type=str, default=os.path.join(
        '/Users', 'kkreth', 'PycharmProjects', 'data', 'Final_Cubed_OG_Data_wLatent'
    ))
    parser.add_argument('--n_files', type=int, default=100, help='Number of files to sample randomly')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for file sampling')
    parser.add_argument('--model_path', type=str, default=os.path.join(
        PROJECT_ROOT, 'encoder', 'autoencoderGEN3', 'saved_models_production',
        'Model_GEN3_05_AttentionSE_absolute_best.pt'
    ))
    parser.add_argument('--summary_csv', type=str, default='validate_model_05_production_summary.csv')
    args = parser.parse_args()

    device = accelerator_report()

    if not os.path.isdir(args.data_dir):
        print(rainbow(f"Error: data_dir not found: {args.data_dir}"))
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(rainbow(f"Error: model weights not found at {args.model_path}"))
        sys.exit(1)

    # Collect and pick files
    all_files = list_candidate_files(args.data_dir)
    if not all_files:
        print(rainbow("Error: No candidate .pkl/.pkl.gz files found in data_dir"))
        sys.exit(1)

    picked = pick_random_files(all_files, args.n_files, args.seed)
    print(rainbow(f"Selected {len(picked)} files (seed={args.seed}) for validation"))

    # Load model
    print(rainbow("Loading model architecture (Model GEN3 05)..."))
    model = get_model_by_index(4)  # Model_GEN3_05_AttentionSE

    print(rainbow(f"Loading weights from {args.model_path}..."))
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(rainbow(f"Failed to load model weights: {e}"))
        sys.exit(1)

    model.to(device)
    model.eval()

    per_file_records = []

    for idx, path in enumerate(picked):
        base = os.path.basename(path)
        print(rainbow(f"Processing [{idx+1}/{len(picked)}]: {base}"))
        try:
            df = load_dataframe(path)
            latent_cols, velocity_cols = find_columns(df)

            latent_data = df[latent_cols].values.astype(np.float32)
            latent_tensor = torch.from_numpy(latent_data).to(device)

            with torch.no_grad():
                decoded = model.decode(latent_tensor)
                decoded_np = decoded.detach().cpu().numpy()

            original_np = df[velocity_cols].values.astype(np.float32)

            metrics = compute_file_metrics(original_np, decoded_np)
            metrics.update({'file': base, 'path': path})
            per_file_records.append(metrics)

            # For the first file, save a detailed CSV
            if idx == 0:
                out_csv = f"validation_detailed_{os.path.splitext(base)[0]}.csv"
                print(rainbow(f"Saving detailed validation CSV for first file to {out_csv}"))
                save_detailed_csv(df, decoded_np, velocity_cols, out_csv)

        except Exception as e:
            print(rainbow(f"Warning: Skipping file due to error: {base} -> {e}"))
            continue

    if not per_file_records:
        print(rainbow("Error: No files processed successfully."))
        sys.exit(1)

    summary_df = pd.DataFrame(per_file_records)
    # Order columns nicely
    cols_order = [
        'file', 'rows', 'cols',
        'mae_mean_norm', 'mae_p95_norm', 'rmse_mean_norm', 'rmse_global_norm',
        'mae_mean_raw', 'mae_median_raw', 'mae_std_raw', 'mae_p95_raw',
        'rmse_mean_raw', 'rmse_median_raw', 'rmse_std_raw', 'rmse_p95_raw', 'rmse_global_raw',
        'max_abs_err_raw', 'path'
    ]
    existing_cols = [c for c in cols_order if c in summary_df.columns]
    summary_df = summary_df[existing_cols]

    summary_df.to_csv(args.summary_csv, index=False)

    # Save summary figure
    fig_path = args.summary_csv.replace('.csv', '.png')
    save_summary_figure(summary_df, fig_path)

    # Print an aggregated overview
    print(rainbow(f"Saved per-file summary to {args.summary_csv}"))

    def aggr(desc: str, series: pd.Series):
        print(rainbow(f"{desc}: mean={series.mean():.6f} | median={series.median():.6f} | p95={np.percentile(series,95):.6f}"))

    print("\n" + "-"*50)
    print(rainbow("Aggregate across processed files (NORMALIZED SCALE):"))
    aggr("MAE (Norm)", summary_df['mae_mean_norm'])
    aggr("RMSE (Norm)", summary_df['rmse_mean_norm'])
    print(rainbow(f"Global RMSE Norm (avg of files): {summary_df['rmse_global_norm'].mean():.6f}"))
    
    print("\n" + rainbow("Aggregate across processed files (RAW SCALE):"))
    aggr("MAE (Raw)", summary_df['mae_mean_raw'])
    aggr("RMSE (Raw)", summary_df['rmse_mean_raw'])
    print(rainbow(f"Global RMSE Raw (avg of files): {summary_df['rmse_global_raw'].mean():.6f}"))
    print("-"*50 + "\n")


if __name__ == "__main__":
    main()
