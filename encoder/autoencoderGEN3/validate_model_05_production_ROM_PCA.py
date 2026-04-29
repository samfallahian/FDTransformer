#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Model GEN3 05 (AttentionSE) vs. Standard PCA (ROM)
=========================================================

This script:
- Randomly selects N pickle files.
- Loads the AttentionSE Autoencoder.
- Performs PCA (Principal Component Analysis) on the same dataset using 47 components.
- Compares reconstruction performance (RMSE/MAE) in both normalized and raw scales.
- Saves a comparative CSV and visualization.

This satisfies the publication requirement of comparing non-linear (AE) vs linear (PCA/POD) 
dimensionality reduction at the same compression ratio.
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List

# Resolve project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index
from helpers.TransformLatent import FloatConverter
try:
    from .config import add_config_argument, choose_path, config_get, load_config
except ImportError:
    from config import add_config_argument, choose_path, config_get, load_config

# === Colors and Helpers ===
CSI = "\033["
RESET = f"{CSI}0m"
COLORS = [31, 33, 32, 36, 34, 35]

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

def list_candidate_files(data_dir: str) -> List[str]:
    patt1 = os.path.join(data_dir, "**", "*.pkl")
    patt2 = os.path.join(data_dir, "**", "*.pkl.gz")
    return sorted(glob.glob(patt1, recursive=True) + glob.glob(patt2, recursive=True))

def find_velocity_columns(df: pd.DataFrame) -> List[str]:
    velocity_cols = [c for c in df.columns if c.startswith('velocity_')]
    if len(velocity_cols) != 375:
        # Simplified fallback for this specific project structure
        metadata_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'time', 'distance', 'vx_original', 'vy_original', 'vz_original']
        velocity_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('latent_')]
        if len(velocity_cols) != 375:
            raise ValueError(f"Expected 375 velocity columns, found {len(velocity_cols)}")
    return velocity_cols

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def main():
    parser = argparse.ArgumentParser(description="Compare AttentionSE AE vs PCA (POD)")
    add_config_argument(parser)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--n_files', type=int, default=None, help='Files to sample for PCA training and validation')
    parser.add_argument('--latent_dim', type=int, default=47, help='Dimensions for both AE and PCA')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for ROM comparison outputs')
    args = parser.parse_args()

    config = load_config(args.config)
    args.data_dir = choose_path(args.data_dir, config, "data.validation_data_dir", os.path.join("data", "Final_Cubed_OG_Data_wLatent"))
    args.model_path = choose_path(
        args.model_path,
        config,
        "paths.production_model_path",
        os.path.join(PROJECT_ROOT, "encoder", "autoencoderGEN3", "saved_models_production", "Model_GEN3_05_AttentionSE_absolute_best.pt"),
    )
    args.n_files = args.n_files if args.n_files is not None else int(config_get(config, "validation.n_files", 20))
    args.seed = args.seed if args.seed is not None else int(config_get(config, "validation.seed", 42))
    output_dir = choose_path(args.output_dir, config, "paths.results_dir", "Documentation")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(rainbow(f"Active Device: {device}"))

    all_files = list_candidate_files(args.data_dir)
    if not all_files:
        print(rainbow(f"Error: No files found in {args.data_dir}"))
        return

    rng = np.random.default_rng(args.seed)
    picked_files = rng.choice(all_files, size=min(len(all_files), args.n_files), replace=False)
    
    # 1. Load AE Model
    print(rainbow(f"Loading AE Model from {args.model_path}..."))
    model = get_model_by_index(4).to(device)
    try:
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(rainbow(f"Failed to load model: {e}"))
        return

    # 2. Accumulate Data for Global PCA Comparison
    print(rainbow(f"Reading {len(picked_files)} files for comparison..."))
    all_data_list = []
    for f in picked_files:
        df = pd.read_pickle(f)
        v_cols = find_velocity_columns(df)
        all_data_list.append(df[v_cols].values.astype(np.float32))
    
    X_all = np.vstack(all_data_list)
    print(rainbow(f"Total samples for evaluation: {X_all.shape[0]}"))

    # 3. PCA Reconstruction (Linear ROM)
    print(rainbow(f"Fitting PCA with {args.latent_dim} components..."))
    
    # Check for NaNs or Inf and clean
    if not np.all(np.isfinite(X_all)):
        print(rainbow("Warning: Non-finite values detected in data. Cleaning..."))
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=1.0, neginf=-1.0)

    # Convert to float64 for PCA stability to avoid overflow warnings
    # And perform a small amount of regularisation if needed, but PCA usually handles it.
    X_all_64 = X_all.astype(np.float64)
    
    # We use randomized solver as it is often more robust to precision issues in large datasets 
    # and handles the float64 data well.
    pca = PCA(n_components=args.latent_dim, svd_solver='randomized', random_state=args.seed)
    X_pca_latent = pca.fit_transform(X_all_64)
    X_pca_recon = pca.inverse_transform(X_pca_latent).astype(np.float32)
    
    var_explained = np.sum(pca.explained_variance_ratio_)
    print(rainbow(f"PCA Total Variance Explained: {var_explained:.4f}"))

    # 4. AE Reconstruction (Non-linear ROM)
    print(rainbow("Running AE Inference..."))
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_all).to(device)
        # We use the full forward pass to get reconstruction
        recon_ae, _ = model(X_tensor)
        X_ae_recon = recon_ae.cpu().numpy()

    # 5. Metrics Calculation
    converter = FloatConverter()
    
    # Normalized
    rmse_pca_norm = compute_rmse(X_all, X_pca_recon)
    rmse_ae_norm = compute_rmse(X_all, X_ae_recon)
    
    # Raw Scale
    X_all_raw = converter.unconvert(X_all)
    X_pca_recon_raw = converter.unconvert(X_pca_recon)
    X_ae_recon_raw = converter.unconvert(X_ae_recon)
    
    rmse_pca_raw = compute_rmse(X_all_raw, X_pca_recon_raw)
    rmse_ae_raw = compute_rmse(X_all_raw, X_ae_recon_raw)

    print("\n" + "="*50)
    print(rainbow(f"RESULTS COMPARISON (Latent Dim = {args.latent_dim})"))
    print("="*50)
    print(f"{'Metric':<20} | {'PCA (Linear)':<15} | {'AttentionSE (AE)':<15}")
    print("-" * 55)
    print(f"{'RMSE (Normalized)':<20} | {rmse_pca_norm:>15.6f} | {rmse_ae_norm:>15.6f}")
    print(f"{'RMSE (Raw Units)':<20} | {rmse_pca_raw:>15.6f} | {rmse_ae_raw:>15.6f}")
    
    improvement = (rmse_pca_raw - rmse_ae_raw) / rmse_pca_raw * 100
    print("-" * 55)
    print(rainbow(f"AE Improvement over PCA: {improvement:.2f}%"))
    print("="*50 + "\n")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    methods = ['PCA (Linear)', 'AttentionSE (AE)']
    rmses = [rmse_pca_raw, rmse_ae_raw]
    bars = plt.bar(methods, rmses, color=['salmon', 'skyblue'])
    plt.ylabel('Global RMSE (Raw Units)')
    plt.title(f'ROM Comparison: Linear (PCA) vs. Non-Linear (AE)\nLatent Dimensions = {args.latent_dim}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', va='bottom', ha='center', fontweight='bold')

    out_plot = os.path.join(output_dir, "ROM_Comparison_AE_vs_PCA.png")
    plt.savefig(out_plot, dpi=300)
    print(rainbow(f"Comparison plot saved to {out_plot}"))

    # Save results to CSV
    results_df = pd.DataFrame([{
        'latent_dim': args.latent_dim,
        'pca_variance_explained': var_explained,
        'rmse_pca_norm': rmse_pca_norm,
        'rmse_ae_norm': rmse_ae_norm,
        'rmse_pca_raw': rmse_pca_raw,
        'rmse_ae_raw': rmse_ae_raw,
        'improvement_pct': improvement
    }])
    out_csv = os.path.join(output_dir, "ROM_Comparison_AE_vs_PCA.csv")
    results_df.to_csv(out_csv, index=False)
    print(rainbow(f"Detailed metrics saved to {out_csv}"))

if __name__ == "__main__":
    main()
