#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation: Latent Space Ablation Study
=======================================

Compares the reconstruction performance (RMSE) across different latent 
dimensions for PCA vs. the fixed 47-dimension AttentionSE Autoencoder.
This demonstrates the "compression efficiency" - i.e., at what point does 
a linear model (PCA) need more dimensions to match your AE?
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob

# Resolve project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index

try:
    from .config import add_config_argument, choose_path, config_get, load_config
except ImportError:
    from config import add_config_argument, choose_path, config_get, load_config

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def main():
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--n_files', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None, help='Directory for ablation plot')
    args = parser.parse_args()

    config = load_config(args.config)
    args.data_dir = choose_path(args.data_dir, config, "data.validation_data_dir", os.path.join("data", "Final_Cubed_OG_Data_wLatent"))
    args.model_path = choose_path(
        args.model_path,
        config,
        "paths.production_model_path",
        os.path.join(PROJECT_ROOT, "encoder", "autoencoderGEN3", "saved_models_production", "Model_GEN3_05_AttentionSE_absolute_best.pt"),
    )
    args.n_files = args.n_files if args.n_files is not None else int(config_get(config, "validation.n_files", 5))
    output_dir = choose_path(args.output_dir, config, "paths.results_dir", "Documentation")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load data
    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.pkl*"), recursive=True))[:args.n_files]
    all_data = []
    for f in files:
        df = pd.read_pickle(f)
        v_cols = [c for c in df.columns if c.startswith('velocity_')]
        all_data.append(df[v_cols].values.astype(np.float32))
    X = np.vstack(all_data)

    # AE Reference (Fixed at 47)
    model = get_model_by_index(4).to(device)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        X_ae = model(torch.from_numpy(X).to(device))[0].cpu().numpy()
    rmse_ae = compute_rmse(X, X_ae)

    # PCA Ablation
    latent_dims = [1, 2, 4, 8, 16, 32, 47, 64, 128, 256]
    pca_rmses = []
    
    print("Running PCA Ablation...")
    for dim in latent_dims:
        pca = PCA(n_components=dim)
        X_pca = pca.inverse_transform(pca.fit_transform(X))
        pca_rmses.append(compute_rmse(X, X_pca))
        print(f"Dim {dim}: RMSE={pca_rmses[-1]:.6f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, pca_rmses, 'o-', label='PCA (Linear)', color='salmon', linewidth=2)
    plt.axhline(y=rmse_ae, color='skyblue', linestyle='--', linewidth=2, label=f'AttentionSE (AE) @ Dim 47')
    plt.axvline(x=47, color='gray', linestyle=':', alpha=0.5)
    
    # Find the "break-even" point
    # At what PCA dim does it beat the AE?
    better_than_ae = [d for d, r in zip(latent_dims, pca_rmses) if r < rmse_ae]
    if better_than_ae:
        break_even = better_than_ae[0]
        plt.annotate(f'PCA matches AE at Dim ≈ {break_even}', 
                     xy=(break_even, rmse_ae), xytext=(break_even+20, rmse_ae+0.001),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.xscale('log')
    plt.xticks(latent_dims, [str(d) for d in latent_dims])
    plt.xlabel('Latent Dimensions (Log Scale)')
    plt.ylabel('Reconstruction RMSE (Normalized)')
    plt.title('Ablation Study: Latent Dimension vs. Reconstruction Error')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    out_path = os.path.join(output_dir, 'ablation_study.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved ablation plot to {out_path}")

    # Summary table in terminal
    print("\n--- Ablation Summary ---")
    print(f"{'Method':<20} | {'Latent Dim':<10} | {'RMSE':<10}")
    print("-" * 45)
    print(f"{'AttentionSE (AE)':<20} | {'47':<10} | {rmse_ae:>10.6f}")
    for dim, rmse in zip(latent_dims, pca_rmses):
        print(f"{'PCA':<20} | {dim:<10} | {rmse:>10.6f}")

if __name__ == "__main__":
    main()
