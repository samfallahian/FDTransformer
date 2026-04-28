#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation: Divergence-Free Condition
=====================================

Calculates the divergence of reconstructed velocity cubes (5x5x5) for both
the AttentionSE Autoencoder and the PCA baseline. 

Divergence is calculated using central finite differences on the 5x5x5 grid.
For incompressible flow, divergence should be zero.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Resolve project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from encoder.autoencoderGEN3.models import get_model_by_index
from TransformLatent import FloatConverter

def calculate_divergence_3d(cube_flat):
    """
    Calculates divergence for a 5x5x5 velocity cube.
    cube_flat: shape (375,) -> (125 points * 3 components)
    The 125 points are ordered in a 5x5x5 grid.
    We assume the grid spacing is unit (1.0) for comparison purposes.
    """
    # Reshape to (5, 5, 5, 3) where last dim is [vx, vy, vz]
    cube = cube_flat.reshape(5, 5, 5, 3)
    
    # Components
    u = cube[:, :, :, 0]
    v = cube[:, :, :, 1]
    w = cube[:, :, :, 2]
    
    # Central differences for internal points (3x3x3 interior)
    # du/dx + dv/dy + dw/dz
    dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / 2.0
    dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / 2.0
    dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / 2.0
    
    div = dudx + dvdy + dwdz
    return div # shape (3, 3, 3)

def batch_divergence(batch_flat):
    """batch_flat: (N, 375)"""
    divs = []
    for i in range(batch_flat.shape[0]):
        div = calculate_divergence_3d(batch_flat[i])
        divs.append(np.abs(div).mean())
    return np.array(divs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent')
    parser.add_argument('--n_files', type=int, default=5)
    parser.add_argument('--model_path', type=str, default=os.path.join(
        PROJECT_ROOT, 'encoder', 'autoencoderGEN3', 'saved_models_production', 'Model_GEN3_05_AttentionSE_absolute_best.pt'
    ))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load data
    files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*.pkl*"), recursive=True))[:args.n_files]
    all_data = []
    for f in files:
        df = pd.read_pickle(f)
        v_cols = [c for c in df.columns if c.startswith('velocity_')]
        all_data.append(df[v_cols].values.astype(np.float32))
    X = np.vstack(all_data)

    # Load AE
    model = get_model_by_index(4).to(device)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # AE Reconstruction
    with torch.no_grad():
        X_ae = model(torch.from_numpy(X).to(device))[0].cpu().numpy()

    # PCA Reconstruction
    pca = PCA(n_components=47)
    X_pca = pca.inverse_transform(pca.fit_transform(X))

    # Calculate Divergence
    div_orig = batch_divergence(X)
    div_ae = batch_divergence(X_ae)
    div_pca = batch_divergence(X_pca)

    # Plotting
    plt.figure(figsize=(10, 6))
    data_to_plot = [div_orig, div_ae, div_pca]
    labels = ['Original', 'AttentionSE (AE)', 'PCA (Baseline)']
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.ylabel('Mean Absolute Divergence (|∇·u|)')
    plt.title('Physical Consistency: Divergence Comparison\n(Lower is better for Incompressible Flow)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    out_path = os.path.join(PROJECT_ROOT, 'Documentation', 'divergence_comparison.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved divergence plot to {out_path}")

    # Print stats
    print(f"Original Mean Divergence: {div_orig.mean():.6f}")
    print(f"AE Mean Divergence: {div_ae.mean():.6f}")
    print(f"PCA Mean Divergence: {div_pca.mean():.6f}")
    
    improvement = (div_pca.mean() - div_ae.mean()) / div_pca.mean() * 100
    print(f"AE Divergence Improvement over PCA: {improvement:.2f}%")

if __name__ == "__main__":
    import glob
    main()
