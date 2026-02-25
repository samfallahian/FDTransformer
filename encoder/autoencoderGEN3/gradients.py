#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation: Velocity Gradients and Vorticity
============================================

Evaluates the reconstruction of velocity gradients and vorticity magnitude.
Standard autoencoders often act as low-pass filters, blurring high-frequency
features. This script checks if the AttentionSE model preserves these
features better than PCA.
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
from TransformLatent import FloatConverter

def calculate_vorticity_mag(cube_flat):
    """
    Calculates vorticity magnitude for a 5x5x5 velocity cube.
    ω = ∇ × u
    |ω| = sqrt(ωx² + ωy² + ωz²)
    """
    cube = cube_flat.reshape(5, 5, 5, 3)
    u = cube[:, :, :, 0]
    v = cube[:, :, :, 1]
    w = cube[:, :, :, 2]
    
    # Finite differences for vorticity components
    # ωx = ∂w/∂y - ∂v/∂z
    # ωy = ∂u/∂z - ∂w/∂x
    # ωz = ∂v/∂x - ∂u/∂y
    dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / 2.0
    dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / 2.0
    wx = dwdy - dvdz
    
    dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / 2.0
    dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / 2.0
    wy = dudz - dwdx
    
    dvdx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / 2.0
    dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / 2.0
    wz = dvdx - dudy
    
    mag = np.sqrt(wx**2 + wy**2 + wz**2)
    return mag.mean()

def batch_vorticity(batch_flat):
    mags = []
    for i in range(batch_flat.shape[0]):
        mags.append(calculate_vorticity_mag(batch_flat[i]))
    return np.array(mags)

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

    # Inference
    with torch.no_grad():
        X_ae = model(torch.from_numpy(X).to(device))[0].cpu().numpy()
    
    pca = PCA(n_components=47)
    X_pca = pca.inverse_transform(pca.fit_transform(X))

    # Calculate vorticity magnitude
    vort_orig = batch_vorticity(X)
    vort_ae = batch_vorticity(X_ae)
    vort_pca = batch_vorticity(X_pca)

    # Plotting: Scatter plot of Vorticity Magnitude
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(vort_orig, vort_ae, alpha=0.3, s=10, label='AE', color='skyblue')
    plt.plot([0, vort_orig.max()], [0, vort_orig.max()], 'r--', label='Perfect Reconstruction')
    plt.xlabel('Original Vorticity Magnitude')
    plt.ylabel('AE Reconstructed Vorticity')
    plt.title('AttentionSE: Vorticity Fidelity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(vort_orig, vort_pca, alpha=0.3, s=10, label='PCA', color='salmon')
    plt.plot([0, vort_orig.max()], [0, vort_orig.max()], 'r--', label='Perfect Reconstruction')
    plt.xlabel('Original Vorticity Magnitude')
    plt.ylabel('PCA Reconstructed Vorticity')
    plt.title('PCA (Baseline): Vorticity Fidelity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, 'Documentation', 'vorticity_fidelity.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved vorticity plot to {out_path}")

    # Histogram of errors
    plt.figure(figsize=(10, 6))
    ae_error = np.abs(vort_ae - vort_orig)
    pca_error = np.abs(vort_pca - vort_orig)
    
    plt.hist(ae_error, bins=50, alpha=0.5, label='AE Error', color='skyblue')
    plt.hist(pca_error, bins=50, alpha=0.5, label='PCA Error', color='salmon')
    plt.xlabel('Vorticity Magnitude Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gradient Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path_hist = os.path.join(PROJECT_ROOT, 'Documentation', 'vorticity_error_dist.png')
    plt.savefig(out_path_hist, dpi=300)
    print(f"Saved vorticity error histogram to {out_path_hist}")

    print(f"Mean Vorticity Error (AE): {ae_error.mean():.6f}")
    print(f"Mean Vorticity Error (PCA): {pca_error.mean():.6f}")
    improvement = (pca_error.mean() - ae_error.mean()) / pca_error.mean() * 100
    print(f"AE Improvement in Gradient Reconstruction: {improvement:.2f}%")

if __name__ == "__main__":
    main()
