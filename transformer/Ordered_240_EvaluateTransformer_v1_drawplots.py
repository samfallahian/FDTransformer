import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
RESULTS_JSON = "evaluation_results.json"

def main():
    if not os.path.exists(RESULTS_JSON):
        print(f"Error: {RESULTS_JSON} not found. Run the evaluation script first.")
        return

    print(f"Loading results from {RESULTS_JSON}...")
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)

    # Reconstruct necessary data structures
    rmse_per_pos = pd.Series(data['rmse_per_pos'])
    rmse_per_pos.index = rmse_per_pos.index.astype(int)
    
    rmse_staircase = pd.Series(data['rmse_staircase'])
    rmse_staircase.index = rmse_staircase.index.astype(int)
    
    rmse_per_param = pd.Series(data['rmse_per_param'])
    
    yz_stats = pd.DataFrame(data['yz_stats'])
    
    # Prediction window RMSEs
    rmse_l4 = data['rmse_l4']
    rmse_l8 = data['rmse_l8']
    rmse_l16 = data['rmse_l16']
    rmse_overall = data['rmse_overall']

    print("GENERATING FIGURES...")
    
    # 1. RMSE vs Position
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_per_pos.index, rmse_per_pos.values, marker='o', linestyle='-', color='b')
    plt.axvspan(22, 25, alpha=0.2, color='red', label='L4 Window')
    plt.axvspan(18, 25, alpha=0.1, color='orange', label='L8 Window')
    plt.axvspan(10, 25, alpha=0.05, color='yellow', label='L16 Window')
    plt.title('RMSE per Position in T80 (Velocity Units)')
    plt.xlabel('Position Index (0-25)')
    plt.ylabel('RMSE (Velocity Units)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig('rmse_per_position.png')
    plt.close()
    print("Saved: rmse_per_position.png")
    
    # 2. RMSE per Window
    plt.figure(figsize=(8, 6))
    windows = ['L4 (Last 4)', 'L8 (Last 8)', 'L16 (Last 16)', 'Overall T80']
    vals = [rmse_l4, rmse_l8, rmse_l16, rmse_overall]
    colors = ['red', 'orange', 'gold', 'green']
    plt.bar(windows, vals, color=colors)
    plt.title('RMSE per Prediction Window (Velocity Units)')
    plt.ylabel('RMSE (Velocity Units)')
    for i, v in enumerate(vals):
        plt.text(i, v + (max(vals)*0.01), f"{v:.4e}", ha='center', fontweight='bold')
    plt.savefig('rmse_per_window.png')
    plt.close()
    print("Saved: rmse_per_window.png")
    
    # 2b. Staircase RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_staircase.index, rmse_staircase.values, marker='s', linestyle='--', color='purple')
    plt.title('Staircase Evaluation: RMSE of Position 8 vs Context Count (Velocity Units)')
    plt.xlabel('Number of Context Points provided from T80')
    plt.ylabel('RMSE (Velocity Units)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    for x, y in zip(rmse_staircase.index, rmse_staircase.values):
        plt.text(x, y, f"{y:.4e}", verticalalignment='bottom')
    plt.savefig('rmse_staircase.png')
    plt.close()
    print("Saved: rmse_staircase.png")

    # 3. RMSE per Experiment
    plt.figure(figsize=(12, 6))
    ax = rmse_per_param.plot(kind='bar', color='skyblue')
    plt.title('RMSE per Experiment (Param) (Velocity Units)')
    plt.ylabel('RMSE (Velocity Units)')
    plt.xlabel('Experiment Param')
    
    # Format x-axis labels to avoid floating point artifacts
    ax.set_xticklabels([f"{float(label.get_text()):.1f}" for label in ax.get_xticklabels()])
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rmse_per_experiment.png')
    plt.close()
    print("Saved: rmse_per_experiment.png")

    # 4. RMSE vs Y/Z Coordinate space (Heatmap-style Scatter)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(yz_stats['y'], yz_stats['z'], c=yz_stats['rmse'], cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(sc, label='RMSE (Velocity Units)')
    plt.title('RMSE Distribution in Y-Z Coordinate Space (Velocity Units)')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('rmse_yz_space.png')
    plt.close()
    print("Saved: rmse_yz_space.png")

    # 5. 3D Error Density/Surface Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter3D(yz_stats['y'], yz_stats['z'], yz_stats['rmse'], 
                       c=yz_stats['rmse'], cmap='magma', s=60)
    
    ax.set_title('3D Error Magnitude across Y-Z Space (Velocity Units)')
    ax.set_xlabel('Y Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('RMSE (Velocity Units)')
    fig.colorbar(img, ax=ax, label='RMSE (Velocity Units)', shrink=0.5, aspect=10)
    
    # Add a "shadow" on the floor
    ax.scatter3D(yz_stats['y'], yz_stats['z'], np.zeros_like(yz_stats['rmse']), 
                 c='gray', alpha=0.1, s=10)
    
    plt.savefig('rmse_3d_density.png')
    plt.close()
    print("Saved: rmse_3d_density.png")

    # 6. Creative Plot: Hexbin Error Density
    plt.figure(figsize=(10, 8))
    # Note: We don't have the raw 'df' here, so we use the aggregated yz_stats for hexbin
    # This might look slightly different than before but is a good approximation
    hb = plt.hexbin(yz_stats['y'], yz_stats['z'], C=yz_stats['rmse'], gridsize=25, cmap='magma', reduce_C_function=np.mean)
    cb = plt.colorbar(hb, label='Mean RMSE (Velocity Units)')
    plt.title('Hexbin RMSE Density Map (Y-Z Plane) (Velocity Units)')
    plt.xlabel('Y Coordinate')
    plt.ylabel('Z Coordinate')
    plt.savefig('rmse_yz_hexbin.png')
    plt.close()
    print("Saved: rmse_yz_hexbin.png")

    print("All figures generated successfully.")

if __name__ == "__main__":
    main()
