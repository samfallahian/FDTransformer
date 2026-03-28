import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
RESULTS_JSON = "evaluation_results.json"
DOC_DIR = "/Users/kkreth/PycharmProjects/cgan/Documentation"

if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

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
    
    # 1. RMSE vs Position (Uncomment if needed)
    # plt.figure(figsize=(10, 6))
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_per_position.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_per_position.png')}")
    
    # 2. RMSE per Window (Uncomment if needed)
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_per_window.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_per_window.png')}")
    
    # 2b. Staircase RMSE (KEEP AS IT WAS THE 'ONE DIAGRAM' PREVIOUSLY DONE)
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_staircase.index, rmse_staircase.values, marker='s', linestyle='--', color='purple')
    plt.title('Staircase Evaluation: T80 RMSE vs Context Time Steps (Velocity Units)')
    plt.xlabel('Number of Context Time Steps provided')
    plt.ylabel('T80 RMSE (Velocity Units)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    for x, y in zip(rmse_staircase.index, rmse_staircase.values):
        plt.text(x, y, f"{y:.4e}", verticalalignment='bottom')
    plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_staircase.png'))
    plt.close()
    print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_staircase.png')}")

    # 3. RMSE per Experiment (Uncomment if needed)
    # plt.figure(figsize=(12, 6))
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_per_experiment.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_per_experiment.png')}")

    # 4. RMSE vs Y/Z Coordinate space (Uncomment if needed)
    # plt.figure(figsize=(10, 8))
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_yz_space.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_yz_space.png')}")

    # 5. 3D Error Density/Surface Plot (Uncomment if needed)
    # fig = plt.figure(figsize=(12, 10))
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_3d_density.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_3d_density.png')}")

    # 6. Creative Plot: Hexbin Error Density (Uncomment if needed)
    # plt.figure(figsize=(10, 8))
    # ...
    # plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_yz_hexbin.png'))
    # plt.close()
    # print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_yz_hexbin.png')}")

    # 7. Interleave/Jump Collapse Plot
    if 'interleave_summary' in data:
        df_int = pd.DataFrame(data['interleave_summary'])
        
        plt.figure(figsize=(12, 7))
        
        # Plot Interleave (Even given Odd)
        mask_inter = df_int['mode'] == 'interleave'
        plt.plot(df_int[mask_inter]['c'], df_int[mask_inter]['rmse'], marker='o', label='Interleave (Context T1..2n-1)')
        
        # Plot Jump C=1
        mask_jump = df_int['mode'] == 'jump_c1'
        plt.plot(df_int[mask_jump]['p'], df_int[mask_jump]['rmse'], marker='s', linestyle='--', label='Jump (Context T1, Predict T2..P)')
        
        # Plot a few Var Context
        for c in [2, 10, 20]:
            mask_var = df_int['mode'] == f'var_c{c}'
            if not df_int[mask_var].empty:
                plt.plot(df_int[mask_var]['p'], df_int[mask_var]['rmse'], marker='x', alpha=0.6, label=f'Context T1..T{c}')
        
        plt.axhline(y=0.05, color='r', linestyle=':', label='Collapse Threshold (0.05)')
        plt.title('RMSE Collapse Analysis: Interleave and Progressive Jumps')
        plt.xlabel('Number of Frames (Context or Prediction)')
        plt.ylabel('RMSE (Velocity Units)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(DOC_DIR, 'transformer_rmse_interleave_collapse.png'))
        plt.close()
        print(f"Saved: {os.path.join(DOC_DIR, 'transformer_rmse_interleave_collapse.png')}")

    print("All requested figures generated successfully.")

if __name__ == "__main__":
    main()
