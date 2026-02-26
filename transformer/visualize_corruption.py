import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_corruption_logic():
    # Parameters (scaled down for visualization)
    B = 1         # 1 sample
    T = 208       # Time sequence length
    LATENT_DIM = 47 # Latent dimension
    INPUT_DIM = 52  # Total input dimension (including coords, param, time)
    
    # 1. Create a "clean" sample (all zeros for contrast)
    data = torch.zeros((B, T, INPUT_DIM))
    
    # 2. Define corruption levels to show
    levels = [0.01, 0.05, 0.10] # 1%, 5%, 10%
    
    # Use a custom discrete colormap for clarity
    from matplotlib.colors import ListedColormap
    # 0 = Clean (light gray), 1 = Corrupted (vibrant blue or orange)
    cmap = ListedColormap(['#f0f0f0', '#1f77b4']) # Light gray and blue

    fig, axes = plt.subplots(len(levels), 1, figsize=(15, 12))
    fig.suptitle('Visualizing Data Corruption (Latent Space Only)', fontsize=16)

    for i, level in enumerate(levels):
        # We only corrupt the latent part: data[:, :, :LATENT_DIM]
        corrupted_data = data.clone()
        
        # Total elements in the latent part
        total_latent_elements = B * T * LATENT_DIM
        num_to_corrupt = int(level * total_latent_elements)
        
        # Generate random indices
        flat_indices = torch.randperm(total_latent_elements)[:num_to_corrupt]
        
        # Mark corrupted elements with 1 (others stay 0)
        latents = corrupted_data[:, :, :LATENT_DIM].reshape(-1)
        latents[flat_indices] = 1.0 # Highlight corrupted elements
        corrupted_mask = latents.reshape(T, LATENT_DIM).numpy()
        
        # Plotting
        ax = axes[i]
        im = ax.imshow(corrupted_mask.T, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.set_title(f'{level*100:.0f}% Corruption ({num_to_corrupt} random elements replaced)')
        ax.set_ylabel('Latent Dim (0-46)')
        if i == len(levels) - 1:
            ax.set_xlabel('Sequence Position (0-207)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Save alongside this script to avoid CWD issues
    output_dir = os.path.dirname(__file__) or "."
    output_path = os.path.join(output_dir, 'corruption_visualization_flow.png')
    plt.savefig(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    visualize_corruption_logic()
