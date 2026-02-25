import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_cube_mapping():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define the 5x5x5 grid indices
    indices = np.arange(-2, 3)
    x, y, z = np.meshgrid(indices, indices, indices)
    
    # Plot all neighbors
    ax.scatter(x, y, z, c='skyblue', alpha=0.3, s=60, label='Neighbor Points')
    
    # Highlight the centroid
    ax.scatter([0], [0], [0], c='red', s=250, label='Centroid (Target Point)', edgecolors='black', linewidth=2)

    # Draw a wireframe for the neighborhood to emphasize the cubic structure
    for i in indices:
        for j in indices:
            ax.plot([-2, 2], [i, i], [j, j], color='gray', alpha=0.15, linewidth=0.5)
            ax.plot([i, i], [-2, 2], [j, j], color='gray', alpha=0.15, linewidth=0.5)
            ax.plot([i, i], [j, j], [-2, 2], color='gray', alpha=0.15, linewidth=0.5)

    # Labeling with LaTeX-style formatting
    ax.set_xlabel(r'$\Delta x$ (Grid Units)', fontsize=12)
    ax.set_ylabel(r'$\Delta y$ (Grid Units)', fontsize=12)
    ax.set_zlabel(r'$\Delta z$ (Grid Units)', fontsize=12)
    ax.set_title(r'Volumetric Sampling Neighborhood ($5 \times 5 \times 5$ Cube)', fontsize=16, pad=20)
    
    # Add a descriptive text box
    info_text = (
        "Neighborhood Composition:\n"
        "• Total Points (N): 125\n"
        "• Components per Point: 3 ($v_x, v_y, v_z$)\n"
        "• Feature Vector Length: 375\n"
        r"• Spatial Extent: $\pm 2$ grid units"
    )
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right', fontsize=12)
    
    # Adjust view angle for better depth perception
    ax.view_init(elev=20, azim=45)
    
    # Set axis ticks to integers only
    ax.set_xticks(indices)
    ax.set_yticks(indices)
    ax.set_zticks(indices)
    
    plt.tight_layout()
    plt.savefig('Documentation/mapping_visualization.png', dpi=1200, bbox_inches='tight')
    plt.close()
    print("Saved Documentation/mapping_visualization.png")

def plot_experimental_box():
    # Width 8, Height 4
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Dimensions representing the TR-PTV measurement volume (scaled for visualization)
    # Based on the LED array 300x100mm, we assume some depth Z
    L, W, H = 300, 100, 80
    
    # Vertices of the outer measurement volume
    v = np.array([
        [0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0],
        [0, 0, H], [L, 0, H], [L, W, H], [0, W, H]
    ])
    
    # Edges of the outer box
    edges = [
        [v[0], v[1], v[2], v[3], v[0]], # bottom
        [v[4], v[5], v[6], v[7], v[4]], # top
        [v[0], v[4]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]] # verticals
    ]

    for edge in edges:
        edge = np.array(edge)
        ax.plot(edge[:,0], edge[:,1], edge[:,2], color='navy', linewidth=1.5, alpha=0.8)

    # Faces of the outer box
    faces = [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]], 
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]], 
        [v[1], v[2], v[6], v[5]], [v[4], v[7], v[3], v[0]]
    ]
    
    poly3d = Poly3DCollection(faces, facecolors='lightskyblue', linewidths=0.5, edgecolors='navy', alpha=0.1)
    ax.add_collection3d(poly3d)

    # Represent the "Interior sampling region" (after trimming edges)
    trim_x, trim_y, trim_z = 30, 15, 15
    v_in = np.array([
        [trim_x, trim_y, trim_z], [L-trim_x, trim_y, trim_z], 
        [L-trim_x, W-trim_y, trim_z], [trim_x, W-trim_y, trim_z],
        [trim_x, trim_y, H-trim_z], [L-trim_x, trim_y, H-trim_z], 
        [L-trim_x, W-trim_y, H-trim_z], [trim_x, W-trim_y, H-trim_z]
    ])
    
    # Inner box edges (dashed)
    inner_edges = [
        [v_in[0], v_in[1], v_in[2], v_in[3], v_in[0]],
        [v_in[4], v_in[5], v_in[6], v_in[7], v_in[4]],
        [v_in[0], v_in[4]], [v_in[1], v_in[5]], [v_in[2], v_in[6]], [v_in[3], v_in[7]]
    ]
    for edge in inner_edges:
        edge = np.array(edge)
        ax.plot(edge[:,0], edge[:,1], edge[:,2], color='crimson', linestyle='--', linewidth=1, alpha=0.7)

    # Fill the sampling region with a slightly different color
    inner_faces = [
        [v_in[0], v_in[1], v_in[2], v_in[3]], [v_in[4], v_in[5], v_in[6], v_in[7]],
        [v_in[0], v_in[1], v_in[5], v_in[4]], [v_in[2], v_in[3], v_in[7], v_in[6]],
        [v_in[1], v_in[2], v_in[6], v_in[5]], [v_in[4], v_in[7], v_in[3], v_in[0]]
    ]
    inner_poly = Poly3DCollection(inner_faces, facecolors='salmon', alpha=0.15)
    ax.add_collection3d(inner_poly)

    # Representative coordinate points (scatter)
    # Sample some points within the volume
    np.random.seed(42)
    sample_pts = np.random.uniform(low=[trim_x, trim_y, trim_z], high=[L-trim_x, W-trim_y, H-trim_z], size=(100, 3))
    ax.scatter(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2], color='crimson', s=5, alpha=0.4, label='Valid Sampling Centroids')

    # Axes and labels
    # Title removed per request
    
    ax.set_xlabel('X (mm) [Streamwise]', fontsize=8, labelpad=15)
    
    # Label only half of the y-axis ticks
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks[::2])
    # Set z-axis ticks to 20, 40, 60, 80
    ax.set_zticks([20, 40, 60, 80])
    ax.set_zlabel('Z (mm) [Wall-normal]', fontsize=8, labelpad=25)
    
    # Force axis limits slightly larger to create internal buffer
    ax.set_xlim(-5, L + 5)
    ax.set_ylim(-5, W + 5)
    ax.set_zlim(0, H + 5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='navy', lw=1.5, label='Total Volume'),
        Line2D([0], [0], color='crimson', lw=1, ls='--', label='Sampling Region'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=4, alpha=0.4, label='Centroids')
    ]
    # Configuration: Legend at bottom center
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=7, frameon=True)

    # Adjust view - Configuration from perm_116
    ax.view_init(elev=10, azim=-45)
    
    # Camera distance - Configuration from perm_116
    try:
        ax.dist = 8.0 
        ax.set_box_aspect([L, W, H]) # Ensure physical aspect ratio
    except:
        pass

    # Manually set axis position - Adjusted for more margin
    ax.set_position([0.15, 0.15, 0.75, 0.75])
    
    plt.savefig('Documentation/experimental_box.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print("Saved Documentation/experimental_box.png")

if __name__ == "__main__":
    os.makedirs('Documentation', exist_ok=True)
    plot_cube_mapping()
    plot_experimental_box()
