import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Coordinates from Ordered_010_Prepare_Dataset.py
X_COORDS = [-50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2, 2, 6, 10, 14, 18, 22, 25, 29, 33, 37, 41, 45, 49]
Z_COORDS = [-33, -29, -25, -21, -17, -13, -9, -5, -1, 3, 7, 11, 14, 18, 22, 26, 30, 34]
Y_COORDS = [-83, -80, -76, -72, -68, -64, -60, -56, -52, -48, -44, -40, -36, -32, -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87]

def create_visualization():
    fig = plt.figure(figsize=(16, 10))
    
    # --- 1. SPATIAL SEARCH SPACE (X, Y, Z) ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Create a grid for the bounding box
    x_min, x_max = min(X_COORDS), max(X_COORDS)
    y_min, y_max = min(Y_COORDS), max(Y_COORDS)
    z_min, z_max = min(Z_COORDS), max(Z_COORDS)
    
    # Draw the bounding box of the search space
    for x in [x_min, x_max]:
        for y in [y_min, y_max]:
            ax1.plot([x, x], [y, y], [z_min, z_max], color='gray', alpha=0.3)
    for x in [x_min, x_max]:
        for z in [z_min, z_max]:
            ax1.plot([x, x], [y_min, y_max], [z, z], color='gray', alpha=0.3)
    for y in [y_min, y_max]:
        for z in [z_min, z_max]:
            ax1.plot([x_min, x_max], [y, y], [z, z], color='gray', alpha=0.3)

    # Show the 26 X-coordinates as dots along the bottom edge to visualize density
    ax1.scatter(X_COORDS, [y_min]*len(X_COORDS), [z_min]*len(X_COORDS), 
                color='red', s=10, label=f'X Coordinates ({len(X_COORDS)} points)')

    # Show three "Training Samples" (lines of X's at different Y, Z)
    # with highlights for the 'questions' (last 4, 8, 16)
    samples = [
        # (Y index, Z index, highlight_count, highlight_color, label)
        (len(Y_COORDS)//4, len(Z_COORDS)//4, 4, 'red', 'Sample 1: Last 4 Targets'),
        (len(Y_COORDS)//2, len(Z_COORDS)//2, 8, 'orange', 'Sample 2: Last 8 Targets'),
        (3*len(Y_COORDS)//4, 3*len(Z_COORDS)//4, 16, 'gold', 'Sample 3: Last 16 Targets')
    ]
    
    base_colors = ['dodgerblue', 'mediumseagreen', 'mediumpurple']
    
    for i, (y_idx, z_idx, h_count, h_color, h_label) in enumerate(samples):
        y_val = Y_COORDS[y_idx]
        z_val = Z_COORDS[z_idx]
        b_color = base_colors[i]
        
        # Split X_COORDS into base and highlight
        split_idx = len(X_COORDS) - h_count
        x_base = X_COORDS[:split_idx+1] # overlap by 1 to connect the line visually
        x_highlight = X_COORDS[split_idx:]
        
        # Plot base part
        ax1.plot(x_base, [y_val]*len(x_base), [z_val]*len(x_base), 
                 color=b_color, linewidth=2, alpha=0.6)
        # Plot highlight part
        ax1.plot(x_highlight, [y_val]*len(x_highlight), [z_val]*len(x_highlight), 
                 color=h_color, linewidth=4, alpha=1.0, label=h_label)
    
    ax1.set_title("Data Search Space (Spatial)\nHighlighting Prediction Targets (Last 4, 8, 16) on 3 samples", fontsize=14)
    ax1.set_xlabel("X (Width)")
    ax1.set_ylabel("Y (Depth)")
    ax1.set_zlabel("Z (Height)")
    ax1.legend()
    
    # Ensure axes are not float if they represent integers
    ax1.set_zticks(Z_COORDS[::4]) # Sparsely label to avoid clutter
    ax1.set_yticks(Y_COORDS[::8])
    ax1.set_xticks(X_COORDS[::5])

    # --- 2. TRAINING LOGIC: INPUTS vs QUESTIONS ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_axis_off()
    
    ax2.text(0.5, 0.98, "Transformer Training Logic & Sequence Math", 
             ha='center', fontsize=14, fontweight='bold', transform=ax2.transAxes)

    # 1. Total Window
    y_pos = 0.85
    ax2.add_patch(patches.Rectangle((0.05, y_pos), 0.9, 0.04, facecolor='lightgray', edgecolor='black', alpha=0.5))
    ax2.text(0.5, y_pos + 0.05, "Full Data Window: 208 Tokens (8 Time Steps × 26 X-coordinates)", 
             ha='center', fontsize=11, fontweight='bold')
    
    # Draw ticks for time steps
    for i in range(1, 8):
        x_tick = 0.05 + (i * 26 / 208) * 0.9
        ax2.plot([x_tick, x_tick], [y_pos, y_pos+0.04], color='black', linewidth=1)
        ax2.text(x_tick - 0.04, y_pos - 0.03, f"T{i}", fontsize=8)
    ax2.text(0.05, y_pos - 0.03, "T0", fontsize=8)

    # 2. Input vs Target shift
    y_in = 0.65
    y_out = 0.55
    # Input: 0 to 206
    in_width = (207/208) * 0.9
    ax2.add_patch(patches.Rectangle((0.05, y_in), in_width, 0.04, facecolor='skyblue', edgecolor='blue'))
    ax2.text(0.06, y_in + 0.05, "INPUT SEQUENCE (Tokens 0 to 206) -> Length = 207", fontsize=10, color='blue')

    # Output: 1 to 207
    out_width = (207/208) * 0.9
    ax2.add_patch(patches.Rectangle((0.05 + (1/208)*0.9, y_out), out_width, 0.04, facecolor='salmon', edgecolor='red'))
    ax2.text(0.95, y_out - 0.04, "TARGET SEQUENCE (Tokens 1 to 207) -> Length = 207", 
             ha='right', fontsize=10, color='darkred')

    # Connecting arrows for next-token prediction
    for i in [10, 50, 100, 150, 200]:
        x_start = 0.05 + (i/208)*0.9
        x_end = 0.05 + ((i+1)/208)*0.9
        ax2.annotate("", xy=(x_end, y_out+0.04), xytext=(x_start, y_in),
                     arrowprops=dict(arrowstyle="->", color="black", alpha=0.3))

    # 3. Prediction Detail
    y_pred = 0.35
    ax2.text(0.05, y_pred + 0.08, "What is each 'Prediction'?", fontsize=11, fontweight='bold')
    # Show one "token" exploded
    ax2.add_patch(patches.Rectangle((0.1, y_pred), 0.15, 0.06, facecolor='salmon', edgecolor='red'))
    ax2.text(0.27, y_pred + 0.015, "= One Prediction Step\n(Vector of 47 Latents)", fontsize=10)
    
    ax2.text(0.55, y_pred + 0.015, "Total Predictions = 207 steps\nTotal Values predicted = 207 × 47 = 9,729", 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # 4. Validation Focus (The Questions)
    y_val = 0.15
    ax2.text(0.05, y_val + 0.08, "Validation Focus (The 'Questions'):", fontsize=11, fontweight='bold')
    
    # Draw the tail of the sequence
    tail_x_start = 0.6
    ax2.add_patch(patches.Rectangle((tail_x_start, y_val), 0.35, 0.04, facecolor='none', edgecolor='black', linestyle='--'))
    
    # Highlight 16, 8, 4
    for count, label, color, offset in [(16, "Last 16", "gold", 0.04), (8, "Last 8", "orange", 0.02), (4, "Last 4", "red", 0.0)]:
        width = (count/208) * 0.9
        ax2.add_patch(patches.Rectangle((0.95 - width, y_val - offset), width, 0.015, color=color, alpha=0.8))
        ax2.text(0.95 - width - 0.01, y_val - offset, label, ha='right', fontsize=9, color='black')

    ax2.text(0.05, y_val - 0.05, "We track 207 predictions, but 'Final Step' performance\n(Last 4/8/16 positions) is our primary quality metric.", 
             fontsize=9, style='italic')

    summary_box = (
        "SUMMARY:\n"
        "• Full Window: 208 steps (8 time × 26 X)\n"
        "• Model Input: Tokens [0...206] (Size: 207)\n"
        "• Model Target: Tokens [1...207] (Size: 207)\n"
        "• Prediction: For each of the 207 inputs, predict 47 latents.\n"
        "• Metrics: RMSE is averaged over Last 4, 8, and 16 steps."
    )
    ax2.text(0.05, -0.1, summary_box, transform=ax2.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray"))

    plt.tight_layout()
    plt.savefig('transformer/training_process_viz.png', dpi=150)
    print("Updated visualization saved to transformer/training_process_viz.png")

if __name__ == "__main__":
    create_visualization()
