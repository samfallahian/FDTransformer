import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec

# ── Color constants ──────────────────────────────────────────────────────────
# User's new requirements: Blue (initial), Red (new), Green (overlap)
BLUE_COL   = '#7BAFD4'  # Carolina blue (initial cube T1)
RED_COL    = '#ef4444'  # Red (newest slice)
NAVY_COL   = '#000080'  # Navy (growing independent part)
GREEN_COL  = '#22c55e'  # Overlapping data (with T1)
BG_COL     = '#cbd5e1'  # Background volume pts (light gray)
PASS_COL   = '#94a3b8'  # Passed blocks (darker gray)
EDGE_COL   = '#475569'  # Darker edge for clarity

# ── projection helpers ───────────────────────────────────────────────────
def rotY(p, a):
    c,s = np.cos(a),np.sin(a)
    return np.c_[p[:,0]*c+p[:,2]*s, p[:,1], -p[:,0]*s+p[:,2]*c]

def rotX(p, a):
    c,s = np.cos(a),np.sin(a)
    return np.c_[p[:,0], p[:,1]*c-p[:,2]*s, p[:,1]*s+p[:,2]*c]

def proj(pts, ra=0.52, rb=0.28):
    r = rotX(rotY(pts, ra), rb)
    return r[:,0], -r[:,1], r[:,2]   # sx, sy, depth

def draw_cube(ax, center, color, alpha=0.3, ra=0.52, rb=0.28, zorder=1):
    """
    Draw a unit cube centered at 'center' (x, y, z) using projected polygons.
    """
    cx, cy, cz = center
    # Define 8 vertices of a unit cube
    v = np.array([
        [cx-0.5, cy-0.5, cz-0.5], [cx+0.5, cy-0.5, cz-0.5],
        [cx+0.5, cy+0.5, cz-0.5], [cx-0.5, cy+0.5, cz-0.5],
        [cx-0.5, cy-0.5, cz+0.5], [cx+0.5, cy-0.5, cz+0.5],
        [cx+0.5, cy+0.5, cz+0.5], [cx-0.5, cy+0.5, cz+0.5]
    ])
    
    faces = [
        [0, 1, 2, 3], # back
        [4, 5, 6, 7], # front
        [0, 1, 5, 4], # bottom
        [2, 3, 7, 6], # top
        [0, 3, 7, 4], # left
        [1, 2, 6, 5]  # right
    ]
    
    sx, sy, dz = proj(v, ra, rb)
    
    # Sort faces back-to-front within the cube
    face_data = []
    for f_idx in faces:
        avg_depth = np.mean(dz[f_idx])
        face_data.append((avg_depth, f_idx))
    
    face_data.sort(key=lambda x: x[0], reverse=True) # High depth to low depth
    
    for _, f_idx in face_data:
        poly_pts = list(zip(sx[f_idx], sy[f_idx]))
        
        # Draw base face
        poly = Polygon(poly_pts, closed=True, facecolor=color, alpha=alpha, 
                       edgecolor=EDGE_COL, linewidth=0.25, zorder=zorder)
        ax.add_patch(poly)

def draw_step_3d(ax, step_idx):
    """
    ax: the matplotlib axis to draw into
    step_idx: current time step (0 to 4)
    """
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')

    ra, rb = 0.52, 0.28
    
    # Grid: 15(x) x 5(y) x 5(z) — User wants a shorter tail on the right
    domain_x = np.arange(0, 13)
    domain_y = np.arange(-2, 3)
    domain_z = np.arange(-2, 3)
    
    # Center of windows: step 0 at x=2, step 1 at x=3, ... step 5 at x=7
    window_centers = [2, 3, 4, 5, 6, 7]
    current_center_x = window_centers[step_idx]
    
    # DRAWING PLAN (Painter's Algorithm):
    all_pts = []
    # Grid: 15(x) x 5(y) x 5(z) — User wants a shorter tail on the right
    domain_x = np.arange(0, 14)
    domain_y = np.arange(-2, 3)
    domain_z = np.arange(-2, 3)
    
    for x in domain_x:
        for y in domain_y:
            for z in domain_z:
                all_pts.append((x, y, z))
    all_pts = np.array(all_pts, dtype=float)
    
    # Project centers for sorting
    _, _, dz_centers = proj(all_pts, ra, rb)
    order = np.argsort(dz_centers)[::-1] 
    
    for idx in order:
        p = all_pts[idx]
        x_p = p[0]
        
        # Default for background
        color = BG_COL
        alpha = 0.02
        
        is_current = abs(x_p - current_center_x) <= 2
        
        if is_current:
            # Color logic based on spatial position relative to T1 (x <= 4)
            if step_idx == 0:
                color = BLUE_COL # Initial cube is all blue
            else:
                new_x = current_center_x + 2
                overlaps_t1 = x_p <= 4
                
                if x_p == new_x:
                    color = RED_COL # Newest slice is red
                elif not overlaps_t1:
                    # Independent part (x > 4) grows as we move
                    color = NAVY_COL
                else:
                    # Overlaps with T1 (x <= 4)
                    color = GREEN_COL
            alpha = 0.7
        else:
            # Check if it was in any PREVIOUS window but NOT current (already passed)
            was_active = False
            for prev_idx in range(step_idx):
                if abs(x_p - window_centers[prev_idx]) <= 2:
                    was_active = True
                    break
            
            if was_active:
                color = PASS_COL
                alpha = 0.1
        
        draw_cube(ax, p, color, alpha=alpha, ra=ra, rb=rb, zorder=1)

    # Label on the right is handled in main by placing text to the right of ax
    
    # Adjusted limits for 14 blocks, zoom out ~20% (10% + additional 10%)
    # Center 6.5, width 18.0
    ax.set_xlim(-2.5, 15.5)
    # Center 2.5, height 13.5
    ax.set_ylim(-4.25, 9.25)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import matplotlib.patches as mpatches
    # 6 steps, 2 columns (x=0,1,2 on left; x=3,4,5 on right)
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    
    # Left margin for legend, right margin for labels
    # gs = rows=3, cols=2
    gs = GridSpec(3, 2, figure=fig, hspace=0.1, wspace=0.35, left=0.1, right=0.9, top=0.95, bottom=0.15)
    
    for i in range(6):
        # row: i%3, col: i//3
        ax = fig.add_subplot(gs[i % 3, i // 3])
        draw_step_3d(ax, i)
        
        # Add (x,y,z) label to the right of each subplot
        x_val = i 
        ax.text(1.05, 0.5, f"(x={x_val}, y=0, z=0)", transform=ax.transAxes, 
                fontsize=10, fontweight='bold', va='center', fontfamily='monospace')
        
        print(f"Finished drawing step {i}")
    
    # Add legend to the bottom center
    legend_elements = [
        mpatches.Patch(facecolor=BLUE_COL, edgecolor=EDGE_COL, label='Initial Cube', alpha=0.7),
        mpatches.Patch(facecolor=RED_COL, edgecolor=EDGE_COL, label='Newest Slice', alpha=0.7),
        mpatches.Patch(facecolor=GREEN_COL, edgecolor=EDGE_COL, label='Overlap with T1', alpha=0.7),
        mpatches.Patch(facecolor=NAVY_COL, edgecolor=EDGE_COL, label='Growing Independent Cube', alpha=0.7),
        mpatches.Patch(facecolor=BG_COL, edgecolor=EDGE_COL, label='Background', alpha=0.2),
        mpatches.Patch(facecolor=PASS_COL, edgecolor=EDGE_COL, label='Passed', alpha=0.2),
    ]
    
    # Place legend at the bottom of the figure
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               fontsize=9, frameon=True, title="Legend", title_fontsize=10, ncol=3)
    
    out_png = '/Users/kkreth/PycharmProjects/cgan/Documentation/time_step_animation.png'
    out_pdf = '/Users/kkreth/PycharmProjects/cgan/Documentation/time_step_animation.pdf'
    plt.savefig(out_png, dpi=150, bbox_inches='tight') 
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    main()
