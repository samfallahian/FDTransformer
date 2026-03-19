import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ── caption text ─────────────────────────────────────────────────────────────
# Figure 2.  Local 5×5×5 volumetric sampling neighborhood used to form velocity cubes.
# Top row: three orthogonal cross-sections through the cube center — XY (blue, z = 0), XZ (teal, y = 0), and YZ (amber, x = 0) —
# each containing 25 of the 125 measurement points.  Bottom: full 3D projection showing all 125 points;
# colored points lie on one or more cross-section planes; gray points are off-plane.
# The red marker indicates the target centroid at (0, 0, 0).
# Each point carries a 375-dimensional feature vector (125 pts × 3 velocity components v_x, v_y, v_z).
# ─────────────────────────────────────────────────────────────────────────────

DPI = 600
FIG_W = 7.2
FIG_H = 8.8

B  = '#378ADD'
T  = '#1D9E75'
A  = '#BA7517'
R  = '#E24B4A'
OFF = '#c0bfb7'

# ── grid ────────────────────────────────────────────────────────────────────
G = np.array([(x,y,z) for z in range(-2,3) for y in range(-2,3) for x in range(-2,3)],
             dtype=float)  # (125, 3)

# ── 3-D projection helpers ───────────────────────────────────────────────────
def rotY(p, a):
    c,s = np.cos(a),np.sin(a)
    return np.c_[p[:,0]*c+p[:,2]*s, p[:,1], -p[:,0]*s+p[:,2]*c]

def rotX(p, a):
    c,s = np.cos(a),np.sin(a)
    return np.c_[p[:,0], p[:,1]*c-p[:,2]*s, p[:,1]*s+p[:,2]*c]

def proj(pts, ra=0.52, rb=0.28):
    r = rotX(rotY(pts, ra), rb)
    return r[:,0], -r[:,1], r[:,2]   # sx, sy, depth

# ── 2D slice panel ───────────────────────────────────────────────────────────
def draw_slice(ax, mask_fn, hfn, vfn, col, hlabel, vlabel, third_label):
    ax.set_facecolor('white')
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.7, 2.7)
    ax.set_aspect('equal')

    # Alternating cell shading
    for j in range(-2, 3):
        for i in range(-2, 3):
            if (i+j) % 2 == 0:
                rect = plt.Rectangle((i-0.5, j-0.5), 1, 1,
                                     facecolor='#00000008', zorder=0)
                ax.add_patch(rect)

    # Grid lines
    for v in np.arange(-2.5, 3, 1):
        ax.axhline(v, color='#00000014', lw=0.5, zorder=1)
        ax.axvline(v, color='#00000014', lw=0.5, zorder=1)

    # Centre guides
    ax.axhline(0, color=col+'30', lw=0.8, ls=(0,(3,4)), zorder=2)
    ax.axvline(0, color=col+'30', lw=0.8, ls=(0,(3,4)), zorder=2)

    # Colored border
    for spine in ax.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(1.2)
        spine.set_alpha(0.55)

    pts_slice = G[np.array([mask_fn(p) for p in G])]

    for p in pts_slice:
        h, v = hfn(p), vfn(p)
        ic = (p[0]==0 and p[1]==0 and p[2]==0)
        if ic:
            ax.plot(h, v, 'o', color=R, ms=6.5, zorder=6)
            ax.plot([h-0.45, h+0.45], [v, v], '-', color=R, lw=0.9, alpha=0.5, zorder=5)
            ax.plot([h, h], [v-0.45, v+0.45], '-', color=R, lw=0.9, alpha=0.5, zorder=5)
        else:
            ax.plot(h, v, 'o', color=col, ms=4.2, alpha=0.78, zorder=4)

    ax.set_xticks(range(-2,3))
    ax.set_yticks(range(-2,3))
    ax.tick_params(labelsize=6, length=2, width=0.5, color='#888', labelcolor='#666',
                   pad=2)

# ── 3D composite panel ───────────────────────────────────────────────────────
def draw_3d(ax):
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')

    ra, rb = 0.52, 0.28
    sc = 1.0   # normalised — we work in projected coords directly

    def p3(pts):
        sx, sy, dz = proj(np.atleast_2d(pts), ra, rb)
        return sx, sy, dz

    # Cage lines
    cage_color = '#00000009'
    for v in range(-2, 3):
        for u in range(-2, 3):
            for axis in ['x','y','z']:
                if axis=='x':
                    a = np.array([[-2,v,u],[2,v,u]], float)
                elif axis=='y':
                    a = np.array([[v,-2,u],[v,2,u]], float)
                else:
                    a = np.array([[v,u,-2],[v,u,2]], float)
                sx,sy,_ = p3(a)
                ax.plot(sx, sy, '-', color='#000000', alpha=0.06, lw=0.35, zorder=1)

    # Three planes (always visible, painter sorted)
    plane_defs = [
        ('xy', B, np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]], float)),
        ('xz', T, np.array([[-2,0,-2],[2,0,-2],[2,0,2],[-2,0,2]], float)),
        ('yz', A, np.array([[0,-2,-2],[0,-2,2],[0,2,2],[0,2,-2]], float)),
    ]
    plane_depths = []
    for pid, col, corners in plane_defs:
        sx,sy,dz = p3(corners)
        plane_depths.append((dz.mean(), pid, col, corners))
    plane_depths.sort(key=lambda x: x[0])

    for _,pid,col,corners in plane_depths:
        sx,sy,_ = p3(corners)
        poly = Polygon(list(zip(sx,sy)), closed=True,
                       facecolor=col+'1e', edgecolor=col+'77',
                       linewidth=0.7, zorder=3)
        ax.add_patch(poly)

    # All 125 points — color by plane membership
    sx_all, sy_all, dz_all = p3(G)
    order = np.argsort(dz_all)

    for idx in order:
        p = G[idx]
        x,y,z = p
        ic    = (x==0 and y==0 and z==0)
        onXY  = (z==0)
        onXZ  = (y==0)
        onYZ  = (x==0)
        sx, sy = sx_all[idx], sy_all[idx]

        if ic:
            ax.plot(sx, sy, 'o', color=R, ms=9, zorder=8,
                    markeredgecolor='white', markeredgewidth=0.4)
            ax.plot([sx-0.22,sx+0.22],[sy,sy], '-', color=R, lw=0.8, alpha=0.4, zorder=7)
            ax.plot([sx,sx],[sy-0.22,sy+0.22], '-', color=R, lw=0.8, alpha=0.4, zorder=7)
        elif onXY and not onXZ and not onYZ:
            ax.plot(sx, sy, 'o', color=B, ms=4.5, alpha=0.82, zorder=5)
        elif onXZ and not onXY and not onYZ:
            ax.plot(sx, sy, 'o', color=T, ms=4.5, alpha=0.82, zorder=5)
        elif onYZ and not onXY and not onXZ:
            ax.plot(sx, sy, 'o', color=A, ms=4.5, alpha=0.82, zorder=5)
        elif onXY or onXZ or onYZ:
            # intersection lines — blend
            ax.plot(sx, sy, 'o', color=B, ms=4.0, alpha=0.75, zorder=5)
        else:
            ax.plot(sx, sy, 'o', color=OFF, ms=2.8, alpha=0.52, zorder=4)

    # Axis arrows removed as per user request

    # Point count removed as per user request

    ax.autoscale_view()
    ax.margins(0.08)

# ── layout ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='white')

gs = GridSpec(3, 3, figure=fig,
              height_ratios=[2.4, 0.15, 3.2],
              hspace=0.0, wspace=0.32,
              left=0.08, right=0.97, top=0.96, bottom=0.05)

# Top row — 2D slices
ax_xy = fig.add_subplot(gs[0, 0])
ax_xz = fig.add_subplot(gs[0, 1])
ax_yz = fig.add_subplot(gs[0, 2])

draw_slice(ax_xy, lambda p: p[2]==0, lambda p: p[0], lambda p: p[1], B, 'X','Y','Z')
draw_slice(ax_xz, lambda p: p[1]==0, lambda p: p[0], lambda p: p[2], T, 'X','Z','Y')
draw_slice(ax_yz, lambda p: p[0]==0, lambda p: p[1], lambda p: p[2], A, 'Y','Z','X')

# Plane label above each panel
titles = [
    ('XY PLANE', 'z = 0  ·  25 pts', B),
    ('XZ PLANE', 'y = 0  ·  25 pts', T),
    ('YZ PLANE', 'x = 0  ·  25 pts', A),
]
for ax, (title, sub, col) in zip([ax_xy, ax_xz, ax_yz], titles):
    ax.set_title(f'{title}\n{sub}', fontsize=7.5, color=col, fontfamily='monospace',
                 fontweight='bold', linespacing=1.6, pad=4)

# Bottom row — 3D composite spanning all columns
ax_3d = fig.add_subplot(gs[2, :])
draw_3d(ax_3d)
# 3D title removed as per user request

# ── connector lines in figure coords ─────────────────────────────────────────
slice_axes = [ax_xy, ax_xz, ax_yz]
colors3     = [B, T, A]

pos3d   = ax_3d.get_position()
top3d_y = pos3d.y1
x_targets_fig = [pos3d.x0 + (pos3d.x1-pos3d.x0)*f for f in [0.22, 0.50, 0.78]]

for sax, col, xt in zip(slice_axes, colors3, x_targets_fig):
    pos   = sax.get_position()
    x_src = (pos.x0 + pos.x1) / 2
    y_src = pos.y0

    line = plt.Line2D([x_src, xt], [y_src, top3d_y],
                      transform=fig.transFigure,
                      color=col, lw=0.9, ls=(0,(4,3)), alpha=0.55, zorder=10)
    fig.add_artist(line)

    dot = mpatches.Circle((x_src, y_src), radius=0.004,
                           transform=fig.transFigure,
                           color=col, alpha=0.7, zorder=11)
    fig.add_artist(dot)

# ── legend ───────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0],[0], marker='o', color='none', markerfacecolor=B,  markersize=5, label='XY plane  (z = 0)  –  25 pts'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=T,  markersize=5, label='XZ plane  (y = 0)  –  25 pts'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=A,  markersize=5, label='YZ plane  (x = 0)  –  25 pts'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=OFF,markersize=5, label='Off-plane points  –  75 pts'),
    Line2D([0],[0], marker='o', color='none', markerfacecolor=R,  markersize=6, label='Target centroid  (0, 0, 0)'),
]
fig.legend(handles=legend_elements,
           loc='lower center', ncol=3,
           fontsize=6.2, frameon=True, framealpha=0.92,
           edgecolor='#ddd', fancybox=False,
           borderpad=0.7, labelspacing=0.5, handletextpad=0.4,
           prop={'family':'monospace', 'size':6.2},
           bbox_to_anchor=(0.525, 0.005))

# ── caption ──────────────────────────────────────────────────────────────────
# The user previously wanted "Figure 2" and the rest of the text to be the same size/font as the legend (monospace, 6.2).
# As of the latest update, the entire block of text (caption) has been removed from the figure itself.
# The text content is now preserved only in the comments at the top of this file.

out_png = '/Users/kkreth/PycharmProjects/cgan/Documentation/LFM_Figure2.png'
out_pdf = '/Users/kkreth/PycharmProjects/cgan/Documentation/LFM_Figure2.pdf'

fig.savefig(out_png, dpi=DPI, bbox_inches='tight', facecolor='white', format='png')
print(f'Saved: {out_png}')

fig.savefig(out_pdf, dpi=1200, bbox_inches='tight', facecolor='white', format='pdf')
print(f'Saved: {out_pdf}')

plt.close(fig)
