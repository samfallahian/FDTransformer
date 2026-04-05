# Identification and Documentation of Vortex Reversals in 6p4 Dataset

This document provides a technical summary of the vortex reversal events identified in the `6p4` fluid dynamics dataset. It details the methodology for 3D vortex identification, the temporal stability of these events, and the visualization suite developed for scientific documentation.

## 1. Methodology: 3D Vortex Identification

To locate vortices within the point data, we employ standard practices in computational fluid dynamics (CFD) to derive rotational structures from the raw velocity field $\vec{v} = (v_x, v_y, v_z)$.

### A. Vorticity Vector Calculation
The primary metric used is the **Vorticity Vector** ($\vec{\omega}$), defined as the curl of the velocity field:
$$\vec{\omega} = \nabla \times \vec{v}$$
In our implementation, we calculate the components (specifically $\omega_z$) using central differences on the spatial grid:
$$\omega_z = \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}$$
*   **Citation:** *Jeong, J., & Hussain, F. (1995). On the identification of a vortex. Journal of Fluid Mechanics, 285, 69-94.*

### B. The Q-Criterion (Vortex Core Identification)
To distinguish between simple shear layers and true vortex cores, we calculate the **Q-Criterion**. $Q$ represents the local balance between the rotation rate and the strain rate. A vortex is defined as a region where $Q > 0$, indicating that rotation dominates.
$$Q = \frac{1}{2}(||\Omega||^2 - ||S||^2)$$
*   **Citation:** *Hunt, J. C. R., Wray, A. A., & Moin, P. (1988). Eddies, streams, and convergence zones in turbulent flows. NASA Report CTR-S88.*

## 2. Summary of Vortex Reversal Events

A systematic search across 1200 timesteps identified **18 distinct reversal events**. Each event occurs within a $x \in [-30, 30]$ spatial window and coincides with a trajectory reversal in the $y$-coordinate.

| Event # | Step | Time (s) | Y-Pos | Reversal Type (Z-Vorticity) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 121 | 2.00 | 46.22 | Negative to Positive |
| 2 | 175 | 2.90 | 57.64 | Positive to Negative |
| 3 | 227 | 3.77 | 60.27 | Positive to Negative |
| 4 | 291 | 4.83 | 59.32 | Negative to Positive |
| 5 | 345 | 5.73 | 36.34 | Negative to Positive |
| 6 | 432 | 7.18 | 64.82 | Positive to Negative (Strong) |
| 7 | 470 | 7.82 | 66.70 | Positive to Negative |
| 8 | 506 | 8.42 | 57.34 | Negative to Positive |
| 9 | 585 | 9.73 | 47.46 | Positive to Negative |
| 10 | 662 | 11.02 | 46.80 | Negative to Positive |
| 11 | 694 | 11.55 | 63.21 | Positive to Negative |
| 12 | 756 | 12.58 | 65.52 | Negative to Positive |
| 17 | 903 | 15.03 | 52.05 | Negative to Positive |
| 18 | 961 | 16.00 | 60.23 | Positive to Negative |

*Note: Only selected high-intensity events are listed above. Full data is available in `vorticity_search_full.csv`.*

## 3. Temporal Stability and Verification

Each event was verified for temporal stability to ensure the reversal represents a physical reorganization of the flow rather than momentary noise.

*   **100ms Pre-Reversal:** The vortex maintains a consistent rotational sign (e.g., $\omega_z > 0$) for at least 6 steps (100ms at 60Hz) leading into the event.
*   **100ms Post-Reversal:** The vortex re-establishes a stable rotation in the opposite direction for at least 100ms following the sign-flip.
*   **Physical Correlation:** 100% of identified reversals correlate with the "top" or "bottom" points in the positional data from `6p4.txt`, where the $y$-velocity vector reverses direction.

## 4. Documentation Suite (1200 DPI & Interactive)

For each event, a dedicated documentation directory has been created under `Documentation/vortex_reversals/event_[STEP]`.

### A. Interactive 3D Animations (`vortex_interactive.html`)
Created using the **Plotly** scientific package, these files allow for:
*   **3D Isosurfaces:** Visualizing the vortex core volume using the Q-criterion.
*   **Velocity Cones:** 3D arrows showing the actual flow direction and rotation.
*   **Temporal Scrubbing:** A slider to move frame-by-frame through the 400ms window.

### B. High-Resolution Static Comparisons (`vortex_comparison_1200dpi.pdf`)
*   **Resolution:** 1200 DPI for publication-quality printing.
*   **Content:** Side-by-side comparison of the vortex state pre- and post-reversal, showing the sign-flip in the $z$-vorticity field.

**Conclusion:** The identified events represent stable, physically significant vortex reversals. The transition from $x \in [-30, 30]$ ensures the analysis focuses on the interaction region of interest, and the use of standard CFD criteria (Vorticity, Q-Criterion) provides a rigorous basis for these findings.
