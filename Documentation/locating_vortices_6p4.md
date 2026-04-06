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
| 1 | 122 | 1.02 | 46.57 | Negative to Positive |
| 2 | 228 | 1.90 | 60.41 | Positive to Negative |
| 3 | 292 | 2.43 | 58.99 | Negative to Positive |
| 4 | 315 | 2.63 | 48.69 | Positive to Negative |
| 5 | 346 | 2.88 | 36.11 | Negative to Positive |
| 6 | 370 | 3.08 | 36.00 | Positive to Negative |
| 7 | 394 | 3.28 | 45.63 | Negative to Positive |
| 8 | 433 | 3.61 | 65.15 | Positive to Negative |
| 9 | 507 | 4.23 | 57.08 | Negative to Positive |
| 10 | 586 | 4.88 | 47.26 | Positive to Negative |
| 11 | 663 | 5.53 | 47.23 | Negative to Positive |
| 12 | 835 | 6.96 | 40.75 | Positive to Negative |
| 13 | 870 | 7.25 | 48.00 | Negative to Positive |
| 14 | 889 | 7.41 | 50.92 | Positive to Negative |
| 15 | 904 | 7.53 | 52.08 | Negative to Positive |
| 16 | 962 | 8.02 | 60.47 | Positive to Negative |

*Note: Only selected high-intensity events are listed above. Full data is available in `vorticity_search_full.csv`.*

## 3. Temporal Stability and Verification

Each event was verified for temporal stability to ensure the reversal represents a physical reorganization of the flow rather than momentary noise.

*   **100ms Pre-Reversal:** The vortex maintains a consistent rotational sign (e.g., $\omega_z > 0$) for at least 12 steps (100ms at 120Hz) leading into the event.
*   **100ms Post-Reversal:** The vortex re-establishes a stable rotation in the opposite direction for at least 100ms (12 steps) following the sign-flip.
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
*   **Detailed Analysis:** [Comprehensive Evaluation PDF Suite](vortex_reversal/comprehensive_evaluation.md)

**Conclusion:** The identified events represent stable, physically significant vortex reversals. The transition from $x \in [-30, 30]$ ensures the analysis focuses on the interaction region of interest, and the use of standard CFD criteria (Vorticity, Q-Criterion) provides a rigorous basis for these findings.
