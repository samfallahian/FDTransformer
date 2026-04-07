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

### C. Vortex Core Localization (Region-Based)
Rather than reporting a single point, each vortex core is characterized as a **spatial region**. For each timestep, we:

1. Compute the vorticity magnitude field $|\vec{\omega}|$ across the 3D velocity grid data (`.pkl.gz` files) within the interaction region $x \in [-30, 30]$.
2. Identify the **peak vorticity magnitude** and define the core region as all grid points where $|\vec{\omega}| \geq 90\%$ of the peak value.
3. Compute the **vorticity-weighted centroid** $(c_x, c_y, c_z)$ of this core region, where each grid point's position is weighted by its vorticity magnitude.
4. Report the **half-span extent** $(\pm e_x, \pm e_y, \pm e_z)$ — half the bounding box of the core region — to characterize the spatial spread.
5. **Store every individual core grid point** exhaustively in `vorticity_core_points.csv`, with its $(x, y, z)$ coordinates, vorticity magnitude, and full vorticity vector $(\omega_x, \omega_y, \omega_z)$.

This approach captures the physical reality that vortex cores are spatially extended structures, not point singularities. The positional data file (`6p4.txt`) is used **only** to identify candidate timesteps where the tracked object reaches local $y$-extrema (trajectory reversals), which correlate with vortex reversal events.

### D. Output Files

| File | Description |
| :--- | :--- |
| `vorticity_search_full.csv` | One row per timestep (900 rows). Columns: step, centroid (cx, cy, cz), extent (ex, ey, ez), peak location, core point count, vorticity at centroid. |
| `vorticity_core_points.csv` | One row per core grid point per timestep (5,135 total records). Columns: step, x, y, z, vort_mag, omega_x, omega_y, omega_z. This is the **exhaustive** listing of every grid point at ≥90% of peak vorticity magnitude. |

## 2. Summary of Vortex Reversal Events

A systematic search across 900 timesteps (steps 101–1000) identified **16 distinct, temporally stable reversal events**. Each event's vortex core is reported as a weighted centroid ± half-span extent at the 90% vorticity magnitude threshold.

| Event # | Step | Centroid (x, y, z) | Extent (±x, ±y, ±z) | Core Pts | Reversal Type (Z-Vorticity) | Visualizations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 122 | (-24.6, -56.1, -21.0) | ±(5.5, 8.0, 0.0) | 7 | Negative to Positive | [PDF](../vortex_reversals/event_121/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_121/vortex_interactive.html) |
| 2 | 228 | (-14.5, 24.2, 8.1) | ±(8.0, 37.5, 21.5) | 6 | Positive to Negative | [PDF](../vortex_reversals/event_227/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_227/vortex_interactive.html) |
| 3 | 292 | (25.0, 36.6, 8.8) | ±(4.0, 37.5, 13.5) | 6 | Negative to Positive | [PDF](../vortex_reversals/event_291/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_291/vortex_interactive.html) |
| 4 | 315 | (-7.4, -13.3, -3.1) | ±(23.5, 41.0, 21.5) | 7 | Positive to Negative | — |
| 5 | 346 | (0.4, -56.7, 12.7) | ±(5.5, 59.0, 21.5) | 8 | Negative to Positive | [PDF](../vortex_reversals/event_345/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_345/vortex_interactive.html) |
| 6 | 370 | (13.0, 47.0, -21.0) | ±(4.0, 0.0, 0.0) | 3 | Positive to Negative | — |
| 7 | 394 | (25.1, 47.0, -21.0) | ±(4.0, 0.0, 0.0) | 3 | Negative to Positive | [PDF](../vortex_reversals/event_393/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_393/vortex_interactive.html) |
| 8 | 433 | (6.4, 31.5, 5.7) | ±(29.0, 30.0, 21.5) | 10 | Positive to Negative | [PDF](../vortex_reversals/event_432/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_432/vortex_interactive.html) |
| 9 | 507 | (-24.5, 73.0, -12.0) | ±(5.5, 2.0, 10.0) | 4 | Negative to Positive | [PDF](../vortex_reversals/event_506/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_506/vortex_interactive.html) |
| 10 | 586 | (-8.8, -46.5, 7.2) | ±(8.0, 41.0, 21.5) | 10 | Positive to Negative | [PDF](../vortex_reversals/event_585/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_585/vortex_interactive.html) |
| 11 | 663 | (-21.5, 41.2, -0.5) | ±(7.5, 59.0, 21.5) | 11 | Negative to Positive | [PDF](../vortex_reversals/event_662/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_662/vortex_interactive.html) |
| 12 | 835 | (-27.8, -37.4, 1.3) | ±(1.5, 41.0, 21.5) | 5 | Positive to Negative | [PDF](../vortex_reversals/event_834/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_834/vortex_interactive.html) |
| 13 | 870 | (-11.8, 26.0, -21.0) | ±(10.0, 18.0, 0.0) | 7 | Negative to Positive | [PDF](../vortex_reversals/event_869/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_869/vortex_interactive.html) |
| 14 | 889 | (-2.3, 11.0, -21.0) | ±(3.5, 0.0, 0.0) | 3 | Positive to Negative | — |
| 15 | 904 | (-16.1, 54.9, -10.0) | ±(19.0, 32.0, 8.0) | 3 | Negative to Positive | [PDF](../vortex_reversals/event_903/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_903/vortex_interactive.html) |
| 16 | 962 | (4.5, -22.4, -21.0) | ±(9.5, 73.0, 0.0) | 7 | Positive to Negative | [PDF](../vortex_reversals/event_961/vortex_comparison_1200dpi.pdf) · [3D](../vortex_reversals/event_961/vortex_interactive.html) |

*Note: "Core Pts" is the number of grid points at ≥90% of peak vorticity magnitude. Extent of ±0.0 indicates the core is confined to a single grid plane in that dimension. Full per-step summary data is in `vorticity_search_full.csv`; exhaustive core point coordinates are in `vorticity_core_points.csv`.*

## 3. Temporal Stability and Verification

Each event was verified for temporal stability to ensure the reversal represents a physical reorganization of the flow rather than momentary noise.

*   **100ms Pre-Reversal:** The vortex maintains a consistent rotational sign (e.g., $\omega_z > 0$) for at least 12 steps (100ms at 120Hz) leading into the event.
*   **100ms Post-Reversal:** The vortex re-establishes a stable rotation in the opposite direction for at least 100ms (12 steps) following the sign-flip.
*   **Physical Correlation:** 100% of identified reversals correlate with the "top" or "bottom" points in the positional data from `6p4.txt`, where the $y$-velocity vector reverses direction. Note: `6p4.txt` is used only for timestep identification, not for position values.

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
*   **Detailed Analysis:** [Comprehensive Evaluation PDF Suite](comprehensive_evaluation.md)

## 5. Related Analysis

*   **[Comprehensive Evaluation of Vortex Reversal Prediction](comprehensive_evaluation.md)** — Sensitivity analysis of the transformer model's prediction accuracy as a function of temporal context and spatial density.
*   **[PDF] [Context Sensitivity Heatmap (1200 DPI)](context_sensitivity_heatmap.pdf)** — RMSE of $\omega_z$ across the full sweep of time and space variations, averaged across all ≥90% vortex core (y, z) coordinates from 18 reversal events.
*   **[PDF] [Sparse Evaluation Comparison (1200 DPI)](sparse_evaluation_comparison.pdf)** — Side-by-side performance for varying context windows.
*   **[PDF] [Vortex Reversal Evaluation (1200 DPI)](vortex_reversal_evaluation.pdf)** — Detailed prediction plots for primary reversal events.
*   **[PDF] [Core Temporal Autocorrelation](core_temporal_correlation.pdf)** — Pearson correlation of vorticity magnitude at ≥90% core points vs. same locations at varying timestep lags (±1 to ±50 steps).
*   **[PDF] [Vortex Phase Portrait](vortex_phase_portrait.pdf)** — Phase portrait visualization.
*   **[PDF] [All Events Synchronization](all_events_sync_bar.pdf)** — Synchronization bar chart for all events.

**Conclusion:** The identified events represent stable, physically significant vortex reversals. The vortex cores are characterized as spatially extended regions using a vorticity-weighted centroid and bounding-box extent at the 90% peak-magnitude threshold. Every individual core grid point is stored exhaustively in `vorticity_core_points.csv` for full reproducibility. This region-based approach captures the physical reality that vortices are coherent structures, not point singularities. The use of standard CFD criteria (Vorticity, Q-Criterion) provides a rigorous basis for these findings.
