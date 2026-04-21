# Comprehensive Evaluation of Vortex Reversal Prediction (6p4 Dataset)

This document details the sensitivity analysis of the ***LFM*** model when predicting vortex reversal events for the `6p4` dataset. We explore the model's performance as a function of both temporal context (look-back window) and spatial density (number of coordinates provided).

## 1. Methodology: Spatio-Temporal Sensitivity Sweep

To determine the minimum data requirements for accurate prediction, we varied two primary parameters:
- **Temporal Context ($T$):** The number of timesteps provided as input (ranging from 1 to 60 steps).
- **Spatial Coordinates ($C$):** The number of X-coordinates provided per timestep (ranging from 5 to 26).

### A. Evaluation Metric: Z-Vorticity ($\omega_z$)
Following standard CFD practices for vortex identification, we compute the Z-component of vorticity ($\omega_z$). A reversal is defined by a sign-flip in this metric.
$$\omega_z = \nabla \times \vec{v} \cdot \hat{k} \approx \frac{\partial v_y}{\partial x} - \frac{\partial v_x}{\partial y}$$
In our 1D spatial line evaluation, we approximate this using the gradient $\frac{\partial v_y}{\partial x}$ across the available X-coordinates.

### B. Scientific Citations
Our identification and evaluation criteria are grounded in established literature:
- **Jeong, J., & Hussain, F. (1995).** *On the identification of a vortex.* Journal of Fluid Mechanics. (Used for $\omega_z$ and Q-criterion foundations).
- **Hunt, J. C. R., et al. (1988).** *Eddies, streams, and convergence zones in turbulent flows.* (Used for defining rotational dominance in fluid structures).

## 2. Key Findings: Data Volume vs. Prediction Accuracy

The evaluation reveals a critical threshold for prediction stability:
- **Spatial Requirement:** Providing fewer than 10 coordinates significantly degrades the gradient approximation, leading to "noisy" predictions. The model performs optimally with the full 26-coordinate spatial line.
- **Temporal Requirement:** A 100ms (12-step) context window is the "inflection point" where the transformer captures enough trajectory curvature to predict the sign-flip. With only 1-5 steps, the model tends to predict linear continuity (no reversal).
- **Deep Context:** Providing 40+ steps allows the model to capture the deceleration phase leading into the reversal, resulting in highly accurate zero-crossing timing.

## 3. Results Visualization

The following high-resolution documentation has been generated to support these findings:

- **[PDF] [Sparse Evaluation Comparison (1200 DPI)](sparse_evaluation_comparison.pdf)**: Shows side-by-side performance for (T=1, C=5), (T=12, C=26), and (T=40, C=26).
- **[PDF] [Context Sensitivity Heatmap (1200 DPI)](context_sensitivity_heatmap.pdf)**: Illustrates the RMSE of $\omega_z$ across the full sweep of time and space variations, averaged across all ≥90% vortex core (y, z) coordinates from 18 reversal events (typically 3–11 core grid points per event; see [core point collection methodology](locating_vortices_6p4.md#c-vortex-core-localization-region-based)). *(Generated via sparse sampling and cubic interpolation)*.
- **[PDF] [Direct Vortex Reversal Evaluation (1200 DPI)](step292_neg_to_pos_2yz_1200dpi.pdf)**: A representative publication-quality figure for Step 292 (Neg to Pos), isolating both vortex core $(y,z)$ coordinate pairs. The shaded bands show ±1 standard deviation across the two core locations — grey for ground truth, red for the ***LFM*** prediction — providing a visual confidence interval (CI) over the spatial variability within the event. This figure is intentionally representative: because individual per-event plots are numerous, summary statistics (RMSE, recovery time) are the primary vehicle for cross-event comparison and are reported in the [full evaluation table](transformer_6p4_reversal_evaluation.md#2-results-summary). The full per-event plot set is retained in [`vortex_reversal_evaluation.pdf`](vortex_reversal_evaluation.pdf) for completeness.
- **[PDF] [Core Temporal Autocorrelation](core_temporal_correlation.pdf)**: For each reversal event, shows the Pearson correlation of vorticity magnitude at the ≥90% core grid points vs. the same locations at varying timestep lags (±1 to ±50 steps). Quantifies how quickly the vortex core structure decorrelates over time.

## 4. Conclusion

***LFM*** demonstrates robust predictive capabilities for vortex reversals, provided it receives at least 100ms of spatio-temporal context. The model's reliance on "Spatio-Temporal Flattening" allows it to leverage neighboring spatial points to compensate for short temporal windows, a feature that aligns with the physically coupled nature of fluid velocity fields.

A notable observation from the heatmap is that spatial coordinate density has surprisingly little effect on reconstruction accuracy compared to temporal context. Our conjecture is that the transformer's **Spatio-Temporal Flattening** compensates: the model architecture flattens spatial and temporal inputs together, allowing it to implicitly interpolate spatial structure from neighboring points across time — it is not purely dependent on the spatial density at any single step. As a result, the spatial axis exhibits diminishing returns once sufficient temporal context is provided. The practical implication is that **time is the scarce resource, space is cheap**: if resources are constrained, investing in more timesteps yields far greater accuracy gains than increasing spatial sampling density.
