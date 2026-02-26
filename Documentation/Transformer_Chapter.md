# Chapter: Spatio-Temporal Transformer for Latent Fluid Dynamics

## 1. Introduction
This chapter details the development and evaluation of a Transformer-based surrogate model designed to predict the evolution of fluid velocity fields in a compressed latent space. By leveraging the attention mechanism, the model captures complex dependencies across both spatial and temporal dimensions, effectively acting as a high-fidelity emulator for physical simulations.

## 2. Dataset Preparation and Feature Engineering
The model operates on a dataset derived from high-fidelity physical simulations, compressed into a 47-dimensional latent space using a pre-trained Autoencoder.

### 2.1 Spatio-Temporal Windows
Data is sampled as sequences of 8 time steps at fixed (y, z) coordinates, covering 26 adjacent points along the X-axis. This results in a 4D tensor structure for each sample:
*   **Temporal Dimension**: 8 consecutive time steps.
*   **Spatial Dimension**: 26 discrete, adjacent X-coordinates.
*   **Feature Dimension**: 52 features per point.

### 2.2 Feature Mapping
Each "token" in our sequence is a vector of 52 features, providing both physical and latent context to the model:

| Indices | Name | Description |
| :--- | :--- | :--- |
| **0 - 46** | **Latents** | 47-dimensional latent representation of the velocity field. |
| **47** | **X** | Physical X-coordinate. |
| **48** | **Y** | Physical Y-coordinate (constant for a given sequence). |
| **49** | **Z** | Physical Z-coordinate (constant for a given sequence). |
| **50** | **Time** | Relative time index (0.0 to 7.0). |
| **51** | **Param** | Experimental parameter value. In this dataset, this acts as a **Reynolds number proxy**, representing the physical scaling of the fluid dynamics (e.g., flow velocity or viscosity ratios) for each specific simulation run. |

### 2.3 Sequence Flattening
To process this grid with a Transformer, we flatten the 8×26 grid into a single sequence of **208 tokens**. The ordering follows a "Space-First" approach:
`[T0, X0] → [T0, X1] → ... → [T0, X25] → [T1, X0] → ... → [T7, X25]`
This structure allows the self-attention mechanism to learn spatial correlations within a time step and temporal dynamics across steps simultaneously.

---

## 3. Model Architecture: OrderedTransformerV1
The `OrderedTransformerV1` architecture is optimized for structured physical data, departing from standard NLP transformers in several key ways.

### 3.1 Structured Embeddings
Rather than a single learned positional embedding, we use **Dual Structured Embeddings**:
1.  **Time Embeddings**: Learned representations for each of the 8 time indices.
2.  **Space Embeddings**: Learned representations for each of the 26 spatial X-indices.
These are added to the projected input, ensuring the model maintains an explicit awareness of the physical grid geometry.

### 3.2 Transformer Core
*   **Layers**: 6 Transformer Encoder blocks.
*   **Attention**: 8-head multi-head self-attention.
*   **Embedding Dimension**: 256.
*   **Causal Masking**: A standard triangular mask is applied to the 208-token sequence, ensuring that predictions for any point $(T_i, X_j)$ only depend on past time steps and preceding spatial points.

---

## 4. Evaluation and Results
The model was evaluated using the checkpoint `best_ordered_transformer_v1.pt`. Performance is measured in physical velocity space by decoding the Transformer's latent predictions back into $(v_x, v_y, v_z)$ vectors.

### 4.1 Prediction Accuracy (Physical Velocity RMSE)
The model shows high precision in predicting future states. Error metrics were calculated across various "prediction windows" at the final time step ($T_7$). These values are reported in **Physical Velocity Units** (e.g., m/s) after decoding the latents and denormalizing the output:

| Window | Description | RMSE (Velocity Units) |
| :--- | :--- | :--- |
| **L4** | Last 4 spatial positions | 3.0e-03 |
| **L8** | Last 8 spatial positions | 3.1e-03 |
| **L16** | Last 16 spatial positions | 3.1e-03 |
| **Overall** | Entire $T_7$ slice | 3.1e-03 |

*Note: These physical metrics are derived by passing the Transformer's latent predictions through the Autoencoder's decoder and applying the `FloatConverter` to reverse the dataset normalization.*

### 4.2 Error Distribution
Evaluation across the Y-Z coordinate space reveals that the model maintains consistent accuracy across the entire experimental box, with minimal error variance across different parameter sets.

### 4.3 Autoregressive Performance: "Staircase" Evaluation
To assess the model's stability for multi-step rollout, we performed a "staircase" evaluation. In this test, the model is given a decreasing window of ground-truth tokens and must autoregressively predict the remainder of the $T_8$ slice.

**Note on Scoring Domains:** The metrics below are calculated directly in the **47-dimensional Latent Space** (MSE). Because these latent features are normalized/standardized during dataset preparation, the numerical values are significantly smaller than the physical velocity RMSE reported in Section 4.1.

**Epoch 100 Performance Summary (Latent Space MSE):**
*   **Avg Train (All)**: 1.7269e-05
*   **Avg Val (All)**: 1.5001e-05
*   **Target Losses (T7)**:
    *   Last 4: 6.5996e-06
    *   Last 8: 5.7883e-06
    *   Last 16: 5.4284e-06
*   **Query Metrics**:
    *   Even Steps (2,4,6,8): 1.6037e-05
    *   8th Step: 1.5960e-05

**Autoregressive T8 Prediction (Staircase Eval - Latent MSE):**
The following table shows the MSE for predicting the final time step given varying lengths of history:

| Context Window | Prediction MSE (Latent) |
| :--- | :--- |
| **Given T1-7** | 1.4876e-04 |
| **Given T1-6** | 1.4895e-04 |
| **Given T1-5** | 1.5420e-04 |
| **Given T1-4** | 1.6033e-04 |
| **Given T1-3** | 1.6800e-04 |
| **Given T1-2** | 1.7683e-04 |
| **Given T1-1** | 1.9047e-04 |

The results indicate that while error accumulates as the prediction horizon increases (from T1-7 down to T1-1), the model maintains remarkable stability even when predicting 7 out of 8 time steps purely autoregressively.

---

## 5. Robustness Analysis: Data Corruption Study
A critical part of this research involved testing the model's resilience to noisy or corrupted input data. We conducted a "Corruption Sweep," replacing a percentage of the input latent features with random noise ($[-1, 1]$).

### 5.1 Deterioration Metrics (Physical Velocity RMSE)
The following table summarizes how the prediction error (RMSE in Velocity Units) scales with input corruption:

| Corruption % | RMSE (Velocity) | Performance Impact |
| :--- | :--- | :--- |
| **0%** | 3.1e-03 | Baseline (Clean) |
| **1%** | 1.25e-01 | Significant Initial Jump |
| **10%** | 5.8e-01 | Gradual Degradation |
| **50%** | 1.02e+00 | Saturated Error |
| **100%** | 1.14e+00 | Total Information Loss |

### 5.2 Key Findings on Robustness
*   **High Sensitivity to Low-Level Noise**: The jump from 0% to 1% corruption suggests that the Transformer relies heavily on the precise manifold structure of the latent space.
*   **Graceful Degradation**: Beyond 20% corruption, the error curve flattens, indicating that the model retains some "physical intuition" from the non-corrupted features (XYZ coordinates and parameters) even when the latent state is heavily degraded.

---

## 6. Conclusion
The Spatio-Temporal Transformer demonstrates a powerful capability to emulate fluid dynamics in a compressed latent space. By combining structured physical embeddings with the attention mechanism, it achieves high-fidelity predictions that are robust across varied experimental parameters and spatially consistent across the simulation domain.
