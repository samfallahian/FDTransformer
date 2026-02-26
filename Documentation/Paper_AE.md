### Paper: Residual Autoencoder with Squeeze-and-Excitation (Model GEN3 05)

#### Architecture Overview
The core of this dimensionality reduction task is the `Model_GEN3_05_AttentionSE` architecture, a Residual Autoencoder (AE) that incorporates Squeeze-and-Excitation (SE) blocks for enhanced feature recalibration. The model is designed to compress high-dimensional fluid dynamics velocity fields into a compact latent representation while maintaining high reconstruction fidelity.

- **Algorithm**: Residual Autoencoder with Squeeze-and-Excitation (AttentionSE).
- **Compression Strategy**: Hierarchical dimensionality reduction using fully connected layers interspersed with SEResidualBlocks.
- **Input Dimension**: 375 floats (representing the flattened velocity field).
- **Latent Dimension**: 47 floats.
- **Compression Ratio**: 87.5% (approx. 1:8 compression).
- **Activation Functions**: 
    - **ELU (Exponential Linear Unit)** for intermediate layers to maintain gradient flow.
    - **Tanh** at the bottleneck to bound the latent space to [-1, 1].
    - **Linear** (no activation) at the output to allow for the full range of physical velocity values.

#### Key Architectural Components
1.  **Residual Connections**: Each stage of the encoder and decoder utilizes skip connections to facilitate gradient flow and allow the network to learn incremental refinements at multiple scales.
2.  **Squeeze-and-Excitation (SE) Blocks**: Integrated within the residual blocks, these components recalibrate feature importance by modeling inter-dependencies between "channels" (features). This allows the model to focus on the most energetically significant flow structures.
3.  **Layer Normalization**: Applied within each residual block to stabilize training and improve convergence.

#### Training and Data Sampling
The model was trained using a large-scale dataset sampled to ensure robustness across various flow conditions.

- **Data Sampling Strategy**: 
    - Training and validation datasets were built by sampling rows uniformly at random from a large pool of existing data files (stored as `.pkl.gz`).
    - The `EfficientDataLoader` was utilized to perform this sampling without being IO-bound, drawing a default of **2,000,000 rows** for the training set and **2,000,000 rows** for the validation set.
    - This approach ensures that both sets are representative of the global flow manifold.
- **Optimizer**: Adam optimizer with a learning rate of `1e-4`.
- **Batch Size**: 4096.
- **Epochs**: 1000.
- **Loss Function**:
    - **Primary Loss**: Mean Squared Error (MSE) between the input and reconstructed velocity fields.
    - **Regularization**: A very weak L2 regularization ($\lambda = 0.00005$) applied to the latent space ($\|z\|^2$) to prevent latent code explosion without sacrificing reconstruction accuracy.

#### Performance Metrics
The model achieved exceptional reconstruction accuracy, as evidenced by the Root Mean Squared Error (RMSE) across the validation sample.

- **Final Validation RMSE (Normalized)**: ~0.00077 (Global average across files).
- **Final Validation RMSE (Raw Units)**: ~0.00036 (Global average across files).
- **Peak Performance**: The "AttentionSE" model was identified as the winning architecture due to its superior ability to maintain performance as data variability increased, outperforming standard baseline residual models.
