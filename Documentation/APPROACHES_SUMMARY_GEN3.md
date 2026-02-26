# 10 GEN3 Autoencoder Architectures for Fluid Dynamics

This document summarizes the "GEN3" series of autoencoder models used for reconstructing fluid dynamics velocity data (vx, vy, vz). These models represent a structural evolution, focusing on advanced architectural techniques like residual blocks, attention mechanisms, and bottleneck designs.

## GEN3 Summary Table

| # | Model | Basis / Paper | Rel. MSE | Key Feature | Best For |
|---|-------|---------------|----------|-------------|----------|
| <font color="red">1</font> | <font color="red">Baseline</font> | <font color="red">[ResNet (He et al., 2016)](https://doi.org/10.1109/CVPR.2016.90)</font> | <font color="red">1.334e-03</font> | <font color="red">Residual blocks (LayerNorm + ELU)</font> | <font color="red">Reliable, general reconstruction</font> |
| 2 | Deep | [ResNet (He et al., 2016)](https://doi.org/10.1109/CVPR.2016.90) | 3.190e-03 | More residual blocks per stage | Hierarchical pattern matching |
| 3 | Wide | [Wide ResNet (Zagoruyko, 2016)](https://doi.org/10.48550/arXiv.1605.07146) | 4.207e-03 | Increased width (512-256-128) | High-bandwidth feature extraction |
| 4 | GELU | [GELU (Hendrycks & Gimpel, 2016)](https://doi.org/10.48550/arXiv.1606.08415) | 1.887e-03 | GELU activation functions | Turbulent flow with stochasticity |
| <font color="red">5</font> | <font color="red">AttentionSE</font> | <font color="red">[SENet (Hu et al., 2018)](https://doi.org/10.1109/CVPR.2018.00745)</font> | <font color="red">1.359e-03*</font> | <font color="red">Squeeze-and-Excitation (SE) Gating</font> | <font color="red">Channel-wise feature weighting</font> |
| 6 | Dense | [DenseNet (Huang et al., 2017)](https://doi.org/10.1109/CVPR.2017.243) | 2.675e-03 | Concatenated feature flow | Fine-scale gradient preservation |
| 7 | BatchNorm | [BN (Ioffe & Szegedy, 2015)](https://proceedings.mlr.press/v37/ioffe15.pdf) | 4.290e-03 | Batch Normalization (BN) | Large-batch training stability |
| 8 | Skip/U-Net | [U-Net (Ronneberger et al., 2015)](https://doi.org/10.1007/978-3-319-24574-4_28) | 1.423e-03 | Encoder-to-Decoder Skip connections | Preserving high-res spatial data |
| 9 | Bottleneck | [ResNet Bottleneck (He et al., 2016)](https://doi.org/10.1109/CVPR.2016.90) | 2.576e-03 | Reduce-Transform-Expand blocks | Efficient, deep compression |
| 10 | Gated Attention | [Transformer (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) | 1.746e-03 | Self-attention gating (Middle) | Global spatial relationships |
| | | | | | |
| **Note:** | **Production Bake-off** | **Baseline vs AttentionSE** | **7.695e-04 (RMSE)*** | **Extended Training** | **AttentionSE Wins Long-term** |

\* *Longer production runs (1000 epochs) achieved a validation RMSE of 7.695e-04 for the AttentionSE model, outperforming the baseline in high-precision reconstruction.*

---

## Technical Documentation & MLA Citations

### 1-2. Residual Autoencoders (Baseline & Deep)
**Originally based on:** "Deep Residual Learning for Image Recognition" (He et al., 2016).
*   **MLA Citation:** He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770-778.
*   **PDF:** [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
*   **Deviations:** Adapts convolutional ResNet concepts to a fully connected (MLP) architecture. Uses LayerNorm and ELU within residual blocks.

### 3. Wide Residual Autoencoder
**Originally based on:** "Wide Residual Networks" (Zagoruyko & Komodakis, 2016).
*   **MLA Citation:** Zagoruyko, Sergey, and Nikos Komodakis. "Wide Residual Networks." *arXiv preprint arXiv:1605.07146*, 2016.
*   **PDF:** [https://arxiv.org/pdf/1605.07146.pdf](https://arxiv.org/pdf/1605.07146.pdf)
*   **Deviations:** Focuses on increasing the width of the hidden layers (up to 512 dimensions) rather than depth to capture broader feature distributions.

### 4. GELU Residual Autoencoder
**Originally based on:** "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016).
*   **MLA Citation:** Hendrycks, Dan, and Kevin Gimpel. "Gaussian Error Linear Units (GELUs)." *arXiv preprint arXiv:1606.08415*, 2016.
*   **PDF:** [https://arxiv.org/pdf/1606.08415.pdf](https://arxiv.org/pdf/1606.08415.pdf)
*   **Deviations:** Replaces standard ELU/ReLU activations with GELU, which weights inputs by their percentile, often performing better in turbulent flow modeling.

### 5. AttentionSE (Squeeze-and-Excitation)
**Originally based on:** "Squeeze-and-Excitation Networks" (Hu et al., 2017).
*   **MLA Citation:** Hu, Jie, et al. "Squeeze-and-Excitation Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2018, pp. 7132-7141.
*   **PDF:** [https://arxiv.org/pdf/1709.01507.pdf](https://arxiv.org/pdf/1709.01507.pdf)
*   **Deviations:** Uses SE-Blocks to perform channel-wise feature recalibration. In the fluid dynamics context, this allows the model to "attend" to specific velocity components (vx, vy, vz) or spatial regions more effectively.

### 6. Dense Autoencoder
**Originally based on:** "Densely Connected Convolutional Networks" (Huang et al., 2017).
*   **MLA Citation:** Huang, Gao, et al. "Densely Connected Convolutional Networks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2017, pp. 4700-4708.
*   **PDF:** [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)
*   **Deviations:** Implements dense connections within blocks using `torch.cat`. A projection layer is used after each dense block to maintain computational efficiency while allowing feature reuse.

### 7. BatchNorm Autoencoder
**Originally based on:** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015).
*   **MLA Citation:** Ioffe, Sergey, and Christian Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." *International Conference on Machine Learning*, 2015, pp. 448-456.
*   **PDF:** [https://proceedings.mlr.press/v37/ioffe15.pdf](https://proceedings.mlr.press/v37/ioffe15.pdf)
*   **Deviations:** Specifically swaps the project-standard LayerNorm for BatchNorm1d to test the effect of batch-wise statistics on turbulent flow data reconstruction.

### 8. Skip/U-Net Autoencoder
**Originally based on:** "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015).
*   **MLA Citation:** Ronneberger, Olaf, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *International Conference on Medical Image Computing and Computer-Assisted Intervention*, 2015, pp. 234-241.
*   **PDF:** [https://arxiv.org/pdf/1505.04597.pdf](https://arxiv.org/pdf/1505.04597.pdf)
*   **Deviations:** Implements long-range skip connections between the encoder and decoder. This allows the decoder to "bypass" the latent bottleneck for high-frequency details, which is reflected in the strong reconstruction performance (0.001423 MSE).

### 9. Bottleneck Residual AE
**Originally based on:** "Deep Residual Learning for Image Recognition" (He et al., 2016).
*   **MLA Citation:** He, Kaiming, et al. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016, pp. 770-778.
*   **PDF:** [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
*   **Deviations:** Uses the "Bottleneck" design (1x1 -> 3x3 -> 1x1 equivalent in MLP terms: reduce -> transform -> expand) to increase depth without proportionally increasing parameter count.

### 10. Gated Attention Autoencoder
**Originally based on:** "Attention Is All You Need" (Vaswani et al., 2017).
*   **MLA Citation:** Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017, pp. 5998-6008.
*   **PDF:** [https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf)
*   **Deviations:** Uses a simplified self-attention gating mechanism (query-key-value multiplication) as a middle layer to capture global relationships within the spatial data before final latent compression.

---

## Experimental Results Summary (MSE/RMSE)

Top Performers:
1.  <font color="red">**AttentionSE (Model 05):** 7.695e-04 RMSE (Longer Run) / 1.359e-03 MSE (Short Run)</font>
2.  <font color="red">**Baseline (Model 01):** 1.334e-03 MSE</font>
3.  **Skip/U-Net (Model 08):** 1.423e-03 MSE

Performance Observations:
-   **Baseline Residual AE** is extremely competitive in short-duration experiments, providing a reliable foundation for all GEN3 models.
-   **AttentionSE (Model 05) Bake-off:** While initially similar to the baseline, extended training runs (production mode) demonstrate that the Squeeze-and-Excitation blocks allow the model to converge to a significantly lower error floor (**7.695e-04 RMSE**).
-   **Data Variability Performance:** Original benchmarks were based on a 1% random sample of experiment, time, and coordinate space. As data variability increased (higher percentage of total data), the **AttentionSE (Model 05)** architecture outperformed the **Baseline** by continuing to exhibit continuous training and validation improvements where the baseline reached its performance ceiling.
-   **Skip Connections (Model 08)** excel at detail preservation but may "cheat" the latent bottleneck, which is reflected in the low MSE but should be verified for latent space utility.
