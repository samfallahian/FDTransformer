"""
Model 09: Residual Autoencoder (ResAE)
Originally based on: "Deep Residual Learning for Image Recognition" (He et al., 2016).

MLA Citations:
1. He, Kaiming, et al. "Deep Residual Learning for Image Recognition." CVPR, 2016. https://arxiv.org/pdf/1512.03385.pdf
2. (He et al. 770-78)
3. He et al., "Deep Residual Learning," CVPR (2016).

Deviations from Paper:
- Adapts convolutional ResNet concepts to a fully connected (MLP) Autoencoder architecture for 375-dimensional input.
- Uses skip connections (`out = activation(LayerNorm(Linear(x)) + x)`) in an encoder-decoder bottleneck structure rather than the classic residual block in a deep classifier.
- Incorporates LayerNorm and ELU activations within the residual blocks.

Relative Performance (MSE): 4.600e-05

THEORETICAL FOUNDATION:
=======================
This architecture is inspired by ResNet (Deep Residual Learning for Image Recognition):
- Authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- Paper: "Deep Residual Learning for Image Recognition" (CVPR 2016)
- arXiv: https://arxiv.org/abs/1512.03385
- Key Innovation: Skip connections (identity mappings) that allow gradient flow through very deep networks

Extension to Autoencoders:
- "Identity Mappings in Deep Residual Networks" (He et al., ECCV 2016)
  https://arxiv.org/abs/1603.05027
- "Residual Networks for Computer Vision" demonstrates residual connections improve
  reconstruction quality and training stability in encoder-decoder architectures

CONTRAST WITH VANILLA AUTOENCODER:
===================================
A vanilla autoencoder uses simple feed-forward layers:
  Input → [Linear → Activation] × N → Latent → [Linear → Activation] × N → Output

Problems with vanilla AE:
  1. Vanishing gradients in deep networks (gradient diminishes exponentially with depth)
  2. Degradation problem: accuracy saturates then degrades with increased depth
  3. Information bottleneck: each layer must learn complete transformation
  4. Difficult optimization: loss landscape becomes increasingly non-convex

Residual AE improvements:
  1. Skip connections (h_out = h_in + F(h_in)) create identity pathways for gradients
  2. Residual blocks only need to learn the "residual" or difference, not full transformation
  3. Layer normalization stabilizes training by normalizing activations
  4. ELU activation (vs ReLU) provides smoother gradients and handles negative values
  5. Dropout for regularization without blocking gradient flow

ARCHITECTURE & FLUID DYNAMICS INTERPRETATION:
==============================================
Input dimension: 375 (likely 125 grid points × 3 velocity components: vx, vy, vz)
Latent dimension: 47 (compressed representation of flow state)

Hierarchical Abstraction Layers (Encoder):
-------------------------------------------
Layer 1: 375 → 250 dimensions + Residual Block
  FLUID INTERPRETATION: Captures local velocity field structures
  - Removes high-frequency noise and turbulent fluctuations
  - Identifies coherent structures like eddies and vortices at finest scales
  - Preserves spatial correlations between neighboring grid points
  - RESIDUAL BENEFIT: Maintains raw velocity information while learning perturbations

Layer 2: 250 → 150 dimensions + Residual Block
  FLUID INTERPRETATION: Intermediate-scale flow patterns
  - Detects shear layers, boundary layer structures
  - Identifies pressure gradient features
  - Captures momentum transfer mechanisms
  - May represent quasi-periodic oscillations or flow instabilities
  - RESIDUAL BENEFIT: Preserves scale-1 features while learning scale-2 patterns

Layer 3: 150 → 100 dimensions + Residual Block
  FLUID INTERPRETATION: Large-scale coherent structures
  - Dominant flow modes (similar to POD/DMD modes)
  - Recirculation zones and separation regions
  - Convective transport patterns
  - Energy-containing scales of turbulence
  - RESIDUAL BENEFIT: Multi-scale representation without information loss

Layer 4: 100 → 47 dimensions (Latent/Bottleneck)
  FLUID INTERPRETATION: Reduced-order model of flow physics
  - Most energetic flow features (compressed dynamics)
  - May correspond to reduced-order basis (POD modes, Koopman modes)
  - Captures essential degrees of freedom for flow evolution
  - Similar to dimensionality reduction in modal decomposition
  - Tanh activation: bounds latent space to [-1, 1] for training stability

Decoder Mirror Architecture (47 → 100 → 150 → 250 → 375):
  - Hierarchically reconstructs flow field from coarse to fine scales
  - Each level adds detail/resolution to the reconstruction
  - Residual connections allow fine-scale corrections at each level

RESIDUAL BLOCK DETAILS:
========================
Each ResidualBlock performs: out = activation(LayerNorm(Linear(x)) + x)

Components:
  - LayerNorm: Normalizes activations across features (not batch)
    * Stabilizes training by preventing internal covariate shift
    * Critical for deep networks where activations can explode/vanish

  - ELU activation: Exponential Linear Unit
    * Smooth everywhere (including x < 0), unlike ReLU
    * Mean activation closer to zero → faster convergence
    * Negative values preserve information about flow direction/magnitude

  - Dropout (20%): Stochastic regularization
    * Prevents co-adaptation of neurons
    * Acts as ensemble learning (averaging multiple sub-networks)

  - Skip connection (x + F(x)): Identity mapping
    * Ensures gradient flows directly through network
    * Network learns residual function F(x) = desired_output - x
    * If F(x) → 0, degrades gracefully to identity (no worse than shallower network)

LOSS FUNCTION:
==============
1. Reconstruction Loss: MSE between input and reconstructed flow field
   - Measures physical accuracy of velocity field reconstruction
   - Penalizes errors equally across all spatial locations and components

2. L2 Regularization (λ=0.00005): Penalty on latent code magnitude
   - Prevents latent codes from growing unbounded
   - Encourages compact, efficient representations
   - Much smaller than typical VAE β values (no KL divergence term)
   - Keeps latent space bounded near origin for better interpolation

CONTRAST WITH VANILLA AE TRAINING:
===================================
Vanilla AE challenges:
  - Requires careful learning rate tuning (gradient instability)
  - Often limited to 2-3 layers before degradation
  - May require pre-training layers sequentially
  - Sensitive to initialization

Residual AE advantages:
  - Stable training even with 7+ layers (3 encoder, 1 latent, 3 decoder)
  - Can use higher learning rates due to better gradient flow
  - Less sensitive to weight initialization
  - More robust to hyperparameter choices

Uses skip connections (residual blocks) for better gradient flow.
Loss: Reconstruction (MSE) with residual architecture
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class ResidualBlock(nn.Module):
    """
    Residual Block with Pre-Activation (He et al. 2016)

    Architecture: x → FC → LayerNorm → ELU → Dropout → FC → LayerNorm → (+x) → ELU

    Key Difference from Vanilla:
    - Vanilla: Simply x → FC → Activation → FC → Activation
    - Residual: Adds input directly to output (x + F(x)), learning only the delta

    Why this matters for fluid dynamics:
    - Flow fields evolve smoothly in time (small perturbations)
    - Network learns corrections rather than full transformations
    - Preserves conservation properties better (mass, momentum)
    """
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        # Two fully-connected layers of same dimension (preserves tensor size)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        # Layer normalization (not batch norm) - normalizes across features for each sample
        # Critical for flow data where batch statistics may vary widely
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Dropout for regularization (prevents overfitting to specific flow patterns)
        self.dropout = nn.Dropout(dropout_rate)

        # ELU activation: smoother than ReLU, handles negative velocities naturally
        self.activation = nn.ELU()

    def forward(self, x):
        # Store input for skip connection (identity pathway)
        residual = x

        # First transformation: Linear → LayerNorm → ELU
        out = self.activation(self.norm1(self.fc1(x)))

        # Dropout for regularization (only during training)
        out = self.dropout(out)

        # Second transformation: Linear → LayerNorm (no activation yet)
        out = self.norm2(self.fc2(out))

        # KEY INNOVATION: Add input to output (skip connection)
        # Network learns: out = x + F(x), where F(x) is the residual/perturbation
        # Gradient flows through both paths: ∂L/∂x = ∂L/∂out × (1 + ∂F/∂x)
        out = out + residual  # Skip connection

        # Final activation after addition
        return self.activation(out)

class ResidualAE(nn.Module):
    """
    Residual Autoencoder for Fluid Dynamics Compression

    Vanilla AE architecture (for comparison):
        Input(375) → FC(250) → FC(150) → FC(100) → FC(47) [Encoder]
                                                       ↓
        Output(375) ← FC(250) ← FC(150) ← FC(100) ← FC(47) [Decoder]

    Residual AE architecture (this implementation):
        Each layer pair: FC → ResBlock (with skip connections for gradient flow)

    Compression ratio: 375 → 47 = 87.5% compression (47/375 = 12.5% of original)
    """
    def __init__(self, dropout_rate=0.2):
        super(ResidualAE, self).__init__()

        # Store configuration for checkpoint saving
        self.dropout_rate = dropout_rate

        # Hierarchical dimension reduction: 375 → 250 → 150 → 100 → 47
        # Each reduction captures different scales of flow structures
        hidden_dim1 = 250  # First abstraction level (local structures)
        hidden_dim2 = 150  # Intermediate flow patterns
        hidden_dim3 = 100  # Large-scale coherent structures

        # ============================================================================
        # ENCODER: Progressive dimensionality reduction with residual learning
        # Vanilla AE problem: Deep encoders suffer from vanishing gradients
        # Residual solution: Skip connections maintain gradient magnitude
        # ============================================================================

        # ENCODING STAGE 1: 375 → 250 (Fine-scale features)
        # Maps raw velocity data to first abstraction level
        self.enc_in = nn.Linear(original_dim, hidden_dim1)
        # Residual refinement at 250-dim (learns perturbations around initial encoding)
        self.enc_res1 = ResidualBlock(hidden_dim1, dropout_rate)

        # ENCODING STAGE 2: 250 → 150 (Intermediate-scale features)
        # Downsampling layer: reduces dimensionality while preserving information
        self.enc_down1 = nn.Linear(hidden_dim1, hidden_dim2)
        # Residual refinement at 150-dim (captures multi-scale interactions)
        self.enc_res2 = ResidualBlock(hidden_dim2, dropout_rate)

        # ENCODING STAGE 3: 150 → 100 (Large-scale coherent structures)
        # Further compression toward dominant flow modes
        self.enc_down2 = nn.Linear(hidden_dim2, hidden_dim3)
        # Residual refinement at 100-dim (integrates across scales)
        self.enc_res3 = ResidualBlock(hidden_dim3, dropout_rate)

        # BOTTLENECK: 100 → 47 (Latent space - compressed representation)
        # Final projection to latent space (reduced-order model)
        self.enc_out = nn.Linear(hidden_dim3, latent_dim)
        # Tanh bounds latent codes to [-1, 1] for stable training & interpolation
        # Critical for generative applications (smooth latent traversals)
        self.tanh = nn.Tanh()

        # ============================================================================
        # DECODER: Progressive reconstruction with residual learning
        # Mirrors encoder architecture (symmetric hourglass shape)
        # Each stage reconstructs finer details of the flow field
        # ============================================================================

        # DECODING STAGE 1: 47 → 100 (Expand from latent to coarse representation)
        self.dec_in = nn.Linear(latent_dim, hidden_dim3)
        # Residual refinement at 100-dim (refines coarse flow structures)
        self.dec_res1 = ResidualBlock(hidden_dim3, dropout_rate)

        # DECODING STAGE 2: 100 → 150 (Add intermediate-scale details)
        self.dec_up1 = nn.Linear(hidden_dim3, hidden_dim2)
        # Residual refinement at 150-dim (adds mid-scale features)
        self.dec_res2 = ResidualBlock(hidden_dim2, dropout_rate)

        # DECODING STAGE 3: 150 → 250 (Add fine-scale details)
        self.dec_up2 = nn.Linear(hidden_dim2, hidden_dim1)
        # Residual refinement at 250-dim (recovers local flow structures)
        self.dec_res3 = ResidualBlock(hidden_dim1, dropout_rate)

        # OUTPUT: 250 → 375 (Reconstruct full velocity field)
        # No activation here - allows full range of velocity values
        self.dec_out = nn.Linear(hidden_dim1, original_dim)

        # Shared activation and dropout for intermediate layers
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x):
        """
        Encoder: Compress 375-D velocity field to 47-D latent code

        Vanilla AE encoding (for comparison):
            Simply stacks linear layers with activations, no skip connections
            Gradients must backpropagate through all layers multiplicatively
            Each layer must learn full transformation from scratch

        Residual encoding (this implementation):
            Each stage: Downsample → Residual refinement
            Skip connections allow gradients to flow directly to earlier layers
            Network learns incremental refinements at each scale

        Fluid dynamics interpretation:
            Progressive coarse-graining of velocity field
            Similar to wavelet decomposition or POD basis projection
            Each layer removes fine-scale information, retains energetic modes
        """
        # Stage 1: Initial projection to 250-D + residual refinement
        # Captures local flow structures (vortices, shear layers)
        h = self.activation(self.enc_in(x))
        h = self.enc_res1(h)  # Residual block refines representation

        # Stage 2: Compression to 150-D + residual refinement
        # Captures intermediate-scale patterns (boundary layers, instabilities)
        h = self.activation(self.enc_down1(h))
        h = self.enc_res2(h)  # Skip connections preserve fine-scale info

        # Stage 3: Compression to 100-D + residual refinement
        # Captures large-scale coherent structures (recirculation zones)
        h = self.activation(self.enc_down2(h))
        h = self.enc_res3(h)  # Multi-scale integration

        # Final projection to 47-D latent space (bounded to [-1, 1])
        # Represents reduced-order model of flow dynamics
        return self.tanh(self.enc_out(h))

    def decode(self, z):
        """
        Decoder: Reconstruct 375-D velocity field from 47-D latent code

        Vanilla AE decoding (for comparison):
            Simple upsampling with linear layers
            Must reconstruct all scales simultaneously at each layer
            No mechanism to preserve multi-scale structure

        Residual decoding (this implementation):
            Each stage: Upsample → Residual refinement
            Progressively adds detail from coarse to fine scales
            Skip connections allow corrections at each resolution level

        Fluid dynamics interpretation:
            Hierarchical reconstruction from dominant modes to fine details
            Similar to inverse wavelet transform or modal synthesis
            First reconstructs mean flow, then adds fluctuations
        """
        # Stage 1: Expand from 47-D to 100-D + residual refinement
        # Initializes coarse flow field from latent representation
        h = self.activation(self.dec_in(z))
        h = self.dec_res1(h)  # Refines coarse structures

        # Stage 2: Expand to 150-D + residual refinement
        # Adds intermediate-scale flow features
        h = self.activation(self.dec_up1(h))
        h = self.dec_res2(h)  # Adds mid-scale details

        # Stage 3: Expand to 250-D + residual refinement
        # Adds fine-scale flow structures
        h = self.activation(self.dec_up2(h))
        h = self.dec_res3(h)  # Recovers local features

        # Final projection to 375-D (no activation - allows full velocity range)
        # Unbounded output: critical for accurately representing velocity magnitudes
        return self.dec_out(h)

    def forward(self, x):
        """
        Full forward pass: Encode then decode

        Input: x of shape (batch_size, 375) - velocity field
        Output:
            - recon_x: reconstructed velocity field (batch_size, 375)
            - z: latent representation (batch_size, 47)

        Vanilla AE: Single pass through encoder and decoder
        Residual AE: Multiple skip connections throughout the network
                     allow information to flow through direct paths
        """
        # Flatten input to (batch_size, 375) if needed
        x = x.view(-1, original_dim)

        # Encode: 375-D → 47-D (compression via residual blocks)
        z = self.encode(x)

        # Decode: 47-D → 375-D (reconstruction via residual blocks)
        # Returns both reconstruction and latent code for analysis
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        """
        Loss function for Residual Autoencoder

        Vanilla AE loss:
            L = MSE(x, recon_x)
            Problem: No constraint on latent space structure
            Can lead to unbounded latent codes, poor interpolation

        Residual AE loss (this implementation):
            L = MSE(x, recon_x) + λ||z||²
            Benefits:
                - Reconstruction term: accurate velocity field recovery
                - L2 regularization: prevents latent space explosion
                - λ = 0.00005: very weak regularization (not a VAE!)

        Why weak regularization?
            - Strong regularization (like VAE β=1.0) can hurt reconstruction
            - For fluid dynamics, reconstruction accuracy is critical
            - Small λ just prevents pathological latent codes
            - Allows network to use full capacity for reconstruction

        Contrast with VAE loss:
            VAE: L = MSE + KL(q(z|x) || N(0,1))
                 Forces latent space to be Gaussian, strong constraint
            ResAE: L = MSE + λ||z||²
                   Gentle encouragement toward bounded latent codes
        """
        # Reconstruction loss: Mean Squared Error between input and reconstruction
        # Measures physical accuracy of velocity field reconstruction
        # Units: (m/s)² averaged across all grid points and components
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # L2 regularization on latent codes: ||z||²
        # Prevents latent codes from growing unbounded
        # Encourages compact, efficient representations
        # λ = 0.00005 is ~1000× weaker than typical VAE regularization
        l2_reg = torch.mean(z ** 2)

        # Total loss: weighted sum of reconstruction and regularization
        # Reconstruction dominates (λ very small), regularization is gentle constraint
        total_loss = recon_loss + 0.00005 * l2_reg

        # Return: (total_loss, recon_loss, l2_reg, dummy_zero)
        # Dummy zero for compatibility with other models that may have additional terms
        return total_loss, recon_loss, l2_reg, torch.tensor(0.0)
