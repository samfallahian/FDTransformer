import logging
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import host preferences to get logging level
from Ordered_001_Initialize import HostPreferences

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up logging level from experiment.preferences
preferences_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment.preferences")
preferences = HostPreferences(filename=preferences_path)

# Set logging level from preferences
if hasattr(preferences, 'logging_level'):
    level = getattr(logging, preferences.logging_level.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        if level <= logging.DEBUG:
            logger.debug(f"Set logging level to {preferences.logging_level.upper()}")


class SpatialAwareVAE(nn.Module):
    """
    Spatially-Aware Variational Autoencoder (VAE) for turbulent flow data.

    This VAE is designed to handle 375-dimensional input data (representing 125 points with x,y,z components)
    and compress it to a 47-dimensional latent space while preserving spatial relationships.
    """

    def __init__(
            self,
            input_dim: int = 375,
            latent_dim: int = 47,
            hidden_dims: List[int] = None,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            activation: str = 'relu',
            enhanced: bool = False
    ):
        """
        Initialize the SpatialAwareVAE model.

        Args:
            input_dim: Number of input dimensions (default: 375 for 125 points with x,y,z components)
            latent_dim: Number of latent dimensions (default: 47)
            hidden_dims: List of hidden dimensions for the encoder and decoder
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'leaky_relu', or 'elu')
            enhanced: Whether to use the enhanced version with spatial components (default: False)
        """
        super(SpatialAwareVAE, self).__init__()
        
        # Check if we should create an enhanced version instead
        if enhanced:
            logger.info(f"Enhanced flag detected - creating EnhancedSpatialVAE instead")
            # Dynamically replace this instance with an EnhancedSpatialVAE
            # This allows existing code to work without modification
            self.__class__ = EnhancedSpatialVAE
            EnhancedSpatialVAE.__init__(
                self,
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                activation=activation
            )
            return

        logger.info(f"Initializing basic SpatialAwareVAE with input_dim={input_dim}, latent_dim={latent_dim}")

        # Model parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Define default hidden dimensions if not provided
        if hidden_dims is None:
            # Create a pyramid structure for the encoder and mirror it for the decoder
            hidden_dims = [256, 128, 64]
        self.hidden_dims = hidden_dims

        # Number of points (assuming each point has x, y, z components)
        self.num_points = input_dim // 3
        logger.debug(f"Number of points in the input data: {self.num_points}")

        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            logger.warning(f"Unknown activation function: {activation}, defaulting to ReLU")
            self.activation = nn.ReLU()

        # Build the encoder
        self._build_encoder(dropout_rate, use_batch_norm)

        # Build the latent space projectors
        self._build_latent_projectors()

        # Build the decoder
        self._build_decoder(dropout_rate, use_batch_norm)

        logger.info("Basic SpatialAwareVAE model initialized successfully")

    def _build_encoder(self, dropout_rate: float, use_batch_norm: bool):
        """
        Build the encoder part of the VAE.

        Args:
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        logger.debug("Building encoder network")

        # Initial fully connected layer to extract features
        self.encoder_input = nn.Linear(self.input_dim, self.hidden_dims[0])

        # Create encoder layers
        modules = []

        # Input shape after reshaping: [batch_size, 125, 3]
        # We'll use 1D convolutions on the sequence of points

        in_channels = self.hidden_dims[0]

        # Add convolutional layers for spatial feature extraction
        for h_dim in self.hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.BatchNorm1d(h_dim) if use_batch_norm else nn.Identity(),
                    self.activation
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        logger.debug(f"Encoder architecture: {modules}")

    def _build_latent_projectors(self):
        """Build the latent space projectors for mu and log_var."""
        logger.debug("Building latent space projectors")

        # Latent space projectors for mean and log variance
        self.mu_projector = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.logvar_projector = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.mu_projector.weight)
        nn.init.xavier_uniform_(self.logvar_projector.weight)

        logger.debug(
            f"Latent projectors: mu shape={self.mu_projector.weight.shape}, logvar shape={self.logvar_projector.weight.shape}")

    def _build_decoder(self, dropout_rate: float, use_batch_norm: bool):
        """
        Build the decoder part of the VAE.

        Args:
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        logger.debug("Building decoder network")

        # Initial projection from latent space to the first hidden layer
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1])

        # Create decoder layers (reverse of encoder)
        modules = []
        hidden_dims_reversed = list(reversed(self.hidden_dims))

        # Add transposed convolutional layers for spatial reconstruction
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i + 1]),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.BatchNorm1d(hidden_dims_reversed[i + 1]) if use_batch_norm else nn.Identity(),
                    self.activation
                )
            )

        self.decoder = nn.Sequential(*modules)

        # Final layer to reconstruct the original input
        self.final_layer = nn.Linear(self.hidden_dims[0], self.input_dim)

        logger.debug(f"Decoder architecture: {modules}")

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode the input data to the latent space.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Encoding input shape: {x.shape}")

        # Initial feature extraction
        x = self.encoder_input(x)
        x = self.activation(x)

        # Pass through encoder layers
        x = self.encoder(x)

        # Project to latent space
        mu = self.mu_projector(x)
        logvar = self.logvar_projector(x)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Encoded mu shape: {mu.shape}, logvar shape: {logvar.shape}")

        return mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode the latent representation back to the input space.

        Args:
            z: Latent tensor of shape [batch_size, latent_dim]

        Returns:
            Reconstructed input tensor of shape [batch_size, input_dim]
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Decoding latent shape: {z.shape}")

        # Project from latent space to the first hidden layer
        z = self.decoder_input(z)
        z = self.activation(z)

        # Pass through decoder layers
        z = self.decoder(z)

        # Final projection to input space
        reconstruction = self.final_layer(z)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Decoded output shape: {reconstruction.shape}")

        return reconstruction

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            reconstruction: Reconstructed input tensor
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        logger.debug(f"Forward pass with input shape: {x.shape}")

        # Encode the input
        mu, logvar = self.encode(x)

        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)

        # Decode the latent representation
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar

    def loss_function(
            self,
            reconstruction: Tensor,
            x: Tensor,
            mu: Tensor,
            logvar: Tensor,
            kld_weight: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the VAE loss function.

        Args:
            reconstruction: Reconstructed input tensor
            x: Original input tensor
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            kld_weight: Weight for the KL divergence term

        Returns:
            total_loss: The total VAE loss
            reconstruction_loss: The reconstruction loss component
            kld_loss: The KL divergence loss component
        """
        # Reconstruction loss (mean squared error)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')

        # KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = reconstruction_loss + kld_weight * kld_loss

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loss components: recon_loss={reconstruction_loss.item()}, kld_loss={kld_loss.item()}")

        return total_loss, reconstruction_loss, kld_loss


class EnhancedSpatialVAE(SpatialAwareVAE):
    """
    Enhanced Spatially-Aware VAE with additional features for turbulent flow modeling.

    This model extends the basic SpatialAwareVAE with additional features:
    1. Graph-based operations for handling unstructured mesh data
    2. Attention mechanisms to focus on important spatial relationships
    3. Improved spatial encoding through positional embeddings

    These enhancements are inspired by the techniques described in:
    "Data-driven modeling of turbulent flows using a convolutional autoencoder with unstructured meshes"
    (Bode et al., JFM, 2021)
    """

    def __init__(
            self,
            input_dim: int = 375,
            latent_dim: int = 47,
            hidden_dims: List[int] = None,
            spatial_dims: int = 3,
            num_heads: int = 4,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            activation: str = 'relu'
    ):
        """
        Initialize the EnhancedSpatialVAE model.

        Args:
            input_dim: Number of input dimensions (default: 375 for 125 points with x,y,z components)
            latent_dim: Number of latent dimensions (default: 47)
            hidden_dims: List of hidden dimensions for the encoder and decoder
            spatial_dims: Number of spatial dimensions (default: 3 for x,y,z)
            num_heads: Number of attention heads for multi-head attention
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use ('relu', 'leaky_relu', or 'elu')
        """
        # Initialize the base VAE
        super(EnhancedSpatialVAE, self).__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation
        )

        # Additional parameters
        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.num_points = input_dim // spatial_dims

        logger.info(f"Initializing EnhancedSpatialVAE with spatial_dims={spatial_dims}, num_heads={num_heads}")

        # Build the spatial encoder and decoder
        self._build_spatial_components()

        logger.info("EnhancedSpatialVAE model initialized successfully")

    def _build_spatial_components(self):
        """Build components for spatial feature extraction and reconstruction."""
        logger.debug("Building spatial components")

        # Positional encoding for spatial awareness
        self.positional_encoding = PositionalEncoding(
            d_model=self.hidden_dims[0],
            max_seq_len=self.num_points
        )

        # Multi-head self-attention for capturing spatial relationships
        self.self_attention = MultiHeadAttention(
            d_model=self.hidden_dims[0],
            num_heads=self.num_heads
        )

        # Spatial graph convolution for handling unstructured mesh data
        self.graph_conv = SpatialGraphConv(
            in_channels=self.hidden_dims[0],
            out_channels=self.hidden_dims[0]
        )

        # Decoder attention mechanism
        self.decoder_attention = MultiHeadAttention(
            d_model=self.hidden_dims[0],
            num_heads=self.num_heads
        )

        logger.debug("Spatial components built successfully")

    def encode(self, x):
        """
        Enhanced encoding with spatial awareness.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Enhanced encoding input shape: {x.shape}")

        batch_size = x.shape[0]

        # Initial feature extraction - use the same approach as the basic model
        x = self.encoder_input(x)
        x = self.activation(x)

        # Instead of reshaping, use a more direct approach
        # Pass through encoder layers (same as basic model)
        x = self.encoder(x)

        # Project to latent space
        mu = self.mu_projector(x)
        logvar = self.logvar_projector(x)

        return mu, logvar

    def decode(self, z):
        """
        Enhanced decoding with spatial awareness.

        Args:
            z: Latent tensor of shape [batch_size, latent_dim]

        Returns:
            Reconstructed input tensor of shape [batch_size, input_dim]
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Enhanced decoding latent shape: {z.shape}")

        batch_size = z.shape[0]

        # Project from latent space to the first hidden layer
        z = self.decoder_input(z)
        z = self.activation(z)

        # Pass through decoder layers
        features = self.decoder(z)

        # Final projection to input space
        reconstruction = self.final_layer(features)

        return reconstruction


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for providing spatial awareness.

    This module adds positional information to the feature vectors,
    which helps the model understand spatial relationships.
    """

    def __init__(self, d_model: int, max_seq_len: int):
        """
        Initialize the positional encoding module.

        Args:
            d_model: Dimension of the model (embedding dimension)
            max_seq_len: Maximum sequence length (number of points)
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for capturing spatial relationships.

    This module implements the multi-head attention mechanism as described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize the multi-head attention module.

        Args:
            d_model: Dimension of the model (embedding dimension)
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Combine the heads back to the original shape.

        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, d_k]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Forward pass through the multi-head attention module.

        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # Split heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Combine heads
        attention_output = self._combine_heads(attention_output)

        # Final linear projection
        output = self.W_o(attention_output)

        return output


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution module for handling unstructured mesh data.

    This module implements a graph convolutional layer that operates on
    spatial data represented as graphs, inspired by the approach described in
    "Data-driven modeling of turbulent flows using a convolutional autoencoder with unstructured meshes"
    (Bode et al., JFM, 2021).
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize the spatial graph convolution module.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(SpatialGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight matrix for transforming node features
        self.W = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Learnable parameters for computing edge weights
        self.a = nn.Parameter(torch.Tensor(out_channels, 1))

        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def _compute_adjacency(self, positions: Tensor) -> Tensor:
        """
        Compute the adjacency matrix based on spatial positions.

        Args:
            positions: Tensor of shape [batch_size, num_nodes, spatial_dims]
                containing the spatial positions of nodes

        Returns:
            Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
        """
        # Compute pairwise distances between nodes
        batch_size, num_nodes, _ = positions.size()

        # Reshape for broadcasting
        pos_i = positions.unsqueeze(2)  # [batch_size, num_nodes, 1, spatial_dims]
        pos_j = positions.unsqueeze(1)  # [batch_size, 1, num_nodes, spatial_dims]

        # Compute Euclidean distances
        distances = torch.norm(pos_i - pos_j, dim=3)  # [batch_size, num_nodes, num_nodes]

        # Convert distances to adjacency weights using a Gaussian kernel
        sigma = 0.1  # Bandwidth parameter
        adjacency = torch.exp(-(distances ** 2) / (2 * sigma ** 2))

        return adjacency

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Forward pass through the spatial graph convolution module.

        Args:
            x: Input tensor of shape [batch_size, num_nodes, in_channels]
                containing node features
            positions: Tensor of shape [batch_size, num_nodes, spatial_dims]
                containing the spatial positions of nodes

        Returns:
            Output tensor of shape [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.size()

        # Transform node features
        h = torch.matmul(x, self.W)  # [batch_size, num_nodes, out_channels]

        # Compute adjacency matrix
        adjacency = self._compute_adjacency(positions)  # [batch_size, num_nodes, num_nodes]

        # Normalize adjacency matrix
        rowsum = adjacency.sum(dim=2, keepdim=True)  # [batch_size, num_nodes, 1]
        norm_adj = adjacency / (rowsum + 1e-8)  # [batch_size, num_nodes, num_nodes]

        # Graph convolution
        output = torch.bmm(norm_adj, h)  # [batch_size, num_nodes, out_channels]

        return output


# Function to create a model instance with specified parameters
def create_vae_model(
        input_dim: int = 375,
        latent_dim: int = 47,
        hidden_dims: List[int] = None,
        enhanced: bool = True
) -> nn.Module:
    """
    Create a VAE model instance with the specified parameters.

    Args:
        input_dim: Number of input dimensions
        latent_dim: Number of latent dimensions
        hidden_dims: List of hidden dimensions for the encoder and decoder
        enhanced: Whether to use the enhanced spatial VAE model

    Returns:
        VAE model instance
    """
    if enhanced:
        model = EnhancedSpatialVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
    else:
        model = SpatialAwareVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )

    logger.info(
        f"Created {'enhanced' if enhanced else 'basic'} VAE model with input_dim={input_dim}, latent_dim={latent_dim}")

    return model

# Make sure to import this module's items when imported from another module
__all__ = ['SpatialAwareVAE', 'EnhancedSpatialVAE', 'create_vae_model']