import torch
import torch.nn as nn
import torch.nn.functional as F

# From the original models
from model_CVAE_3D_01 import CVAE_3D, Flatten, UnFlatten


class CVAE_3D_Adapter(nn.Module):
    def __init__(self):
        super(CVAE_3D_Adapter, self).__init__()

        # Set constants to match WAE model
        self.original_dim = 375
        self.latent_dim = 47

        # Create the core CVAE model with modified latent dimension
        self.cvae = CVAE_3D(image_channels=3, h_dim=128, z_dim=self.latent_dim)

        # Add input and output adapters
        # This transforms flat 375-dim vector to 3D tensor format
        # Assuming the 375 values represent 125 points with xyz values
        # We'll reshape to a 5x5x5 grid with 3 channels
        self.input_adapter = nn.Linear(self.original_dim, 5 * 5 * 5 * 3)
        self.output_adapter = nn.Linear(5 * 5 * 5 * 3, self.original_dim)

    def forward(self, x):
        # Ensure input is treated as flat vector
        x = x.view(-1, self.original_dim)

        # Transform input to 3D format
        x_3d = self.input_adapter(x)
        x_3d = x_3d.view(-1, 3, 5, 5, 5)  # [batch, channels, D, H, W]

        # Process with original CVAE model
        recon_x_3d, mu, logvar, z_representation = self.cvae(x_3d)

        # Convert output back to flat format
        recon_x_flat = recon_x_3d.view(-1, 5 * 5 * 5 * 3)
        recon_x = self.output_adapter(recon_x_flat)

        return recon_x, mu, logvar, z_representation

    # We use the existing loss function from the CVAE model
    def loss_function(self, recon_x, x, mu, logvar):
        # First reshape data to match expected format
        # The WAE model's loss function expects a specific format
        return self.cvae.loss_function(recon_x, x, mu, logvar)