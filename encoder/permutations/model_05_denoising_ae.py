"""
Model 05: Denoising Autoencoder
Originally based on: "Extracting and Composing Robust Features with Denoising Autoencoders" (Vincent et al., 2008).

MLA Citations:
1. Vincent, Pascal, et al. "Extracting and Composing Robust Features with Denoising Autoencoders." ICML, 2008. https://www.cs.toronto.edu/~larochelle/publications/icml-2008-denoising-autoencoders.pdf
2. (Vincent et al. 1096-103)
3. Vincent et al., "Extracting and Composing Robust Features," ICML (2008).

Deviations from Paper:
- Uses additive Gaussian noise during training instead of masking noise (zeroing out features).
- Incorporates Dropout (0.2) as an additional regularization layer, which was not part of the original denoising autoencoder proposal.

Relative Performance (MSE): 5.510e-04
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class DenoisingAE(nn.Module):
    def __init__(self, dropout_rate=0.2, noise_factor=0.3):
        super(DenoisingAE, self).__init__()
        self.noise_factor = noise_factor

        hidden_dim1 = 250
        hidden_dim2 = 150
        hidden_dim3 = 100

        # Encoder
        self.fc1 = nn.Linear(original_dim, hidden_dim1)
        self.elu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_dim3, latent_dim)
        self.tanh = nn.Tanh()

        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.elu5 = nn.ELU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.elu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)

        self.fc8 = nn.Linear(hidden_dim1, original_dim)

    def add_noise(self, x):
        """Add Gaussian noise to input during training"""
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

    def encode(self, x):
        h1 = self.dropout1(self.elu1(self.fc1(x)))
        h2 = self.dropout2(self.elu2(self.fc2(h1)))
        h3 = self.dropout3(self.elu3(self.fc3(h2)))
        return self.tanh(self.fc4(h3))

    def decode(self, z):
        h1 = self.dropout4(self.elu4(self.fc5(z)))
        h2 = self.dropout5(self.elu5(self.fc6(h1)))
        h3 = self.dropout6(self.elu6(self.fc7(h2)))
        return self.fc8(h3)

    def forward(self, x):
        x = x.view(-1, original_dim)
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss (reconstruct clean from noisy)
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # Simple L2 regularization on latent
        latent_reg = torch.mean(z ** 2)

        total_loss = recon_loss + 0.0001 * latent_reg
        return total_loss, recon_loss, latent_reg, torch.tensor(0.0)
