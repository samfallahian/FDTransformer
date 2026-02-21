"""
Model 10: Mixture Density Autoencoder
Originally based on: "Mixture Density Networks" (Bishop, 1994).

MLA Citations:
1. Bishop, Christopher M. "Mixture Density Networks." Neural Computing Research Group Report, 1994. https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
2. (Bishop 1-26)
3. Bishop, "Mixture Density Networks," NCRG (1994).

Deviations from Paper:
- Integrated MDN as an Autoencoder's decoder for predicting the target distribution parameters (pi, mu, sigma).
- Combines Negative Log-Likelihood (NLL) with Mean Squared Error (MSE) as an auxiliary loss for reconstruction.
- Uses modern ELU/ReLU activations and Dropout (0.2).

Relative Performance (MSE): 3.260e-04
"""
import torch
from torch import nn
import torch.nn.functional as F
import math

original_dim = 375
latent_dim = 47
num_mixtures = 3  # Number of Gaussian components

class MixtureDensityAE(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(MixtureDensityAE, self).__init__()
        self.num_mixtures = num_mixtures

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

        # Decoder - outputs mixture parameters
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout(dropout_rate)

        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.elu5 = nn.ELU()
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.elu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)

        # Output heads for mixture parameters
        # For each output dimension, predict: pi (mixture weights), mu (means), sigma (stds)
        self.fc_pi = nn.Linear(hidden_dim1, original_dim * num_mixtures)
        self.fc_mu = nn.Linear(hidden_dim1, original_dim * num_mixtures)
        self.fc_sigma = nn.Linear(hidden_dim1, original_dim * num_mixtures)

    def encode(self, x):
        h1 = self.dropout1(self.elu1(self.fc1(x)))
        h2 = self.dropout2(self.elu2(self.fc2(h1)))
        h3 = self.dropout3(self.elu3(self.fc3(h2)))
        return self.tanh(self.fc4(h3))

    def decode(self, z):
        h1 = self.dropout4(self.elu4(self.fc5(z)))
        h2 = self.dropout5(self.elu5(self.fc6(h1)))
        h3 = self.dropout6(self.elu6(self.fc7(h2)))

        # Get mixture parameters
        pi = self.fc_pi(h3)
        mu = self.fc_mu(h3)
        sigma = self.fc_sigma(h3)

        return pi, mu, sigma

    def forward(self, x):
        x = x.view(-1, original_dim)
        z = self.encode(x)
        pi, mu, sigma = self.decode(z)

        # Reshape to (batch, original_dim, num_mixtures)
        batch_size = x.size(0)
        pi = pi.view(batch_size, original_dim, self.num_mixtures)
        mu = mu.view(batch_size, original_dim, self.num_mixtures)
        sigma = sigma.view(batch_size, original_dim, self.num_mixtures)

        # Apply activations
        pi = F.softmax(pi, dim=-1)
        sigma = F.softplus(sigma) + 1e-6  # Ensure positive

        # Compute expected value as reconstruction
        recon_x = torch.sum(pi * mu, dim=-1)

        return recon_x, z, (pi, mu, sigma)

    def gaussian_probability(self, x, mu, sigma):
        """Compute Gaussian probability"""
        return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def loss_function(self, recon_x, x, z, mixture_params):
        pi, mu, sigma = mixture_params
        batch_size = x.size(0)
        x_expanded = x.view(batch_size, original_dim, 1).expand_as(mu)

        # Compute mixture likelihood
        gaussian_probs = self.gaussian_probability(x_expanded, mu, sigma)
        weighted_probs = pi * gaussian_probs
        mixture_likelihood = torch.sum(weighted_probs, dim=-1)

        # Negative log likelihood
        nll_loss = -torch.mean(torch.log(mixture_likelihood + 1e-8))

        # Simple MSE as auxiliary loss
        mse_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # Latent regularization
        latent_reg = torch.mean(z ** 2)

        total_loss = 0.7 * nll_loss + 0.3 * mse_loss + 0.0001 * latent_reg
        return total_loss, mse_loss, nll_loss, latent_reg
