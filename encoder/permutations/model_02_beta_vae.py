"""
Model 02: β-VAE (Beta Variational Autoencoder)
Uses adjustable β parameter to balance reconstruction and disentanglement.
Loss: Reconstruction (MSE) + β * KL divergence
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class BetaVAE(nn.Module):
    def __init__(self, dropout_rate=0.2, beta=4.0):
        super(BetaVAE, self).__init__()
        self.beta = beta

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

        self.fc_mu = nn.Linear(hidden_dim3, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim3, latent_dim)

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

    def encode(self, x):
        h1 = self.dropout1(self.elu1(self.fc1(x)))
        h2 = self.dropout2(self.elu2(self.fc2(h1)))
        h3 = self.dropout3(self.elu3(self.fc3(h2)))
        return self.fc_mu(h3), self.fc_logvar(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = self.dropout4(self.elu4(self.fc5(z)))
        h2 = self.dropout5(self.elu5(self.fc6(h1)))
        h3 = self.dropout6(self.elu6(self.fc7(h2)))
        return self.fc8(h3)

    def forward(self, x):
        x = x.view(-1, original_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss, torch.tensor(0.0)
