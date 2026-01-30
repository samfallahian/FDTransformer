"""
Model 03: Sparse Autoencoder
Applies L1 regularization on latent activations to encourage sparsity.
Loss: Reconstruction (MSE) + L1 penalty on latent space
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class SparseAE(nn.Module):
    def __init__(self, dropout_rate=0.2, sparsity_weight=1e-3):
        super(SparseAE, self).__init__()
        self.sparsity_weight = sparsity_weight

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
        z = self.encode(x)
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # L1 sparsity penalty on latent activations
        sparsity_loss = torch.mean(torch.abs(z))

        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        return total_loss, recon_loss, sparsity_loss, torch.tensor(0.0)
