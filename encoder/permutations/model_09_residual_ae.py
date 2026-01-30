"""
Model 09: Residual Autoencoder
Uses skip connections (residual blocks) for better gradient flow.
Loss: Reconstruction (MSE) with residual architecture
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ELU()

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = out + residual  # Skip connection
        return self.activation(out)

class ResidualAE(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(ResidualAE, self).__init__()

        hidden_dim1 = 250
        hidden_dim2 = 150
        hidden_dim3 = 100

        # Encoder
        self.enc_in = nn.Linear(original_dim, hidden_dim1)
        self.enc_res1 = ResidualBlock(hidden_dim1, dropout_rate)

        self.enc_down1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.enc_res2 = ResidualBlock(hidden_dim2, dropout_rate)

        self.enc_down2 = nn.Linear(hidden_dim2, hidden_dim3)
        self.enc_res3 = ResidualBlock(hidden_dim3, dropout_rate)

        self.enc_out = nn.Linear(hidden_dim3, latent_dim)
        self.tanh = nn.Tanh()

        # Decoder
        self.dec_in = nn.Linear(latent_dim, hidden_dim3)
        self.dec_res1 = ResidualBlock(hidden_dim3, dropout_rate)

        self.dec_up1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.dec_res2 = ResidualBlock(hidden_dim2, dropout_rate)

        self.dec_up2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.dec_res3 = ResidualBlock(hidden_dim1, dropout_rate)

        self.dec_out = nn.Linear(hidden_dim1, original_dim)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x):
        h = self.activation(self.enc_in(x))
        h = self.enc_res1(h)

        h = self.activation(self.enc_down1(h))
        h = self.enc_res2(h)

        h = self.activation(self.enc_down2(h))
        h = self.enc_res3(h)

        return self.tanh(self.enc_out(h))

    def decode(self, z):
        h = self.activation(self.dec_in(z))
        h = self.dec_res1(h)

        h = self.activation(self.dec_up1(h))
        h = self.dec_res2(h)

        h = self.activation(self.dec_up2(h))
        h = self.dec_res3(h)

        return self.dec_out(h)

    def forward(self, x):
        x = x.view(-1, original_dim)
        z = self.encode(x)
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # Small L2 regularization
        l2_reg = torch.mean(z ** 2)

        total_loss = recon_loss + 0.00005 * l2_reg
        return total_loss, recon_loss, l2_reg, torch.tensor(0.0)
