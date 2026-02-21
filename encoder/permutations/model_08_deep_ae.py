"""
Model 08: Deep Autoencoder
Originally based on: "Reducing the Dimensionality of Data with Neural Networks" (Hinton & Salakhutdinov, 2006).

MLA Citations:
1. Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "Reducing the Dimensionality of Data with Neural Networks." Science, vol. 313, no. 5786, 2006, pp. 504-07. https://www.science.org/doi/10.1126/science.1127647 (PDF: https://www.cs.toronto.edu/~hinton/science.pdf)
2. (Hinton and Salakhutdinov 504-07)
3. Hinton and Salakhutdinov, "Reducing the Dimensionality," Science (2006).

Deviations from Paper:
- Trained end-to-end with backpropagation; original paper relied on RBM-based pre-training.
- Uses modern activation functions (GELU, ELU, LeakyReLU) and LayerNorm, which were not available at the time of publication.

Relative Performance (MSE): 5.730e-04
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class DeepAE(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeepAE, self).__init__()

        # Deeper architecture with more gradual dimension reduction
        hidden_dims = [300, 250, 200, 150, 100, 75]

        # Encoder
        self.enc1 = nn.Linear(original_dim, hidden_dims[0])
        self.enc1_act = nn.GELU()
        self.enc1_drop = nn.Dropout(dropout_rate)
        self.enc1_norm = nn.LayerNorm(hidden_dims[0])

        self.enc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.enc2_act = nn.ELU()
        self.enc2_drop = nn.Dropout(dropout_rate)
        self.enc2_norm = nn.LayerNorm(hidden_dims[1])

        self.enc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.enc3_act = nn.LeakyReLU(0.2)
        self.enc3_drop = nn.Dropout(dropout_rate)
        self.enc3_norm = nn.LayerNorm(hidden_dims[2])

        self.enc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.enc4_act = nn.ELU()
        self.enc4_drop = nn.Dropout(dropout_rate)
        self.enc4_norm = nn.LayerNorm(hidden_dims[3])

        self.enc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.enc5_act = nn.GELU()
        self.enc5_drop = nn.Dropout(dropout_rate)
        self.enc5_norm = nn.LayerNorm(hidden_dims[4])

        self.enc6 = nn.Linear(hidden_dims[4], hidden_dims[5])
        self.enc6_act = nn.ELU()
        self.enc6_drop = nn.Dropout(dropout_rate)

        self.enc_final = nn.Linear(hidden_dims[5], latent_dim)
        self.tanh = nn.Tanh()

        # Decoder (mirror of encoder)
        self.dec1 = nn.Linear(latent_dim, hidden_dims[5])
        self.dec1_act = nn.ELU()
        self.dec1_drop = nn.Dropout(dropout_rate)

        self.dec2 = nn.Linear(hidden_dims[5], hidden_dims[4])
        self.dec2_act = nn.GELU()
        self.dec2_drop = nn.Dropout(dropout_rate)
        self.dec2_norm = nn.LayerNorm(hidden_dims[4])

        self.dec3 = nn.Linear(hidden_dims[4], hidden_dims[3])
        self.dec3_act = nn.ELU()
        self.dec3_drop = nn.Dropout(dropout_rate)
        self.dec3_norm = nn.LayerNorm(hidden_dims[3])

        self.dec4 = nn.Linear(hidden_dims[3], hidden_dims[2])
        self.dec4_act = nn.LeakyReLU(0.2)
        self.dec4_drop = nn.Dropout(dropout_rate)
        self.dec4_norm = nn.LayerNorm(hidden_dims[2])

        self.dec5 = nn.Linear(hidden_dims[2], hidden_dims[1])
        self.dec5_act = nn.ELU()
        self.dec5_drop = nn.Dropout(dropout_rate)
        self.dec5_norm = nn.LayerNorm(hidden_dims[1])

        self.dec6 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.dec6_act = nn.GELU()
        self.dec6_drop = nn.Dropout(dropout_rate)
        self.dec6_norm = nn.LayerNorm(hidden_dims[0])

        self.dec_final = nn.Linear(hidden_dims[0], original_dim)

    def encode(self, x):
        h = self.enc1_norm(self.enc1_drop(self.enc1_act(self.enc1(x))))
        h = self.enc2_norm(self.enc2_drop(self.enc2_act(self.enc2(h))))
        h = self.enc3_norm(self.enc3_drop(self.enc3_act(self.enc3(h))))
        h = self.enc4_norm(self.enc4_drop(self.enc4_act(self.enc4(h))))
        h = self.enc5_norm(self.enc5_drop(self.enc5_act(self.enc5(h))))
        h = self.enc6_drop(self.enc6_act(self.enc6(h)))
        return self.tanh(self.enc_final(h))

    def decode(self, z):
        h = self.dec1_drop(self.dec1_act(self.dec1(z)))
        h = self.dec2_norm(self.dec2_drop(self.dec2_act(self.dec2(h))))
        h = self.dec3_norm(self.dec3_drop(self.dec3_act(self.dec3(h))))
        h = self.dec4_norm(self.dec4_drop(self.dec4_act(self.dec4(h))))
        h = self.dec5_norm(self.dec5_drop(self.dec5_act(self.dec5(h))))
        h = self.dec6_norm(self.dec6_drop(self.dec6_act(self.dec6(h))))
        return self.dec_final(h)

    def forward(self, x):
        x = x.view(-1, original_dim)
        z = self.encode(x)
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # L2 regularization on latent
        l2_reg = torch.mean(z ** 2)

        total_loss = recon_loss + 0.0001 * l2_reg
        return total_loss, recon_loss, l2_reg, torch.tensor(0.0)
