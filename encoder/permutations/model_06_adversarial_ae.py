"""
Model 06: Adversarial Autoencoder
Originally based on: "Adversarial Autoencoders" (Makhzani et al., 2015).

MLA Citations:
1. Makhzani, Alireza, et al. "Adversarial Autoencoders." arXiv preprint arXiv:1511.05644, 2015. https://arxiv.org/pdf/1511.05644.pdf
2. (Makhzani et al. 1-16)
3. Makhzani et al., "Adversarial Autoencoders," arXiv (2015).

Deviations from Paper:
- Uses a simplified 3-layer discriminator (128-64-1) for a 47D latent space.
- Combined loss function in a single forward pass, while the paper often describes alternating training phases.

Relative Performance (MSE): 2.840e-04
"""
import torch
from torch import nn
import torch.nn.functional as F

original_dim = 375
latent_dim = 47

class AdversarialAE(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(AdversarialAE, self).__init__()

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

        # Discriminator for latent space
        self.disc_fc1 = nn.Linear(latent_dim, 128)
        self.disc_fc2 = nn.Linear(128, 64)
        self.disc_fc3 = nn.Linear(64, 1)

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

    def discriminate(self, z):
        h = F.relu(self.disc_fc1(z))
        h = F.relu(self.disc_fc2(h))
        return torch.sigmoid(self.disc_fc3(h))

    def forward(self, x):
        x = x.view(-1, original_dim)
        z = self.encode(x)
        return self.decode(z), z

    def loss_function(self, recon_x, x, z):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')

        # Adversarial loss: encoder tries to fool discriminator
        real_prior = torch.randn_like(z)
        disc_real = self.discriminate(real_prior)
        disc_fake = self.discriminate(z)

        # Generator loss (encoder wants disc_fake to be 1)
        gen_loss = F.binary_cross_entropy(disc_fake, torch.ones_like(disc_fake))

        # Discriminator loss
        disc_loss = (F.binary_cross_entropy(disc_real, torch.ones_like(disc_real)) +
                     F.binary_cross_entropy(disc_fake.detach(), torch.zeros_like(disc_fake)))

        total_loss = recon_loss + 0.1 * gen_loss
        return total_loss, recon_loss, gen_loss, disc_loss
