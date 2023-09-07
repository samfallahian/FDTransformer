import torch
import torch.nn as nn
import torch.nn.functional as F

class HybrdidAutoencoder(nn.Module):
    def __init__(self, latent_size=(8, 6)):
        super(HybrdidAutoencoder, self).__init__()

        # Parameters
        self.latent_size = latent_size

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Linear(3 * 125, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        ])

        self.fc_mu = nn.Linear(32, self.latent_size[0] * self.latent_size[1])
        self.fc_logvar = nn.Linear(32, self.latent_size[0] * self.latent_size[1])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Linear(self.latent_size[0] * self.latent_size[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 125)
        ])

        # Loss criterion
        self.criterion = nn.MSELoss()

    def encode(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3*125))
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 3, 125), mu, logvar

    def loss_function(self, reconstruction, original, mu, logvar):
        MSE = self.criterion(reconstruction.view(-1, 3*125), original.view(-1, 3*125))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = MSE + KLD
        return loss
