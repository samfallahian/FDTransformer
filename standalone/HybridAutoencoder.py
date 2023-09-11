import torch
import torch.nn as nn
import torch.optim as optim
import wandb

class HybridAutoencoder(nn.Module):
    """
    A simple Hybrid (Variational) Autoencoder that encodes input of size (batch, 125, 3)
    into latent representation of size (batch, 8, 6) and decodes it back.
    """
    def __init__(self, latent_size=(8, 6)):
        super(HybridAutoencoder, self).__init__()

        # Convolution for 3 channels as input
        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)

        # Encoding layers
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 63, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size[0] * latent_size[1] * 2)
        )

        # Decoding layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size[0] * latent_size[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 63)
        )

        # Deconvolution to revert the convolution
        self.deconv1 = nn.ConvTranspose1d(16, 3, kernel_size=5, stride=2, padding=2)

    def encode(self, x):
        # Change dimensions to fit the Conv1D: (batch_size, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        results = self.encoder(x)
        split_dim = results.size(1) // 2
        return results[:, :split_dim], results[:, split_dim:]

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(x.size(0), 16, 63)  # Adjust shape for deconv layer
        x = self.deconv1(x)
        x = x.permute(0, 2, 1)  # Revert to original dimensions: (batch_size, length, channels)
        return x

    def forward(self, x):
        # Encoding step
        mu, logvar = self.encode(x)

        # Reparameterization trick for variational part
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decoding step
        return self.decode(z), mu, logvar

def loss_function(reconstructed_x, x, mu, logvar, beta):
    """
    Combined loss for VAE = reconstruction loss + KL divergence
    """
    # MSE loss for reconstruction error
    MSE = nn.MSELoss()(reconstructed_x, x)

    # KL Divergence to ensure the latent space has good properties
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine the two losses using the beta hyperparameter
    total_loss = MSE + beta * KLD

    return total_loss, MSE, KLD

def TrainHA(model, data_loader, epochs=10, learning_rate=0.001, beta=1.0):
    """
    Function to train the HybridAutoencoder using provided data.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize wandb for logging
    wandb.init(project="HybridAutoencoder")
    for epoch in range(epochs):
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()

            # Forward pass: get the reconstructed data and latent variables
            reconstructed_batch, mu, logvar = model(data)

            # Compute the loss
            loss, mse_loss, kld_loss = loss_function(reconstructed_batch, data, mu, logvar, beta)

            # Backward pass: compute gradient and update weights
            loss.backward()
            optimizer.step()

            # Find max and min values in the latent space
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            max_val = torch.max(z).item()
            min_val = torch.min(z).item()

            # Log metrics to wandb
            wandb.log({
                "Loss": loss.item(),
                "Reconstruction Loss": mse_loss.item(),
                "KLD Loss": kld_loss.item(),
                "Max Latent Value": max_val,
                "Min Latent Value": min_val
            })

        # Save the model every 50 epochs
        if epoch % 50 == 0:
            save_path = f"/Users/kkreth/PycharmProjects/cgan/standalone/saved_models/HYBRID_checkpoint_{epoch}.pth"
            torch.save({"model_state_dict": model.state_dict()}, save_path)
