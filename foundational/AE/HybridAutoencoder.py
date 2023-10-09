import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os


class HybridAutoencoder(nn.Module):
    def __init__(self, latent_size=(8, 6)):
        super(HybridAutoencoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 63, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size[0] * latent_size[1] * 2)
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_size[0] * latent_size[1], 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 63)
        )

        self.deconv1 = nn.ConvTranspose1d(16, 3, kernel_size=5, stride=2, padding=2)

    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        results = self.encoder(x)
        split_dim = results.size(1) // 2
        return results[:, :split_dim], results[:, split_dim:]

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(x.size(0), 16, 63)
        x = self.deconv1(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = torch.tanh(z)

        return self.decode(z), mu, logvar, z


def regularization_term(z):
    return torch.sum(torch.clamp(z - 1, min=0) ** 2) + torch.sum(torch.clamp(-1 - z, min=0) ** 2)


def loss_function(reconstructed_x, x, mu, logvar, beta, lambda_reg=0.1):
    MSE = nn.MSELoss()(reconstructed_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    REG = regularization_term(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar))

    #lambda_reg = 0.5  # As per your comment
    total_loss = MSE + beta * KLD + lambda_reg * REG

    return total_loss, MSE, KLD, REG


def TrainHA(model, data_loader, epochs=10, learning_rate=0.001, beta=1.0, lambda_reg=0.1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    wandb.init(project="HybridAutoencoder")

    for epoch in range(epochs):
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()

            reconstructed_batch, mu, logvar, z = model(data)
            loss, mse_loss, kld_loss, reg_loss = loss_function(reconstructed_batch, data, mu, logvar, beta, lambda_reg)

            loss.backward()
            optimizer.step()

            max_val = torch.max(z).item()
            min_val = torch.min(z).item()

            wandb.log({
                "Loss": loss.item(),
                "Reconstruction Loss": mse_loss.item(),
                "KLD Loss": kld_loss.item(),
                "Regularization Loss": reg_loss.item(),
                "Max Latent Value": max_val,
                "Min Latent Value": min_val
            })

        if epoch % 50 == 0:
            save_path = f"/Users/kkreth/PycharmProjects/cgan/standalone/saved_models/HYBRIDv2_checkpoint_{epoch}.pth"
            torch.save({"model_state_dict": model.state_dict()}, save_path)
