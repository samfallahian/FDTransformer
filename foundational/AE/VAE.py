import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

wandb.init(project="Variational_V2")



class VAE(nn.Module):
    def __init__(self, latent_dim=48, debug=False):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.debug = debug

        # Encoding layers
        self.enc1 = nn.Linear(3 * 125, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, 128)
        self.enc4 = nn.Linear(128, 64)
        self.enc5 = nn.Linear(64, 32)

        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)

        # Decoding layers
        self.dec1 = nn.Linear(latent_dim, 32)
        self.dec2 = nn.Linear(32, 64)
        self.dec3 = nn.Linear(64, 128)
        self.dec4 = nn.Linear(128, 256)
        self.dec5 = nn.Linear(256, 3 * 125)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        if self.debug:
            print("After enc1:", x.shape)
        x = F.relu(self.enc2(x))
        if self.debug:
            print("After enc2:", x.shape)
        x = F.relu(self.enc3(x))
        if self.debug:
            print("After enc3:", x.shape)
        x = F.relu(self.enc4(x))
        if self.debug:
            print("After enc4:", x.shape)
        x = F.relu(self.enc5(x))
        if self.debug:
            print("After enc5:", x.shape)

        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.dec1(z))
        if self.debug:
            print("After dec1:", z.shape)
        z = F.relu(self.dec2(z))
        if self.debug:
            print("After dec2:", z.shape)
        z = F.relu(self.dec3(z))
        if self.debug:
            print("After dec3:", z.shape)
        z = F.relu(self.dec4(z))
        if self.debug:
            print("After dec4:", z.shape)
        z = torch.tanh(self.dec5(z))
        if self.debug:
            print("After dec5:", z.shape)
        return z

    def forward(self, x):
        # Ensure input shape
        assert x.shape[1:] == (3, 125), "Input tensor should have shape [batch_size, 3, 125]"

        x = x.view(x.shape[0], -1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z).view(-1, 3, 125)

        # Logging values
        wandb.log({"Max latent value": torch.max(z).item(), "Min latent value": torch.min(z).item()})

        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    #print(recon_x.shape)
    #print(x.shape)
    MSE = F.mse_loss(recon_x.view(x.size(0), -1), x.view(x.size(0), -1))

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Adjusting the balance by increasing the weight of KLD
    loss = MSE + 1.2 * KLD

    # Logging values
    wandb.log({"Reconstruction Loss": MSE.item(), "KL Divergence": KLD.item()})
    return loss


# Test function as provided
import pytest


@pytest.mark.parametrize("batch_size", [32])
def test_vae_with_random_batch_sizes(batch_size):
    # Create a random input tensor
    input_tensor = torch.randn(32, 3, 125)
    latent_dim = 48

    # Initialize the VAE model
    model = VAE(debug=True)  # Debug mode enabled

    # Forward pass the input tensor through the VAE
    recon_tensor, mu, logvar = model(input_tensor)

    # Print shapes for debugging
    print("Input tensor shape:", input_tensor.shape)
    print("Reconstructed tensor shape:", recon_tensor.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    # Assert that the reconstructed tensor has the same shape as the input tensor
    assert recon_tensor.shape == input_tensor.shape

    # Assert the shape of mu and logvar to be [batch_size, latent_dim]
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


# Example training loop
def train(model, dataloader, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

# You'll need to define your dataloader and optimizer, and then call:
# train(model, dataloader, optimizer)
