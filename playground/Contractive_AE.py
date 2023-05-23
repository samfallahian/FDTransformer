import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()
        self.beta = 1e-4

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded):
        mse_loss = self.criterion(decoded, x)
        jacobian = torch.autograd.functional.jacobian(lambda x: encoded, x)
        contraction_loss = torch.mean(torch.sum(jacobian ** 2, dim=1))
        total_loss = mse_loss + (self.beta * contraction_loss)
        return total_loss

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters
input_size = 784  # MNIST images are 28x28 pixels
hidden_size = 64
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Load MNIST dataset
train_dataset = MNIST(root="./data", train=True, transform=ToTensor(), download=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the autoencoder
autoencoder = ContractiveAutoencoder(input_size, hidden_size).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_dataloader):
        # Move images to the device
        images = images.to(device)
        inputs = images.view(images.size(0), -1)

        # Forward pass
        encoded, decoded = autoencoder(inputs)

        # Compute the loss
        loss = autoencoder.loss_function(inputs, decoded, encoded)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")
