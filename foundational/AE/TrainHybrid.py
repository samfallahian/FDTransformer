import wandb
import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import DataLoader, random_split
from HD5Dataset import HD5Dataset
from torch.optim.lr_scheduler import StepLR
from HybridAutoencoder import HybridAutoencoder, loss_function # Import HybridAutoencoder

wandb.init(project='hybridAutoencoderV42')

device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # Adjusted device fallback to "cpu"
model = HybridAutoencoder().to(device)  # Replaced CAE with HybridAutoencoder
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = HD5Dataset('/Users/kkreth/PycharmProjects/merged_10p4.hdf')
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, mu, logvar, z = model(inputs)

        loss, mse_loss, kld_loss, reg_loss = loss_function(outputs, inputs, mu, logvar, beta=0.5, lambda_reg=0.0)

        loss.backward()
        optimizer.step()

        min_val = torch.min(z).item()  # get the minimum value in latent space
        max_val = torch.max(z).item()  # get the maximum value in latent space

        wandb.log({
            "MSE": mse_loss.item(), "Loss": loss.item(),
            "Epoch Percentage": (epoch / num_epochs) * 100,
            "KLD Loss": kld_loss.item(),
            "Min Latent Value": min_val,  # Log min latent value
            "Max Latent Value": max_val  # Log max latent value
        })

    scheduler.step()
    print(f"Epoch: {epoch}, Total Loss: {loss.item()}, MSE Loss: {mse_loss.item()}, KLD Loss: {kld_loss.item()}")

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs, mu, logvar, z = model(inputs)
            loss, mse_loss, kld_loss, reg_loss = loss_function(outputs, inputs, mu, logvar, beta=0.5, lambda_reg=0.0)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    wandb.log({"Validation Loss": avg_loss})
    print(f"Validation Loss: {avg_loss}")