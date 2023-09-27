import wandb
import torch.nn as nn
from torch import optim
from CAE import CAE  # Ensure CAE is the correct name of your AutoEncoder Class
import torch
from torch.utils.data import DataLoader, random_split
from HD5Dataset import HD5Dataset  # Ensure HD5Dataset is the correct name of your Dataset Class

# Initialize wandb
wandb.init(project='cae')

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = CAE().to(device)
print(model)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize and split the dataset
dataset = HD5Dataset('/Users/kkreth/PycharmProjects/merged_10p4.hdf')
train_size = int(0.5 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Initialize the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs in train_loader:
        inputs = inputs.to(device)
        print("Input Shape: ", inputs.shape)  # Should print: torch.Size([32, 125, 3])

        optimizer.zero_grad()
        outputs = model(inputs)
        mse_loss = criterion(outputs, inputs)
        kl_div = -0.5 * torch.sum(1 + outputs - inputs.pow(2) - outputs.exp())
        loss = 0.75 * kl_div + 0.25 * mse_loss

        loss.backward()
        optimizer.step()

        # Log to wandb
        wandb.log({"MSE": mse_loss.item(), "KL divergence": kl_div.item(), "Loss": loss.item(),
                   "Min Latent Value": torch.min(outputs).item(), "Max Latent Value": torch.max(outputs).item(),
                   "Epoch Percentage": (epoch / num_epochs) * 100})

    # Learning rate scheduler for every 10% of epochs
    if epoch % (num_epochs // 10) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9

    print(f"Epoch: {epoch}, MSE Loss: {mse_loss.item()}, KL Div: {kl_div.item()}, Total Loss: {loss.item()}")
