import wandb
import torch.nn as nn
from torch import optim
from CAE import CAE
import torch
from torch.utils.data import DataLoader, random_split
from HD5Dataset import HD5Dataset
from torch.optim.lr_scheduler import StepLR

wandb.init(project='caeV2')

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = CAE().to(device)
print(model)
criterion = nn.MSELoss()
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
    model.train()  # Set model to training mode
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)  # Only MSE Loss

        loss.backward()
        optimizer.step()

        wandb.log({"MSE": loss.item(), "Loss": loss.item(),
                   "Min Latent Value": torch.min(outputs).item(),
                   "Max Latent Value": torch.max(outputs).item(),
                   "Epoch Percentage": (epoch / num_epochs) * 100})

    scheduler.step()  # adjust the learning rate through scheduler
    print(f"Epoch: {epoch}, MSE Loss: {loss.item()}")

    # A Basic Validation Step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0.0
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    wandb.log({"Validation Loss": avg_loss})
    print(f"Validation Loss: {avg_loss}")
