import torch.optim as optim
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tensorflow import keras
import wandb
from wandb.keras import WandbCallback
import os
import numpy as np
import matplotlib.pyplot as plt

from ConvolutionalAutoencoder import ConvolutionalAutoencoder


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.view(3, 125)
        return sample


class Train_Conv:
    def split_data(self, data, split_ratio=0.8):
        train_len = int(len(data) * split_ratio)
        val_len = len(data) - train_len
        train_data, val_data = random_split(data, [train_len, val_len])
        return train_data, val_data

    def __init__(self, model, device, data_path="_data_train_autoencoder_flat.pickle", batch_size=1000, lr=0.00001):
        wandb.init(project='ConvolutionalAEv3')
        config = wandb.config
        config.batch_size = batch_size
        config.lr = lr

        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = CustomDataset(pickle.load(open(data_path, "rb")))
        print(f"Data loaded. Total samples: {len(self.data)}")

        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.train_data, self.val_data = self.split_data(self.data)
        self.epochs = 10000
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = "saved_models"  # Directory to save the models
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist

    def train(self):
        train_loss = []
        val_loss = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_error = 0.0

            for i, data in enumerate(self.train_loader):
                inputs = data.float().to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(inputs)
                    loss = self.model.criterion(inputs, outputs)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                # Placeholder error calculation
                running_error += torch.mean(torch.abs(inputs - outputs[0])).item()

            wandb.log({"loss": running_loss / 1000, "error": running_error / 1000})

            if epoch % 100 == 99:
                print(f"Epoch: {epoch + 1} Loss: {running_loss / 1000} Error: {running_error / 1000}")

            train_loss.append(running_loss / 1000)

            # Validation
            self.model.eval()
            running_val_loss = 0.0
            running_val_error = 0.0

            with torch.no_grad():
                for i, data in enumerate(self.val_loader):
                    inputs = data.float().to(self.device)
                    outputs = self.model(inputs)
                    loss = self.model.loss_function(inputs, outputs)
                    running_val_loss += loss.item()
                    # Placeholder error calculation
                    running_val_error += torch.mean(torch.abs(inputs - outputs[0])).item()

            val_loss.append(running_val_loss / len(self.val_loader))
            wandb.log({"val_loss": running_val_loss / len(self.val_loader),
                       "val_error": running_val_error / len(self.val_loader)})

            if epoch % 100 == 99:
                print(f"Validation Loss: {running_val_loss / len(self.val_loader)} "
                      f"Validation Error: {running_val_error / len(self.val_loader)}")

            # Save the model every 100 epochs
            if epoch % self.save_interval == 0:
                model_path = os.path.join(self.save_directory, f"model_epoch_{epoch}.pt")
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved at {model_path}")

                # Generate a heat map for an arbitrary input
                # Generate a heat map for an arbitrary input
                arbitrary_input = torch.randn(1, 3, 125).to(self.device)
                _, encoded = self.model(arbitrary_input)

                encoded = encoded.view(8, 125)  # Reshape the encoded tensor

                encoded = encoded.detach().cpu().numpy()

                # Rest of the code for logging and plotting the heatmap
                # ...

                # Log the encoded portion of the auto-encoder to wandb
                wandb.log({"encoded": wandb.Histogram(np.histogram(encoded, bins='auto')[0].tolist())})

                # Plot the heat map
                plt.imshow(encoded, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Epoch: {epoch + 1}")
                plt.xlabel("Encoded Dimension")
                plt.ylabel("Sample")
                plt.savefig(f"heatmap_epoch_{epoch}.png")
                plt.close()

        print('Finished Training')
        return train_loss, val_loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = ConvolutionalAutoencoder().to(device)
    trainer = Train_Conv(model, device)
    train_loss, val_loss = trainer.train()
