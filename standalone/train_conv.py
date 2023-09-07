import torch.optim as optim
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt

from HybrdidAutoencoder import HybrdidAutoencoder


class CustomDataset(torch.utils.data.Dataset):
    # Dataset class for loading and pre-processing data
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves a sample at the given index idx
        sample = self.data[idx]
        sample = sample.view(3, 125)  # Reshapes the sample to match the input size of the model
        return sample


class Train_Conv:
    # Trainer class for the Convolutional Autoencoder model
    def split_data(self, data, split_ratio=0.8):
        # Splits the dataset into training and validation sets
        train_len = int(len(data) * split_ratio)
        val_len = len(data) - train_len
        train_data, val_data = random_split(data, [train_len, val_len])
        return train_data, val_data

    def __init__(self, model, device, data_path="/home/kkreth_umassd_edu/DL-PTV-TrainingData/AE_training_data.hdf",
                 batch_size=1000, lr=0.001):
        # Initializes the model and the necessary parameters for training
        wandb.init(project='ConvAEv7UNITY')  # Starts a new run on Weights & Biases
        config = wandb.config
        config.batch_size = batch_size
        config.lr = lr
        self.debug = True
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = CustomDataset(torch.load(data_path)) # Loads the dataset
        print(f"Data loaded. Total samples: {len(self.data)}")

        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.train_data, self.val_data = self.split_data(self.data)
        self.epochs = 100
        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data,
                                                      batch_size=self.batch_size, shuffle=True)
        self.save_interval = 50  # Save the model every 100 epochs
        self.save_directory = "saved_models"  # Directory to save the models
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist

    def train(self):
        # Function to perform the training of the model
        train_loss = []
        val_loss = []
        encoded_shape = None  # Variable to store the shape of the encoded tensor

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_error = 0.0

            for i, data in enumerate(self.train_loader):
                inputs = data.float().to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    # Forward pass through the model
                    reconstruction, mu, logvar = self.model(inputs)
                    # Compute the loss using the new loss function
                    loss = self.model.loss_function(reconstruction, inputs, mu, logvar)

                # Backward pass and optimization
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Store the shape of the encoded tensor after the first forward pass
                if encoded_shape is None:
                    encoded_shape = mu.shape[1:]

                running_loss += loss.item()

            # Compute and log the average training loss
            train_loss.append(running_loss / len(self.train_loader))
            wandb.log({"Train Loss": train_loss[-1]})

            # Validate the model
            self.model.eval()
            running_loss = 0.0
            for i, data in enumerate(self.val_loader):
                inputs = data.float().to(self.device)
                with torch.no_grad():
                    reconstruction, _, _ = self.model(inputs)  # Updated this line
                    loss = self.model.criterion(inputs, reconstruction)
                running_loss += loss.item()

            # Compute and log the average validation loss
            val_loss.append(running_loss / len(self.val_loader))
            wandb.log({"Validation Loss": val_loss[-1]})

            # Save the model
            if epoch % self.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss[-1],
                    'encoded_shape': encoded_shape
                }, os.path.join(self.save_directory, f"checkpoint_AEHybrid_BatchNORM_{epoch}.pth"))
        print('Finished Training')
        return train_loss, val_loss

    def plot_loss(self, train_loss, val_loss):
        # Plots the training and validation losses
        epochs = range(len(train_loss))
        plt.figure()
        plt.plot(epochs, train_loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = HybrdidAutoencoder().to(device)
trainer = Train_Conv(model, device)
train_loss, val_loss = trainer.train()
trainer.plot_loss(train_loss, val_loss)
