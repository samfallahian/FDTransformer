import torch.optim as optim
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
# import wandb
import os
import numpy as np
import matplotlib.pyplot as plt

from ConvolutionalAutoencoder import ConvolutionalAutoencoder


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


class Encode_Data:
    # Encode data using the Convolutional Autoencoder pre-trained model

    def __init__(self, model, device, saved_model_path="saved_models/checkpoint_400.pth",
                 data_path="_data_train_autoencoder_flat.pickle",
                 batch_size=1):
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = CustomDataset(pickle.load(open(data_path, "rb")))  # Loads the dataset
        self.saved_model_path = saved_model_path

        print(f"Data loaded. Total samples: {len(self.data)}")

        self.batch_size = batch_size

        self.orig_data_loader = torch.utils.data.DataLoader(self.data,
                                                            batch_size=self.batch_size)

        self.save_directory = "dataset"  # Directory to save the encoded
        os.makedirs(self.save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    def create_encoded_tensor(self):
        if self.device == 'cuda':
            self.model.load_state_dict(torch.load(self.saved_model_path)["model_state_dict"])
        else:
            self.model.load_state_dict(torch.load(self.saved_model_path, map_location=self.device)["model_state_dict"])

        appended_tensor = torch.empty(0, 8, 6).float().to(self.device)

        for i, data in enumerate(self.orig_data_loader):
            reconstruction, encoded = self.model(data.float().to(self.device))
            appended_tensor = torch.cat((appended_tensor, encoded), dim=0)
        print(f"Encoded tensor shape: {appended_tensor.shape}")
        # Save the tensor using pickle
        with open(f"{self.save_directory}/appended_tensor_08082023.pickle", 'wb') as f:
            pickle.dump(appended_tensor, f)
        return appended_tensor


saved_model_path = r"/mnt/d/sources/cgan/playground/convolutional/saved_models/checkpoint_400.pth"
data_path = r"/mnt/d/sources/cgan/playground/convolutional/_data_train_autoencoder_flat.pickle"
# saved_model_path = "saved_models/checkpoint_400.pth"
# data_path = "_data_train_autoencoder_flat.pickle"
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = ConvolutionalAutoencoder().to(device)
encoding = Encode_Data(model=model, device=device, saved_model_path=saved_model_path, data_path= data_path)
encoding.create_encoded_tensor()
