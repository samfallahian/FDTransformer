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
import pandas as pd

from ConvolutionalAutoencoder import ConvolutionalAutoencoder


# class CustomDataset(torch.utils.data.Dataset):
#     # Dataset class for loading and pre-processing data
#     def __init__(self, data):
#         self.data = data
#
#     def __len__(self):
#         # Returns the total number of samples
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # Retrieves a sample at the given index idx
#         sample = self.data[idx]
#         sample = sample.view(3, 125)  # Reshapes the sample to match the input size of the model
#         return sample


class Decode_Data:
    # Encode data using the Convolutional Autoencoder pre-trained model

    def __init__(self, device, saved_model_path="/mnt/d/sources/cgan/playground/convolutional/saved_models/checkpoint_300.pth"):
        self.model = ConvolutionalAutoencoder().to(device)
        self.device = device
        # self.data = data
        self.saved_model_path = saved_model_path


        # self.orig_data_loader = torch.utils.data.DataLoader(self.data,
        #                                                     batch_size=self.batch_size)

    def decoded_tensor1(self, encoded_data):
        # print("Input Shape: ",encoded_data.shape)
        df = self.tensor_to_dataframe(encoded_data)
        self.model.load_state_dict(torch.load(self.saved_model_path)["model_state_dict"])
        appended_tensor = torch.empty(0, 3, 125).float().to(self.device)

        new_data = self.dataframe_to_tensor(df)

        for i, data in enumerate(new_data):
            decoded = self.model.decode(data.float().to(self.device))
            appended_tensor = torch.cat((appended_tensor, decoded), dim=0)
        print(f"Encoded tensor shape: {appended_tensor.shape}")
        return appended_tensor

    @staticmethod
    def tensor_to_dataframe(tensor):
        tensor_np = tensor.detach().cpu().numpy()
        # Reshape tensor to 2D
        num_samples = tensor_np.shape[0]
        tensor_flat = tensor_np.reshape(num_samples, -1)

        df = pd.DataFrame(tensor_flat)
        return df

    @staticmethod
    def dataframe_to_tensor(df):
        tensor_flat = df.to_numpy()

        # Reshape tensor to original shape
        num_samples, num_features = tensor_flat.shape
        tensor_shape = (num_samples, 48, 48)  # Adjust the shape according to your original tensor

        tensor = torch.from_numpy(tensor_flat.reshape(tensor_shape))
        tensor.requires_grad = False
        return tensor

    def decoded_tensor(self, encoded_data):
        # print("Input Shape: ",encoded_data.shape)
        self.model.load_state_dict(torch.load(self.saved_model_path)["model_state_dict"])
        appended_tensor = torch.empty(0, 3, 125).float().to(self.device)

        for i, data in enumerate(encoded_data):
            decoded = self.model.decode(data.float().to(self.device))
            appended_tensor = torch.cat((appended_tensor, decoded), dim=0)
        # print(f"Encoded tensor shape: {appended_tensor.shape}")
        return appended_tensor