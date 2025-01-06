import torch
from torch.utils.data import DataLoader, random_split


class CustomDataset(torch.utils.data.Dataset):
    # Dataset class for loading and pre-processing data
    def __init__(self, data, batch_size = 1):
        self.data = data
        self.batch_size= batch_size

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieves a sample at the given index idx
        sample = self.data[idx]
        sample = sample.view(3, 125)  # Reshapes the sample to match the input size of the model
        data_loader = DataLoader(sample, batch_size=self.batch_size)
        return data_loader
