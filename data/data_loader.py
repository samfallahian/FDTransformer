import torch
from sklearn.model_selection import train_test_split
from utils import helpers
from torch.utils.data import DataLoader


class DataModelLoader:
    def __init__(self, data, labels):
        """Convert to tensor"""
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        """ Load  configurations """
        config = helpers.Config()
        cfg = config.from_json("data")
        self.batch_size = cfg.batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



    def train_test_data_loader(self):
        """Split data into train and test"""
        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=.2)
        """Convert into PyTorch Datasets"""
        train_data_t = torch.utils.data.TensorDataset(train_data, train_labels)
        test_data_t = torch.utils.data.TensorDataset(test_data, test_labels)
        """Convert into dataloader objects"""
        train_loader = DataLoader(train_data_t, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data_t, batch_size=test_data_t.tensors[0].shape[0])

        return train_loader, test_loader

    def all_data_loader(self):
        """Convert into PyTorch Datasets"""
        data = torch.utils.data.TensorDataset(self.data, self.labels)
        """Convert into dataloader objects"""
        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        return data_loader
