import h5py
import torch
from torch.utils.data import Dataset


class HD5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            matrix = f[self.keys[index]][:]
        return torch.tensor(matrix, dtype=torch.float32)

    def run_tests(self):
        # Test the first item in dataset to ensure shape is [125, 3]
        item = self.__getitem__(0)
        assert item.shape == (125, 3), f"Expected shape (125, 3), but got {item.shape}"


# Instantiate your dataset object with the path to your hdf5 file
dataset = HD5Dataset('/Users/kkreth/PycharmProjects/merged_10p4.hdf')

# Run tests
dataset.run_tests()
