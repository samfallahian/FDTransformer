from torch.utils.data import Dataset
import h5py
import torch

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            # Get the shape of the data and store it
            self.data_shape = f['my_data'].shape

    def __len__(self):
        return self.data_shape[0]

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            # Return the tensor at the specified index
            return torch.tensor(f['my_data'][idx])
