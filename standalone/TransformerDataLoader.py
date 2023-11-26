from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SpatioTemporalDataset(Dataset):
    def __init__(self, dataframe, num_files, window_size=5, step_size=6):
        self.dataframe = dataframe
        self.window_size = window_size
        self.unique_locations = dataframe[['x', 'y', 'z']].drop_duplicates()
        self.step_size = step_size
        self.num_files = num_files

    def __len__(self):

        # return len(self.unique_locations) * (self.num_files - self.window_size + 1)
        num_windows_per_location = (self.num_files - self.window_size) // self.step_size + 1
        return len(self.unique_locations) * num_windows_per_location

    def __getitem__(self, idx):
        # Determine location and time window based on index
        # loc_idx = idx // (self.num_files - self.window_size + 1)
        # time_idx = idx % (self.num_files - self.window_size + 1)

        loc_idx = idx // ((self.num_files - self.window_size) // self.step_size + 1)
        time_idx = (idx % ((self.num_files - self.window_size) // self.step_size + 1)) * self.step_size

        # Extract the specific location
        location = self.unique_locations.iloc[loc_idx]
        x, y, z = location

        # Extracting the sliding window for this location
        window_data = self.dataframe[(self.dataframe['x'] == x) &
                                     (self.dataframe['y'] == y) &
                                     (self.dataframe['z'] == z)]
        window_data = window_data.iloc[time_idx:time_idx + self.window_size]

        # Extracting source, target, and time sequences
        source = np.array(window_data.iloc[:-1]['latent_representation'].tolist())
        target = window_data.iloc[-1]['latent_representation']  # Assuming this is a numpy array
        time_seq = np.array(window_data['time'].tolist())

        # Converting to tensors
        source_tensor = torch.FloatTensor(source)
        target_tensor = torch.FloatTensor(target)
        time_seq_tensor = torch.LongTensor(time_seq)

        return source_tensor, target_tensor, (x, y, z), time_seq_tensor