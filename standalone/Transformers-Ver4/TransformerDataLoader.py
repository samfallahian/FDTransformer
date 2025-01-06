from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class SpatioTemporalDataset(Dataset):
    def __init__(self, dataframe, start_time_frame, sequence_length=5):
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.start_time_frame = start_time_frame
        self.num_samples = len(self.dataframe)
        self.max_time_frame = self.dataframe.columns[-1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of range")

        # Retrieve spatial coordinates
        coordinates = self.dataframe.iloc[idx][['x', 'y', 'z']].values.astype(np.float32)

        # Prepare the sequence of latent representations
        sequences = []
        num_time_window = (self.max_time_frame - self.start_time_frame + 1) // (self.sequence_length)

        for t in range(0, num_time_window):
            c_range = self.start_time_frame + t * self.sequence_length
            # windows = []
            for c in range(c_range, c_range + self.sequence_length):
                time_col = c
                if time_col > self.max_time_frame:
                    raise IndexError(f"Time column {time_col} not found in the dataframe")

                vector = self.dataframe[int(time_col)].iloc[idx]
                # windows.append(vector)
                sequences.append(vector)

            # windows = np.stack(windows)
            # sequences.append(windows)
        sequences = np.stack(sequences)
        return coordinates, torch.tensor(sequences, dtype=torch.float)
