from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, data_by_coords, source_len=8, target_len=2):
        self.data_by_coords = data_by_coords
        self.source_len = source_len
        self.target_len = target_len
        self.total_len = source_len + target_len

        # Flatten all the data into sequences and keep track of the coordinates
        self.sequences = []
        for coord, time_series in self.data_by_coords.items():
            for i in range(len(time_series) - self.total_len + 1):
                self.sequences.append((coord,
                                       time_series[i:i + self.source_len],
                                       time_series[i + self.source_len:i + self.source_len + target_len]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        coords, source_sequence, target_sequence = self.sequences[idx]

        # Convert each sequence into a single tensor
        source = torch.stack(source_sequence).view(self.source_len, -1)
        target = torch.stack(target_sequence).view(self.target_len, -1)

        # Convert coordinates to tensor
        coords_tensor = torch.tensor(coords).float()

        return coords_tensor, source, target
