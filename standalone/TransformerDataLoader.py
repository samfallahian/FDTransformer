from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, source_len=4):
        self.df = df
        self.source_len = source_len

        # Flatten all the data into sequences and keep track of the coordinates
        self.sequences = []

    def __len__(self):
        return len(self.df) * (self.df.columns.size - self.source_len)  # sliding window of 4

    def __getitem__(self, idx):
        row_idx = idx // (self.df.columns.size - self.source_len)
        col_idx = idx % (self.df.columns.size - self.source_len)

        src_values = [self.df.iloc[row_idx, col_idx + i] for i in range(4)]
        x = torch.tensor(np.array(src_values), dtype=torch.float32)

        y_values = self.df.iloc[row_idx, col_idx + 4]
        y = torch.tensor(np.array(y_values), dtype=torch.float32)

        return x, y
