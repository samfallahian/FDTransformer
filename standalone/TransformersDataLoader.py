from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, source_size=8, target_size=2):
        self.data = data
        self.source_size = source_size
        self.target_size = target_size
        self.chunk_size = source_size + target_size

    def __len__(self):
        # Number of chunks we can make
        return len(self.data) // self.chunk_size

    def __getitem__(self, idx):
        # Start index of chunk
        start_idx = idx * self.chunk_size

        # Splitting the chunk into source and target
        source = self.data[start_idx:start_idx + self.source_size]
        target = self.data[start_idx + self.source_size:start_idx + self.chunk_size]

        return source, target
