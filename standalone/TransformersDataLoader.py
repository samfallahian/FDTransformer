from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, seq_len, batch_src_seq, batch_tgt_seq):
        self.data = data
        self.seq_len = seq_len
        self.total_seq = batch_src_seq + batch_tgt_seq  # Here is 10
        self.batch_src_seq = batch_src_seq  # Here is 9 sequences for source

    def __len__(self):
        return (len(self.data) - self.total_seq * self.seq_len) + 1  # + 1 means the last possible sequence

    def __getitem__(self, idx):
        src = [self.data[idx + i * self.seq_len: idx + (i + 1) * self.seq_len] for i in range(self.batch_src_seq)]
        tgt = self.data[idx + self.batch_src_seq * self.seq_len: idx + self.total_seq * self.seq_len]
        return src, tgt
