import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import pickle
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define constants
seq_len = 48
epochs = 10
ninp = 48  # The dimension of your input feature
nhid = 128  # 200  # Dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # Number of heads in nn.MultiheadAttention models
dropout = 0.1  # Dropout value
lr = 0.001
log_interval = 50
batch_src_seq = 9
batch_tgt_seq = 1
scheduler_step = 1000
lr_gamma = 0.95


def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


# data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor_small.pickle")
data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle")
data = data.view(-1, 48)
print(data.shape)

class TensorDataset(Dataset):
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


dataset = TensorDataset(data, seq_len, batch_src_seq, batch_tgt_seq)
dataloader = DataLoader(dataset, batch_size=seq_len, shuffle=False)


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, src_mask)
        return output


model = TransformerModel(ninp, nhead, nhid, nlayers, dropout).to(device)

scaler = GradScaler()  # For mixed precision training
# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# initialize the scheduler
scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=lr_gamma)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.
    running_loss = 0.
    start_time = time.time()
    print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')
    src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
    for batch, (src_batch, tgt) in enumerate(dataloader):
        # print("batch no ", batch, " len src", len(src_batch), " * ",len(src_batch[0]), " len target", len(tgt))
        tgt = tgt.to(device)
        optimizer.zero_grad()

        # with autocast():
        #     if src.size(0) != seq_len:
        #         src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
        #     output = model(src, src_mask)
        #     loss = criterion(output, tgt)
        # # Backward pass and optimization
        # scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # # Unscales the gradients of optimizer's assigned params in-place
        # scaler.unscale_(optimizer)
        # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # # Unscales gradients and calls or skips optimizer.step()
        # scaler.step(optimizer)
        # # Updates the scale for next iteration
        # scaler.update()
        # scheduler.step()

        # if src.size(0) != seq_len:
        #     src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
        # output = model(src, src_mask)
        # loss = criterion(output, tgt)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        for src in src_batch:
            src = src.to(device)
            if src.size(0) != seq_len:
                src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
            output = model(src, src_mask)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        running_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                epoch + 1, batch, len(data) // seq_len, scheduler.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss))
            total_loss = 0
            start_time = time.time()
    running_loss /= len(dataloader)
    print(f'End of epoch {epoch + 1}, Running loss {running_loss:.2f}')

print("===============================================")
print(f'End of training at {datetime.now().time().strftime("%H:%M:%S")}')
