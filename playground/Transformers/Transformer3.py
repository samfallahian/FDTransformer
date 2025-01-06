import time

from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


# data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor_small.pickle")
data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle")
data = data.view(-1, 48)
print(data.shape)

# Define constants
SEQ_LEN = 48  # As per your requirement

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


# Prepare data
def get_batch(data, seq_len):
    for i in range(0, len(data) - seq_len, seq_len):
        src = data[i:i+seq_len]
        tgt = data[i+1:i+1+seq_len]
        yield src, tgt


epochs = 10
ninp = 48  # The dimension of your input feature
nhid = 64 #200  # Dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # Number of heads in nn.MultiheadAttention models
dropout = 0.1 # Dropout value
lr = 0.001

model = TransformerModel(ninp, nhead, nhid, nlayers, dropout).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# initialize the scheduler
scheduler = StepLR(optimizer, step_size=2000, gamma=0.95)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(SEQ_LEN).to(device)
    for batch, (src, tgt) in enumerate(get_batch(data, SEQ_LEN)):
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        if src.size(0) != SEQ_LEN:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
        output = model(src, src_mask)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(data) // SEQ_LEN, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
