import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


num_files=15
file_prefix="/mnt/d/sources/cgan/standalone/dataset/latent_representation_for_"
all_dfs = []

for i in range(1, num_files+1):
    df = pd.read_pickle(f"{file_prefix}{i}.pkl.zip", compression="zip")
    step = len(df) // 3360
    sampled_df = df.iloc[::step].copy()
    sampled_df = sampled_df[['x', 'y', 'z', 'time', 'latent_representation']]
    sampled_df['latent_representation'] = sampled_df['latent_representation'].apply(lambda x: x[0])
    all_dfs.append(sampled_df)

# Concatenate all dataframes
df_combined = pd.concat(all_dfs, ignore_index=True)

# Pivot table to get v as a matrix with shape (num_samples, 1200, 48)
df_pivot = df_combined.pivot_table(index=['x','y','z'], columns='time', values='latent_representation', aggfunc='first')
print(df_pivot.shape)


class TimeSeriesDataset(Dataset):
    def __init__(self, df_pivot):
        self.df = df_pivot

    def __len__(self):
        return len(self.df) * (self.df.columns.size - 4)  # sliding window of 4

    def __getitem__(self, idx):
        row_idx = idx // (self.df.columns.size - 4)
        col_idx = idx % (self.df.columns.size - 4)

        src_values = [self.df.iloc[row_idx, col_idx + i] for i in range(4)]
        x = torch.tensor(np.array(src_values), dtype=torch.float32)

        y_values = self.df.iloc[row_idx, col_idx + 4]
        y = torch.tensor(np.array(y_values), dtype=torch.float32)

        return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# class Seq2PointTransformer(nn.Module):
#     def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
#         super(Seq2PointTransformer, self).__init__()
#         self.model_type = 'Transformer'
#
#         # Encoder part
#         self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
#         self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)
#
#         # Linear layer to produce the 1x48 output
#         self.decoder = nn.Linear(d_model, 48)
#
#     def forward(self, src):
#         # Get the output from the transformer encoder
#         encoder_output = self.transformer_encoder(src)
#
#         # Use the last timestep of the encoder output for prediction
#         output = self.decoder(encoder_output[-1])
#         return output

class Seq2PointTransformer(nn.Module):
    def __init__(self, nhead, num_encoder_layers, dim_feedforward, feature_size = 48, max_seq_len=5000):
        super(Seq2PointTransformer, self).__init__()

        self.embedding = nn.Linear(feature_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder = nn.Linear(feature_size, feature_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        encoder_output = self.transformer_encoder(src)
        output = self.decoder(encoder_output[-1])  # Get the output of the last time step
        return output

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        # Embedding layers
        self.encoder_embedding = nn.Embedding(ntoken, d_model)
        self.decoder_embedding = nn.Embedding(ntoken, d_model)

        # Positional Encoders
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, nhid, dropout)

        # Final Decoder
        self.fc_out = nn.Linear(d_model, ntoken)

        self.d_model = d_model
        self.ntoken = ntoken

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Embed and add positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)

        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = self.fc_out(output)
        return output

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for i, (src, tgt) in enumerate(loader):
        src = src.permute(1, 0, 2)
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# Hyperparameters
d_model = 48  # size of each velocity vector
nhead = 4  # number of heads in multihead attention
num_encoder_layers = 2  # number of encoder layers
dim_feedforward = 2048  # size of feedforward network in transformer
lr = 0.001  # learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
# model = Seq2PointTransformer(d_model, nhead, num_encoder_layers, dim_feedforward).to(device)
model = Seq2PointTransformer(nhead, num_encoder_layers, dim_feedforward).to(device)
for name, param in model.named_parameters():
    print(name, param.shape)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Create dataset and dataloader
dataset = TimeSeriesDataset(df_pivot)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 10
for epoch in range(epochs):
    loss = train(model, loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")