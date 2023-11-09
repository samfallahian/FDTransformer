import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
#     def __init__(self, nhead, num_encoder_layers, dim_feedforward, feature_size = 48, max_seq_len=5000, dropout=0.1):
#         super(Seq2PointTransformer, self).__init__()
#
#         self.embedding = nn.Linear(feature_size, feature_size)
#         self.pos_encoder = PositionalEncoding(feature_size, max_len=max_seq_len)
#
#         encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         self.decoder = nn.Linear(feature_size, feature_size)
#
#     def forward(self, src):
#         src = self.embedding(src)
#         src = self.pos_encoder(src)
#         encoder_output = self.transformer_encoder(src)
#         output = self.decoder(encoder_output[-1])  # Get the output of the last time step
#         return output


class Seq2PointTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, feature_size = 48, dropout=0.1):
        super(Seq2PointTransformer, self).__init__()
        self.model_type = 'Transformer'

        # Encoder part
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_encoder_layers)

        # Linear layer to produce the 1x48 output
        self.decoder = nn.Linear(d_model, feature_size)

    def forward(self, src):
        # Get the output from the transformer encoder
        encoder_output = self.transformer_encoder(src)

        # Use the last timestep of the encoder output for prediction
        output = self.decoder(encoder_output[-1])
        return output

class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True
                                          , dropout=dropout)
        self.fc = nn.Linear(d_model, d_model)  # Output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return torch.tanh(self.fc(output))