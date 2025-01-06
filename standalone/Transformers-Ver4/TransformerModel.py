import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# num_layers Number of encoder layers
# d_model Encoder layer dimension
# d_ff Feed forward dimension

class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_size=48, target_size=48, num_encoder_layers=2, d_model=128, d_ff=512, num_heads=8,
                 dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, target_size)

    def forward(self, src):
        # Project source input
        src = self.input_proj(src)

        # Run through encoder
        output = self.encoder(src)

        # Take the last embedded time step
        output = output[-1]

        # Project to target
        output = self.output_proj(output)

        return torch.tanh(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModelTarget(nn.Module):
    def __init__(self, input_size=48, ninp=48, nhead=8, nhid=2048, nlayers=6, dropout=0.1):
        super(TransformerModelTarget, self).__init__()
        from torch.nn import Transformer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.input_fc = nn.Linear(input_size, ninp)
        self.transformer = Transformer(d_model=ninp, nhead=nhead,
                                       num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers,
                                       dim_feedforward=nhid, dropout=dropout)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, input_size)

    def forward(self, src, tgt):
        src = self.input_fc(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        tgt = self.input_fc(tgt) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        # print(tgt.shape)
        # print(src.shape)
        # print(src[-1].shape)
        output = self.transformer(src, tgt)

        output = self.decoder(output)
        return output


class Seq2PointPosTransformer(nn.Module):
    def __init__(self, nhead=8, num_encoder_layers=2, dim_feedforward=2048, feature_size=48, max_seq_len=5000,
                 dropout=0.1):
        super(Seq2PointPosTransformer, self).__init__()

        self.embedding = nn.Linear(feature_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder = nn.Linear(feature_size, feature_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        encoder_output = self.transformer_encoder(src)
        output = self.decoder(encoder_output[-1])  # Get the output of the last time step
        return torch.tanh(output)


class CustomTransformer(nn.Module):
    def __init__(self, input_size=48, num_heads=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048):
        super(CustomTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(input_size, input_size)

    def forward(self, src, tgt):
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        return self.out(output)
