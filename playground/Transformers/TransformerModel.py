import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
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