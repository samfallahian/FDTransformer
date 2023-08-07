import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)  # Output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc(output)