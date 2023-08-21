import torch
import torch.nn as nn
import math


class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True
                                          , dropout=dropout)
        self.fc = nn.Linear(d_model, d_model)  # Output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc(output)

    # def forward(self, src, tgt=None):
    #     if tgt is None:
    #         # Here, you'll implement the logic to generate the target sequence.
    #         # Typically, you'd start with a start-of-sequence token and then
    #         # generate the sequence token-by-token.
    #
    #         # Placeholder for generated (target_len = 2):
    #         tgt = torch.zeros((src.size(0), 2, src.size(2)), device=src.device)
    #
    #         for i in range(2):
    #             tgt_out = self.transformer(src, tgt)
    #             tgt_out = self.fc(tgt_out)
    #             # Take the last predicted token (or compute it based on your logic)
    #             next_token = tgt_out[:, i, :]
    #             tgt[:, i + 1, :] = next_token
    #
    #     else:
    #         tgt_out = self.transformer(src, tgt)
    #         tgt_out = self.fc(tgt_out)
    #
    #     return tgt_out