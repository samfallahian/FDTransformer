import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model")
        cfg_training = config.from_json("training")
        self.cfg = cfg
        self.cfg_training = cfg_training

        """ layers """
        self.autoencoderLayers = nn.ModuleDict()
        self.nLayers = len(cfg.autoencoderUnits)
        self.autoencoderLayers["input_layer"] = nn.Linear(cfg.autoencoderUnits[0], cfg.autoencoderUnits[1])
        # Hidden layers
        for i in range(1, len(cfg.autoencoderUnits) - 2):
            self.autoencoderLayers[f"fc_{i}"] = nn.Linear(cfg.autoencoderUnits[i], cfg.autoencoderUnits[i + 1])
            # self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        # output layer
        self.autoencoderLayers["output_layer"] = nn.Linear(cfg.autoencoderUnits[-2], cfg.autoencoderUnits[-1])

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def encode(self, x):
        x = F.relu(self.autoencoderLayers["input_layer"](x))
        x = F.dropout(x, p=self.cfg.dropout)
        print(int(self.nLayers / 2), self.nLayers - 3)
        for i in range(1, int(self.nLayers / 2) - 1):
            # x = self.autoencoderLayers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.relu(self.autoencoderLayers[f"fc_{i}"](x))
        return x

    def decode(self, x):
        for i in range(int(self.nLayers / 2), self.nLayers - 2):
            # x = self.autoencoderLayers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.relu(self.autoencoderLayers[f"fc_{i}"](x))
        x = torch.sigmoid(self.autoencoderLayers["output_layer"](x))
        x = self.autoencoderLayers["output_layer"](x)
        return x
