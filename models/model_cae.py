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

        """ Encoder layers """
        self.encoderLayers = nn.ModuleDict()
        self.nEncoderLayers = len(cfg.encoderUnits)
        self.encoderLayers["input_layer"] = nn.Linear(cfg.encoderUnits[0], cfg.encoderUnits[1])
        # Hidden layers
        for i in range(1, len(cfg.encoderUnits) - 2):
            self.encoderLayers[f"hidden_{i}"] = nn.Linear(cfg.encoderUnits[i], cfg.encoderUnits[i + 1])
            # self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        # output layer
        self.encoderLayers["output_layer"] = nn.Linear(cfg.encoderUnits[-2], cfg.encoderUnits[-1])

        """ Decoder layers """
        self.decoderLayers = nn.ModuleDict()
        self.nDecoderLayers = len(cfg.decoderUnits)
        self.decoderLayers["input_layer"] = nn.Linear(cfg.decoderUnits[0], cfg.decoderUnits[1])
        # Hidden layers
        for i in range(1, len(cfg.decoderUnits) - 2):
            self.decoderLayers[f"hidden_{i}"] = nn.Linear(cfg.decoderUnits[i], cfg.decoderUnits[i + 1])
            # self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        # output layer
        self.decoderLayers["output_layer"] = nn.Linear(cfg.decoderUnits[-2], cfg.decoderUnits[-1])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encoder(self, x):
        x = F.relu(self.encoderLayers["input_layer"](x))
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nEncoderLayers - 2):
            # x = self.encoderLayers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.relu(self.encoderLayers[f"hidden_{i}"](x))
        x = F.relu(self.encoderLayers["output_layer"](x))
        return x

    def decoder(self, x):
        x = F.relu(self.decoderLayers["input_layer"](x))
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nDecoderLayers - 2):
            # x = self.decoderLayers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.relu(self.decoderLayers[f"hidden_{i}"](x))
        x = torch.sigmoid(self.decoderLayers["output_layer"](x))
        return x

