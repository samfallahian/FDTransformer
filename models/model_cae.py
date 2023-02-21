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
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers """
        self.nLayers = len(cfg.caeUnits)
        """Define input layer"""
        self.layers["input_layer"] = nn.Linear(cfg.caeUnits[0], cfg.caeUnits[1])
        """Define hidden layers and batch normalization"""
        for i in range(1, len(cfg.caeUnits) - 2):
            self.layers[f"hidden_{i}"] = nn.Linear(cfg.caeUnits[i], cfg.caeUnits[i + 1])
            # self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        """Define output layer"""
        self.layers["output_layer"] = nn.Linear(cfg.caeUnits[-2], cfg.caeUnits[-1])

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        x = torch.cat((x, label), dim=1)
        x = F.relu(self.layers["input_layer"](x))
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nLayers - 2):
            # x = self.layers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.relu(self.layers[f"hidden_{i}"](x))
        x = self.layers[f"output_layer"](x)
        return x

    def encoder(self, x):
        x = F.leaky_relu(self.layers["input_layer"](x), negative_slope=self.cfg.negative_slope)
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nLayers - 2):
            # x = self.layers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.leaky_relu(self.layers[f"hidden_{i}"](x), negative_slope=self.cfg.negative_slope)
        x = self.layers[f"output_layer"](x)
        return x

