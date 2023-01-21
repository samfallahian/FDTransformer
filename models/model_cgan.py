import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model")
        self.cfg = cfg
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers """
        self.nLayers = len(cfg.generatorUnits)
        for i in range(len(cfg.generatorUnits)-1):
            self.layers[f"layer_{i + 1}"] = nn.Linear(cfg.generatorUnits[i], cfg.generatorUnits[i + 1])

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        for i in range(self.nLayers - 1):
            if i == (self.nLayers - 1):
                x = self.layers[f"layer_{i + 1}"](x)
            else:
                x = F.leaky_relu(self.layers[f"layer_{i + 1}"](x), negative_slope=self.cfg.negative_slope)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model")
        self.cfg = cfg
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers by reversing generator layer"""
        self.nLayers = len(cfg.discriminatorUnits)
        for i in range(len(cfg.discriminatorUnits) - 1):
            self.layers[f"layer_{i + 1}"] = nn.Linear(cfg.discriminatorUnits[i], cfg.discriminatorUnits[i + 1])

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        for i in range(self.nLayers - 1):
            if i == (self.nLayers - 1):
                x = F.sigmoid(self.layers[f"layer_{i + 1}"](x))
            else:
                x = F.leaky_relu(self.layers[f"layer_{i + 1}"](x), negative_slope=self.cfg.negative_slope)
        return x
