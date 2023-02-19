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
        cfg_training = config.from_json("training")
        self.cfg = cfg
        self.cfg_training = cfg_training
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers """
        self.nLayers = len(cfg.generatorUnits)
        """Define input layer"""
        self.layers["input_layer"] = nn.Linear(cfg.generatorUnits[0], cfg.generatorUnits[1])
        """Define hidden layers and batch normalization"""
        for i in range(1, len(cfg.generatorUnits) - 2):
            self.layers[f"hidden_{i}"] = nn.Linear(cfg.generatorUnits[i], cfg.generatorUnits[i + 1])
            self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        """Define output layer"""
        self.layers["output_layer"] = nn.Linear(cfg.generatorUnits[-2], cfg.generatorUnits[-1])

    def forward(self, x, label):
        x = torch.cat((x, label), dim=1)
        x = F.leaky_relu(self.layers["input_layer"](x), negative_slope=self.cfg.negative_slope)
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nLayers - 2):
            x = self.layers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.leaky_relu(self.layers[f"hidden_{i}"](x), negative_slope=self.cfg.negative_slope)
        x = self.layers[f"output_layer"](x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model")
        self.cfg = cfg
        cfg_training = config.from_json("training")
        self.cfg_training = cfg_training
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers by reversing generator layer"""
        self.nLayers = len(cfg.discriminatorUnits)
        """Define input layer"""
        self.layers["input_layer"] = nn.Linear(cfg.discriminatorUnits[0], cfg.discriminatorUnits[1])
        """Define hidden layers and batch normalization"""
        for i in range(1, len(cfg.discriminatorUnits) - 2):
            self.layers[f"hidden_{i}"] = nn.Linear(cfg.discriminatorUnits[i], cfg.discriminatorUnits[i + 1])
            self.layers[f"batch_norm_{i}"] = nn.BatchNorm1d(cfg.discriminatorUnits[i])
        """Define output layer"""
        self.layers["output_layer"] = nn.Linear(cfg.discriminatorUnits[-2], cfg.discriminatorUnits[-1])

    def forward(self, x, label):
        x = torch.cat((x, label), dim=1)
        x = F.leaky_relu(self.layers["input_layer"](x), negative_slope=self.cfg.negative_slope)
        x = F.dropout(x, p=self.cfg.dropout)
        for i in range(1, self.nLayers - 2):
            x = self.layers[f"batch_norm_{i}"](x)
            x = F.dropout(x, p=self.cfg.dropout)
            x = F.leaky_relu(self.layers[f"hidden_{i}"](x), negative_slope=self.cfg.negative_slope)

        if self.cfg_training.is_critic:
            x = self.layers["output_layer"](x)
        else:
            x = torch.sigmoid(self.layers["output_layer"](x))
        return x
