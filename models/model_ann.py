import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model")
        """ Create dictionary to store the layers """
        self.layers = nn.ModuleDict()
        """ Define layers """
        self.layers["input"] = nn.Linear(cfg.nInput, cfg.nUnits[0])
        self.nLayers = len(cfg.nUnits)
        for i in range(self.nLayers):
            if i < self.nLayers - 1:
                self.layers[f"hidden{i}"] = nn.Linear(cfg.nUnits[i], cfg.nUnits[i + 1])
            else:
                self.layers["output"] = nn.Linear(cfg.nUnits[i], cfg.nOutput)

    def forward(self, x):
        x = self.layers["input"](x)
        for i in range(self.nLayers-1):
            x = F.relu(self.layers[f"hidden{i}"](x))
        x = self.layers["output"](x)

        return x
