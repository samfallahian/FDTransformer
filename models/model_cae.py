import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import helpers


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        """ Load model configurations """
        config = helpers.Config()
        cfg = config.from_json("model").cae
        cfg_training = config.from_json("training")
        self.cfg = cfg
        self.cfg_training = cfg_training

        """ layers """
        # Input layers
        self.input = nn.Linear(cfg.autoencoderUnits[0], cfg.autoencoderUnits[1])
        # encoder layer
        self.encoding = nn.Linear(cfg.autoencoderUnits[1], cfg.autoencoderUnits[2])
        # bottleneck layer
        self.bottleneck = nn.Linear(cfg.autoencoderUnits[2], cfg.autoencoderUnits[3])
        # decoder layer
        self.decoding = nn.Linear(cfg.autoencoderUnits[3], cfg.autoencoderUnits[4])

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def encode(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.encoding(x))
        x = F.relu(self.bottleneck(x))
        return x

    def decode(self, x):
        x = self.decoding(x)
        return x
