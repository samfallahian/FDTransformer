import torch
from models import cgan, loss
from utils import helpers


class Training:
    def __init__(self):
        super().__init__()
        """ Load training configurations """
        config = helpers.Config()
        cfg = config.from_json("training")
        """ Load model configurations """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = cgan.ANN()
        self.loss_function = loss.CustomLoss()
        """ Dynamic optimizer based on config """
        optimizer_function = getattr(torch.optim, cfg.optimzer)
        self.optimizer = optimizer_function(self.model.parameters(), lr=cfg.lr)

    def train(self):
        pass

    def test(self):
        pass
