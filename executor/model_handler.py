import torch
from utils import helpers
from datetime import datetime


class ModelHandler:
    def __init__(self):
        super(ModelHandler, self).__init__()
        config = helpers.Config()
        self.cfg_cgan = config.from_json("training").cgan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.from_json("data").batch_size

    def save_model(self, model, file_name):
        torch.save(model.state_dict(),
                   f"saved_models/{datetime.now().strftime('%Y-%m-%d %H%M%S')}-{file_name}.pt")

    def load_model(self, model, file_name):
        model.load_state_dict(torch.load(f"saved_models/{file_name}.pt"))
        return model.to(self.device)

    def generate_data(self, generator):
        noise = torch.rand(self.batch_size, self.cfg_cgan.n_input).to(self.device) * 2 - 1
        label = torch.ones(self.batch_size, 1).to(self.device)
        fake_data = generator(torch.cat((noise, label), 1))
        return fake_data

