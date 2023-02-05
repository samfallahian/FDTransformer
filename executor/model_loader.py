import torch
from utils import helpers
from datetime import datetime


class ModelLoader:
    def __init__(self):
        super(ModelLoader, self).__init__()
        config = helpers.Config()
        self.cfg = config.from_json("training")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.from_json("data").batch_size

    def save_model(self, dis_model, gen_model):
        torch.save(gen_model.state_dict(),
                   f"saved_models/{datetime.now()}-{self.cfg.model_file_name}-generator_model.pt")
        torch.save(dis_model.state_dict(),
                   f"saved_models/{datetime.now()}-{self.cfg.model_file_name}-discriminator_model.pt")

    def load_model(self, model, file_name):
        model.load_state_dict(torch.load(f"saved_models/{file_name}.pt"))
        return model.to(self.device)

    def generate_data(self, generator):
        noise = torch.rand(self.batch_size, self.cfg.n_input).to(self.device) * 2 - 1
        label = torch.ones(self.batch_size, 1).to(self.device)
        fake_data = generator(torch.cat((noise, label), 1))
        return fake_data

