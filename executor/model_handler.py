import torch
from utils import helpers
from datetime import datetime
import numpy as np
from sklearn import preprocessing


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

    def generate_data(self, generator, sample, scalar):
        noise = torch.rand(sample, self.cfg_cgan.n_input).to(self.device) * 2 - 1
        generated_label = np.array([1, 1])
        label = np.tile(generated_label, (sample, 1))[:, :3]
        label_tensor= torch.tensor(label).to(self.device)

        # label_tensor = torch.from_numpy(label).to(self.device).unsqueeze(0)
        # print(label.shape)
        fake_data = generator(noise, label_tensor)
        print(fake_data)
        return scalar.inverse_transform(fake_data.detach().cpu())

