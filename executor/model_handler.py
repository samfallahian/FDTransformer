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
        self.n_input = config.from_json("model").cgan.generatorUnits[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model(self, model, file_name):
        torch.save(model.state_dict(),
                   f"saved_models/{datetime.now().strftime('%Y-%m-%d %H%M%S')}-{file_name}.pt")

    def load_model(self, model, file_name):
        model.load_state_dict(torch.load(f"saved_models/{file_name}.pt"))
        return model.to(self.device)

    # def generate_data(self, generator, labels, scalar):
    #     noise = torch.rand(labels.shape[0], self.n_input).to(self.device) * 2 - 1
    #     label_tensor = torch.tensor(labels.to_numpy()).to(self.device)
    #     # label_tensor = torch.from_numpy(label).to(self.device).unsqueeze(0)
    #     fake_data = generator(noise, label_tensor)
    #     # result = torch.cat((label_tensor, scalar.inverse_transform(fake_data.detach())),1)
    #     return scalar.inverse_transform(fake_data.detach().cpu())
    #     # return result

    # def generate_data(self, generator, labels, scalar):
    #     for i in range(5):
    #         noise = torch.rand(1, self.n_input).to(self.device) * 2 - 1
    #         print(labels.to_numpy()[i])
    #         label_tensor = torch.tensor(labels.to_numpy()[i]).to(self.device)
    #         label_tensor = torch.reshape(label_tensor, (1,3))
    #         fake_data = generator(noise, label_tensor)
    #         # print(fake_data)
    #
    #     pass
        # return scalar.inverse_transform(fake_data.detach().cpu())

    def generate_data(self, generator, labels, scalar):
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(labels.shape[0], self.n_input).to(self.device) * 2 - 1
            label_tensor = torch.tensor(labels.to_numpy()).to(self.device)
            generated_data = generator(noise, label_tensor)
            transformed_generated = scalar.inverse_transform(generated_data.cpu().numpy())
            result = torch.cat((label_tensor, torch.tensor(transformed_generated).to(self.device)), dim=1)

        return result.detach().cpu().numpy()
