import torch
from HybrdidAutoencoder import HybrdidAutoencoder


class DecodeData:

    def __init__(self, device,
                 saved_model_path="/mnt/d/sources/cgan/playground/convolutional/saved_models/checkpoint_300.pth"):
        self.model = HybrdidAutoencoder().to(device)
        self.device = device
        self.saved_model_path = saved_model_path

    def decoded_tensor(self, encoded_data):
        self.model.load_state_dict(torch.load(self.saved_model_path)["model_state_dict"])
        appended_tensor = torch.empty(0, 3, 125).float().to(self.device)

        for i, data in enumerate(encoded_data):
            decoded = self.model.decode(data.float().to(self.device))
            appended_tensor = torch.cat((appended_tensor, decoded), dim=0)
        return appended_tensor
