import torch
from ConvolutionalAutoencoder import ConvolutionalAutoencoder


class DecodeData:

    def __init__(self, device,
                 saved_model_path=r"/mnt/d/sources/cgan/playground/convolutional/saved_models/checkpoint_400.pth"):
        self.model = ConvolutionalAutoencoder().to(device)
        self.device = device
        self.saved_model_path = saved_model_path

    def decoded_tensor(self, encoded_data):
        print("encoded_data shape ", encoded_data.shape)
        print("size 0 ", encoded_data.size()[0])
        self.model.load_state_dict(torch.load(self.saved_model_path, map_location=torch.device(self.device))["model_state_dict"])
        # appended_tensor = torch.empty(0, 3, 125).float().to(self.device)
        # data_reshaped = encoded_data.view(256, 8, 8, 6)
        data_reshaped = encoded_data.reshape(256, 8, 8, 6)
        print("data_reshaped size", data_reshaped.shape)

        data_to_decode = data_reshaped[:, :, -1, :]
        print("data_reshaped size", data_to_decode.shape)
        decoded = self.model.decode(data_to_decode.float().to(self.device))
        decoded_data_reshaped = decoded.view(256, 8, -1)
        # for i, data in enumerate(encoded_data):
        #     decoded = self.model.decode(data.float().to(self.device))
        #     appended_tensor = torch.cat((appended_tensor, decoded), dim=0)
        #

        # encoded_data shape  torch.Size([256, 8, 48])
        # local decoded shape torch.Size([256, 3, 125])
        # appended_tensor shape torch.Size([10, 8, 1200])

        # I need to iterate over encoded_data shape and send 48=8*6 to encoder and get 3,125
        # then append it together


        print("local decoded shape", decoded.shape)
        print("decoded_data_reshaped shape", decoded_data_reshaped.shape)
        # return appended_tensor
        return decoded_data_reshaped
