import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)  # output: (16, 125)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)  # output: (8, 125)

        # Decoder
        self.conv_transpose1 = nn.ConvTranspose1d(8, 16, kernel_size=3, stride=1, padding=1)  # output: (16, 125)
        self.conv_transpose2 = nn.ConvTranspose1d(16, 3, kernel_size=3, stride=1, padding=1)  # output: (3, 125)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv_transpose1(x))
        x = self.conv_transpose2(x)
        return x

    def loss_function(self, *args, **kwargs):
        criterion = nn.MSELoss()
        return criterion(*args, **kwargs)
