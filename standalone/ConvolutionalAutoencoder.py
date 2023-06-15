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

        # Additional Convolutional Layers for Downsampling and Upsampling
        self.downsample = nn.Conv1d(8, 8, kernel_size=2, stride=2)  # output: (8, 62)
        self.upsample = nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2)  # output: (8, 125)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Additional Downsampling Step
        x = self.downsample(x)

        # Store the encoded representation
        encoded = x

        # Additional Upsampling Step
        x = self.upsample(x)

        x = F.relu(self.conv_transpose1(x))
        x = self.conv_transpose2(x)

        # Adjust the output size to match input size
        x = F.pad(x, (0, 1))  # Pad the last dimension with zeros

        return x, encoded  # Return both the reconstruction and encoded tensors


    def criterion(self, inputs, outputs):
        batch_size = inputs.size(0)
        loss = F.mse_loss(inputs, outputs, reduction='none')
        loss = loss.view(batch_size, -1).mean(dim=1)
        return loss.mean()

