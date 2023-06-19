import torch
import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, batch_size, latent_size=(8, 6)):
        super(ConvolutionalAutoencoder, self).__init__()

        #self.batch_size = batch_size
        self.latent_size = latent_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=8*15, out_features=8*6)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8 * 6, out_features=8 * 16),  # Input: (batch_size, 8*6), Output: (batch_size, 8*16)
            nn.Unflatten(1, (8, 16)),  # Input: (batch_size, 8*16), Output: (batch_size, 8, 16)
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Input: (batch_size, 8, 16), Output: (batch_size, 16, 33)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=2, padding=1),
            # Input: (batch_size, 16, 33), Output: (batch_size, 16, 65)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=2, padding=1),
            # Input: (batch_size, 16, 65), Output: (batch_size, 16, 129)
            nn.ReLU(),
            nn.Conv1d(16, 3, kernel_size=5, stride=1, padding=2),
            # Input: (batch_size, 16, 129), Output: (batch_size, 3, 125)
            nn.ReLU()
        )

        # Reconstruction loss
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded)
        encoded = encoded.view(x.size(0), *self.latent_size)  # Reshape the tensor to desired latent size
        # print(encoded)

        assert encoded.size()[
               1:] == self.latent_size, f"Encoder output size {encoded.size()[1:]} does not match the specified latent size {self.latent_size}."

        decoded = self.decoder(encoded.view(x.size(0), -1))

        assert decoded.size()[1:] == (
        3, 125), f"Decoder output size {decoded.size()[1:]} does not match the specified output size (3, 125)."

        return decoded, encoded


def print_model_architecture():
    model = ConvolutionalAutoencoder(batch_size=1)
    print(model)

if __name__ == '__main__':
    target_input_size = (1, 3, 125)
    target_latent_size = (8, 6)

    print("Model architecture:")
    print_model_architecture()

    model = ConvolutionalAutoencoder(batch_size=1, latent_size=target_latent_size)
    encoder_output = torch.zeros(target_input_size)
    decoded_output = model(encoder_output)

    assert encoder_output.size() == torch.Size(target_input_size), f"Encoder input size {encoder_output.size()} does not match the target input size {target_input_size}."
    assert decoded_output.size() == torch.Size((1, 3, 125)), f"Decoder output size {decoded_output.size()} does not match the target output size (1, 3, 125)."
