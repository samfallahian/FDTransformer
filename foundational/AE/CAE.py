import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False  # Debug switch


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv1d(125, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(64, 32, kernel_size=2)
        self.enc_fc1 = nn.Linear(32 * 2, 48)  # 32 channels, 2 width
        self.enc_fc2 = nn.Linear(48, 48)  # [8, 6] latent vector

        # Decoder
        self.dec_fc1 = nn.Linear(48, 48)
        self.dec_fc2 = nn.Linear(48, 64)
        self.dec_conv1 = nn.ConvTranspose1d(32, 64, kernel_size=2)
        self.dec_conv2 = nn.Conv1d(64, 125, kernel_size=1)  # Adjusting the last decoding convolution layer


    def encoder(self, x):
        if DEBUG: print("Input Shape: ", x.shape)
        x = F.dropout(F.relu(self.enc_conv1(x)), p=0.05)
        if DEBUG: print("After enc_conv1 Shape: ", x.shape)
        x = F.dropout(F.relu(self.enc_conv2(x)), p=0.05)
        if DEBUG: print("After enc_conv2 Shape: ", x.shape)

        x = x.view(x.size(0), -1)  # flatten
        x = F.dropout(F.relu(self.enc_fc1(x)), p=0.05)
        if DEBUG: print("After enc_fc1 Shape: ", x.shape)

        x = F.tanh(self.enc_fc2(x))
        if DEBUG: print("After enc_fc2 Shape: ", x.shape)
        return x

    def decoder(self, x):
        if DEBUG: print("Latent Shape: ", x.shape)
        x = F.dropout(F.relu(self.dec_fc1(x)), p=0.05)
        if DEBUG: print("After dec_fc1 Shape: ", x.shape)

        x = F.dropout(F.relu(self.dec_fc2(x)), p=0.05)
        x = x.view(x.size(0), 32, 2)  # reshape back to feature map
        if DEBUG: print("Before dec_conv1 Shape: ", x.shape)

        x = F.dropout(F.relu(self.dec_conv1(x)), p=0.05)
        if DEBUG: print("After dec_conv1 Shape: ", x.shape)

        x = F.dropout(F.relu(self.dec_conv2(x)), p=0.05)
        if DEBUG: print("After dec_conv2 Shape: ", x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAE().to(device)
    model.eval()
    test_data = torch.randn((1000, 125, 3), dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(test_data)
        assert test_data.shape == outputs.shape, f"Expected {test_data.shape} but got {outputs.shape}"
        print("Input Shape: ", test_data.shape)
        print("Output Shape: ", outputs.shape)
