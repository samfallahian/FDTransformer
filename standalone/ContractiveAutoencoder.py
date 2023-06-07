import torch
import torch.nn as nn

class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_size=375, latent_size=17):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def jacobian(self, x):
        n = x.size()[0]
        x.requires_grad_(True)
        y = self.encoder(x)
        y.backward(torch.ones_like(y), retain_graph=True)
        return x.grad.data

    def loss_function(self, input_tensor, decoded, encoded):
        mse_loss = nn.MSELoss()(decoded, input_tensor)
        jacobian_loss = torch.mean(torch.sum(torch.pow(self.jacobian(input_tensor), 2), dim=1))
        return mse_loss + jacobian_loss
