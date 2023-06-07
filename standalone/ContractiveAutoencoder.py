import torch
import torch.nn as nn
device = torch.device("mps")


class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, contraction_coefficient):
        super(ContractiveAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, input_size)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.contraction_coefficient = contraction_coefficient

    def forward(self, x):
        #x = x.float()  # Convert input to 'float'
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded):
        x = x.float()  # Convert input to 'float'
        decoded = decoded.float()  # Convert decoded to 'float'

        reconstruction_loss = nn.MSELoss(reduction='mean')(x, decoded)
        jacobian_loss = torch.mean(torch.sum(torch.pow(self.jacobian(x), 2), dim=1))
        loss = reconstruction_loss + (self.contraction_coefficient * jacobian_loss)
        return loss

    def jacobian(self, x):
        n = x.size()[0]
        jacobian = torch.zeros(n, self.input_size).to(x.device)
        for i in range(n):
            x.requires_grad_(True)
            y = self.encoder(x)
            y.backward(torch.ones_like(y), retain_graph=True)
            jacobian[i] = x.grad.data
            x.grad.data.zero_()
        return jacobian



# Example usage
input_size = 125 * 3
hidden_size = 64
contraction_coefficient = 1e-3

# Create an instance of the contractive autoencoder
autoencoder = ContractiveAutoencoder(input_size, hidden_size, contraction_coefficient)

# Generate random input tensor
input_tensor = torch.randn(1, input_size)

# Forward pass
encoded, decoded = autoencoder(input_tensor)

# Calculate loss
loss = autoencoder.loss_function(input_tensor, decoded, encoded)

# Backward pass and optimization step
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()
