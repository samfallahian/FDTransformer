import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define a function to compute the Jacobian matrix
def calc_Jacobian(encoder, x):
    x.requires_grad_(True)
    encoded = encoder(x)
    num_outputs = np.prod(encoded.shape[1:])
    Jacobian = torch.zeros(encoded.shape[0], num_outputs, x.shape[-1])
    for i in range(encoded.shape[0]):
        gradients = torch.autograd.grad(outputs=encoded[i,:], inputs=x, allow_unused=True, grad_outputs=torch.ones_like(encoded[i,:]), create_graph=True, retain_graph=True)[0]
        Jacobian[i,:,:] = gradients.view(-1, num_outputs, x.shape[-1])[0,:,:]
    return Jacobian
# Instantiate the model
model = ContractiveAutoencoder(input_size=375, hidden_size=128, latent_size=10)

# Generate random data for training
num_samples = 1000
data = np.random.rand(num_samples, 375)
data = torch.from_numpy(data).float()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for 100 epochs
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, data)
    # Add the contractive regularization term
    Jacobian = calc_Jacobian(model.encoder, data)
    Jacobian_norm = torch.norm(Jacobian, dim=(1, 2))
    contractive_loss = torch.mean(Jacobian_norm ** 2)
    loss += 0.1 * contractive_loss
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print loss and save the model every 10 epochs
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        torch.save(model.state_dict(), 'contractive_autoencoder.pth')
