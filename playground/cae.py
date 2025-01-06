import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define hyperparameters
num_epochs = 50
batch_size = 64
learning_rate = 0.01
weight_decay = 1e-5
contractive_coef = 1e-3

# Load data into PyTorch dataset
data = np.loadtxt('your_data_file.csv', delimiter=',')
data_tensor = torch.from_numpy(data).float()
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Define Contractive Autoencoder architecture
class ContractiveAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ContractiveAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def contractive_loss(self, encoded, x):
        jacobian = torch.autograd.functional.jacobian(self.encoder, x)
        jacobian_norm = torch.norm(jacobian, dim=(0, 2))

        return torch.mean(jacobian_norm ** 2 * encoded ** 2)


# Initialize model, optimizer and loss function
model = ContractiveAutoencoder(input_dim=12, hidden_dim=6)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

# Train model
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Load batch of data
        data_batch = batch[0]

        # Forward pass
        encoded, decoded = model(data_batch)
        loss = loss_fn(decoded, data_batch) + contractive_coef * model.contractive_loss(encoded, data_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss after every epoch
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Test model
test_data = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float)
encoded, decoded = model(test_data)
print('Original input: {}'.format(test_data))
print('Encoded output: {}'.format(encoded))
print('Decoded output: {}'.format(decoded))
