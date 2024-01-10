import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time

original_dim = 375
latent_dim = 47
epochs = 50
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(original_dim, 200)
        self.fc21 = nn.Linear(200, latent_dim)  # mu layer
        self.fc22 = nn.Linear(200, latent_dim)  # logvariance layer

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, original_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, original_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    total_elements = x.nelement() # total number of elements in the tensor
    weight_for_others = 1.0  # weight for all other elements

    # Creating a tensor of ones with the same size as input
    custom_weights = torch.ones(total_elements).to(x.device)

    # Finding the index for the 63rd triple assuming the triples are flattened in the tensor
    index_63rd_triple = 63 * 3  # each triple consists of 3 elements (x,y,z)

    # Assigning higher weight to the 63rd triple
    weight_for_63rd_triple = 10.0  # weight for 63rd triple
    custom_weights[index_63rd_triple:index_63rd_triple + 3] = weight_for_63rd_triple

    # Reshaping custom_weights to match original tensor shape
    custom_weights = custom_weights.view_as(x)

    # Deriving L1 loss, which is absolute difference between reconstructed and actual data
    loss = torch.abs(recon_x - x.view(-1, original_dim))

    # Applying weights to individually calculated losses
    weighted_loss = loss * custom_weights

    # Mean of the weighted losses.
    # This acts like an expectation since we are summing over the losses of all entries and dividing by the total entries.
    # And as the weights are normalized the sum of all weights is equal to the total entries.
    # So, the mean here would be equivalent to expectation if the weights were probabilities (sum to 1)
    L1 = weighted_loss.sum() / custom_weights.sum()

    # KL Divergence loss same as before
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # the final loss is the sum of the MSE and KLD
    return L1 + KLD

def train(model, dataloader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(dataloader.dataset)}')
        print(f'Epoch Time: {epoch_time} seconds')
    total_training_time = end_time - start_time_epochs
    print('Total training time: {} seconds'.format(total_training_time))

def main():
    data = pd.read_csv('/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/5p2_for_testing.csv')
    data = data.iloc[:, 1:]  # Exclude the first column
    data = data.astype(np.float32)
    data = data.values  # Convert to np array
    data = data.reshape((len(data), np.prod(data.shape[1:])))
    data = torch.Tensor(data)  # convert to Tensor
    data = data.to(device)
    vae = VAE().to(device)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    start_time_epochs = time.time()
    train(vae, dataloader, epochs)

if __name__ == "__main__":
    main()