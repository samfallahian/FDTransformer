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
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, original_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

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