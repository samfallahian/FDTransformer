import pickle

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import wandb
import os
from sklearn.model_selection import train_test_split

# Initialize wandb
wandb.init(project="Small amount added to Minkowski Distance")
# Minkowski Distance Parameter
p_minkowski = 1.5

import matplotlib.pyplot as plt
import seaborn as sns



def generate_heatmap(raw_data, reconstruct_data):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Determine common color scale from raw_input
    vmin, vmax = raw_data.min(), raw_data.max()

    sns.heatmap(raw_data, ax=axs[0], vmin=vmin, vmax=vmax)
    axs[0].set_title('Raw Data')

    sns.heatmap(reconstruct_data, ax=axs[1], vmin=vmin, vmax=vmax)
    axs[1].set_title('Reconstructed Data')

    filename = "heatmap.png"
    plt.savefig(filename, dpi=600)
    plt.close(fig)
    return filename

original_dim = 375
latent_dim = 47
epochs = 500
batch_size = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        hidden_dim1 = 250
        hidden_dim2 = 150
        hidden_dim3 = 60

        # Encoder
        self.fc1 = nn.Linear(original_dim, hidden_dim1)
        self.relu1 = nn.Softplus()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc21 = nn.Linear(hidden_dim2, hidden_dim3)  # extra layer
        self.relu2 = nn.LeakyReLU()
        self.fc31 = nn.Linear(hidden_dim3, latent_dim)  # mu layer
        self.fc32 = nn.Linear(hidden_dim3, latent_dim)  # logvariance layer

        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim3)
        self.fc5 = nn.Linear(hidden_dim3, hidden_dim2)  # extra layer
        self.relu3 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_dim2, hidden_dim1)
        self.relu4 = nn.Softplus()
        self.fc7 = nn.Linear(hidden_dim1, original_dim)

    def encode(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        h3 = self.relu2(self.fc21(h2))
        return self.fc31(h3), self.fc32(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = self.relu3(self.fc4(z))
        h5 = self.relu3(self.fc5(h4))
        h6 = self.relu4(self.fc6(h5))
        return self.fc7(h6)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, original_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
def minkowski_distance(x, y, p, eps=1e-7):
    """
    Function to compute the Minkowski distance.
    """
    md = torch.pow(torch.sum(torch.pow(torch.abs(x - y) + eps, p)), 1/p)
    return md

def loss_function(recon_x, x, mu, logvar):
    # Continue with your previous code...
    # ...

    recon_x_grouped = recon_x.view(-1, int(recon_x.nelement()/3), 3)
    x_grouped = x.view(-1, int(x.nelement()/3), 3)

    triplet_minkowski = minkowski_distance(recon_x_grouped, x_grouped, p_minkowski)

    # Calculate MAE for grouped data
    triplet_mae = torch.max(torch.abs(recon_x_grouped - x_grouped), dim=2)[0]
    MAE = torch.mean(triplet_mae)

    # Calculating the mean, variance and standard deviation of triplet_mae
    minkowski_mean = torch.mean(triplet_minkowski)
    # Calculating the variance and standard deviation of triplet_mae
    variance = torch.var(triplet_mae)
    std_dev = torch.std(triplet_mae)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    l1_lamda = 1.0

    total_loss = l1_lamda * minkowski_mean + KLD + variance + std_dev

    # Logging the losses to wandb
    wandb.log({
        "Minkowski Loss": minkowski_mean.item(),
        "Variance Loss": variance.item(),
        "MAE Loss": MAE.item(),
        "Std Dev Loss": std_dev.item(),
        "KLD Loss": KLD.item(),
        "Total Loss": total_loss.item()
    })

    wandb.log({"63rd Triple Loss": triplet_minkowski.mean().item()})

    return total_loss

def train(model, dataloader, epochs):
    model.train()
    from torch.optim import Adam

# Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=1e-4)
    start_time_epochs = time.time()
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

        # After each epoch, randomly select 5 rows from the raw and reconstructed data,
        # generate a heatmap, and log it to wandb
        raw_data = batch.cpu().detach().numpy()
        reconstruct_data = recon_batch.cpu().detach().numpy()
        idx = np.random.choice(len(raw_data), 5)
        heatmap_filename = generate_heatmap(raw_data[idx], reconstruct_data[idx])

        # New code: print raw and reconstructed data to console
        print("Raw data for selected indices: \n", raw_data)
        print("Reconstructed data for selected indices: \n", reconstruct_data)

        # manual MSE calculation for all rows
        for i in range(len(raw_data)):
            spot_mse = np.mean((raw_data[i] - reconstruct_data[i]) ** 2)
            print(f'Raw data for index {i}: ', raw_data[i])
            print(f'Reconstructed data for index {i}: ', reconstruct_data[i])
            print(f'Manual spot MSE for index {i}: ', spot_mse)

        wandb.log({"heatmap": wandb.Image(heatmap_filename)})
        end_time = time.time()

    total_training_time = end_time - start_time_epochs

def main():
    # Path to your pickled dataframe
    pickle_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_data_for_training_AE.dataframe.pkl'

    # Load the pickled DataFrame
    with open(pickle_file_path, 'rb') as file:
        df_pandas = pickle.load(file)

    #TODO We will have to add official validation logic, but for now this will help
    # Split data into 80% train and 20% validation
    train_data, validation_data = train_test_split(df_pandas, test_size=.980, random_state=42)
    del validation_data
    del df_pandas
    df_pandas_truncated = train_data.iloc[:, 1:]
    data = df_pandas_truncated.values  # <--- This line is changed.
    data = torch.Tensor(data)  # convert to Tensor
    print("Dtypes in use in our dataframe are: " + str(data.dtype))
    # Check the data type of the tensor
    if data.dtype == torch.float16:
        # Convert to Float32
        data = data.float()
    data = data.to(device)

    vae = VAE().to(device)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    start_time_epochs = time.time()
    train(vae, dataloader, epochs)

if __name__ == "__main__":
    main()