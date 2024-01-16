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
wandb.init(project="Adjusted L1 Lamda and Batch Size")

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
epochs = 50
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        hidden_dim1 = 200
        hidden_dim2 = 100
        hidden_dim3 = 50

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

def loss_function(recon_x, x, mu, logvar):
    total_elements = x.nelement() # total number of elements in the tensor
    weight_for_others = 1.0  # weight for all other elements

    # Creating a tensor of ones with the same size as input
    custom_weights = torch.ones(total_elements).to(x.device)

    # Assign higher weight to desired elements (1st, 4th, etc.)
    # These are the "x" values.
    indices_to_bias = list(range(0, total_elements, 3))  # get indices: 1st, 4th, ..
    weight_for_desired_elements = 1.2  # weight for desired elements
    for idx in indices_to_bias:
        custom_weights[idx] = weight_for_desired_elements

    # Finding the index for the 63rd triple assuming the triples are flattened in the tensor
    index_63rd_triple = 63 * 3  # each triple consists of 3 elements (x,y,z)

    # Assigning higher weight to the 63rd triple
    weight_for_63rd_triple = 1.5  # weight for 63rd triple
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
    MAE = weighted_loss.sum() / custom_weights.sum()

    # KL Divergence loss same as before
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1_lamda = 5

    total_loss = l1_lamda * MAE + KLD
    wandb.log({"MAE Loss": MAE.item(), "KLD Loss": KLD.item(), "Total Loss": total_loss.item()})
    wandb.log({"63rd Triple Loss": weighted_loss[index_63rd_triple:index_63rd_triple + 3].mean().item()})
    # the final loss is the sum of the MAE and KLD
    return MAE + KLD

def train(model, dataloader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters())
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
    train_data, validation_data = train_test_split(df_pandas, test_size=.970, random_state=42)
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