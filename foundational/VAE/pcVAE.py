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
wandb.init(project="Wasserstein Autoencoder V04 Might be Working")
# Minkowski Distance Parameter
p_minkowski = 1.5

import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.functional as F



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
batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Main class definition for your model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Define the sizes for the hidden layers
        hidden_dim1 = 250  # Modify as needed
        hidden_dim2 = 150  # Modify as needed
        hidden_dim3 = 100  # Modify as needed

        # Encoder
        self.fc1 = nn.Linear(original_dim, hidden_dim1)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(hidden_dim3, latent_dim)

        # Decoder
        self.fc5 = nn.Linear(latent_dim, hidden_dim3)
        self.elu4 = nn.ELU()
        self.fc6 = nn.Linear(hidden_dim3, hidden_dim2)
        self.elu5 = nn.ELU()
        self.fc7 = nn.Linear(hidden_dim2, hidden_dim1)
        self.elu6 = nn.ELU()
        self.fc8 = nn.Linear(hidden_dim1, original_dim)

    def decode(self, z):
        h1 = self.elu4(self.fc5(z))
        h2 = self.elu5(self.fc6(h1))
        h3 = self.elu6(self.fc7(h2))
        return self.fc8(h3)

    def encode(self, x):
        h1 = self.elu1(self.fc1(x))
        h2 = self.elu2(self.fc2(h1))
        h3 = self.elu3(self.fc3(h2))
        return self.fc4(h3)

    def forward(self, x):
        mu = self.encode(x.view(-1, original_dim))
        return self.decode(mu), mu



    def loss_function(self, recon_x, x, mu):
        recon_x_grouped = recon_x.view(-1, int(recon_x.nelement()/3), 3)
        x_grouped = x.view(-1, int(x.nelement()/3), 3)

        '''
        # In case shapes are not the expected ones, print them out.
        if recon_x_grouped.shape[0] < 63 or x_grouped.shape[0] < 63:
            print('recon_x_grouped shape:', recon_x_grouped.shape)
            print('x_grouped shape:', x_grouped.shape)
        '''
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x_grouped, x_grouped)

        # MMD loss: enforcing prior = posterior in the latent space
        true_samples = torch.randn(mu.shape).to(device)
        mmd_loss = self._compute_mmd(mu, true_samples)

        # Previous code was assuming there are 63 examples / batch_size
        # Changed it to compute triplet loss tensor-wide
        # OR for the 63rd triplet of a random example if there are more than 63

        # Instead of [62] this should probably be something like [random_index, 62]
        # 'random_index' is choosing a random sample in the batch
        random_index = torch.randint(len(x_grouped), size=(1,)).item()
        triplet_63_loss = F.mse_loss(recon_x_grouped[random_index, 62], x_grouped[random_index, 62])

        # Log the losses to wandb
        wandb.log({"Reconstruction Loss": recon_loss.item(),
                   "MMD Loss": mmd_loss.item(),
                   "63rd Reconstruction": triplet_63_loss.item()})

        return recon_loss + mmd_loss + triplet_63_loss

    def _compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]
        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
        return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / dim * 2.0)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

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
            recon_batch, mu = model(batch)
            #print('Batch shape:', batch.shape)
            #print('Reconstructed batch shape:', recon_batch.shape)
            loss = model.loss_function(recon_batch, batch, mu) # Adjusted here
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