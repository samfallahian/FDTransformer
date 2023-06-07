import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from ContractiveAutoencoder import ContractiveAutoencoder
device = torch.device("mps:0")


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item = item.reshape(-1).float()  # Flatten and convert to 32-bit float
        return item


# Define your ContractiveAutoencoder class here (including the model architecture and training-related methods)

# Define the path to your data file
data_file = '_data_train_autoencoder_flat.pickle'

# Load the data from the pickle file
with open(data_file, 'rb') as f:
    data = pickle.load(f)

# Print shapes of the first few tensors
for i, tensor in enumerate(data[:5]):
    print(f"Tensor {i}: shape {tensor.shape}")

# Create a dataset and data loader
dataset = MyDataset(data)


def collate_fn(batch):
    data = [item.to(device) for item in batch]  # Move each tensor to the target device
    data = torch.stack(data, dim=0)  # Now all tensors are on the same device
    return data





# Set up the training parameters
num_epochs = 1000
save_interval = 10
print_interval = 1
learning_rate = 0.001

# Initialize the contractive autoencoder
input_size = 375
hidden_size = 64
batch_size = 64
contraction_coefficient = 1e-3
autoencoder = ContractiveAutoencoder(input_size, hidden_size, contraction_coefficient)
autoencoder = autoencoder.to(device)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define the optimizer
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
total_steps = 0
for epoch in range(num_epochs):
    for i, data_batch in enumerate(dataloader):
        #data_batch = data_batch.reshape(data.size(0), -1)
        #data_batch = data_batch.reshape(data_batch.size(0), -1)
        #data_batch = data_batch.reshape(375)
        # Forward pass
        encoded, decoded = autoencoder(data_batch)

        # Calculate the loss
        loss = autoencoder.loss_function(data_batch, decoded, encoded)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_steps += 1

    # Print the loss and error every print_interval epochs
    if (epoch + 1) % print_interval == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Save the model every save_interval epochs
    if (epoch + 1) % save_interval == 0:
        torch.save(autoencoder.state_dict(), f'autoencoder_epoch_{epoch + 1}.pt')

# Save the final model
torch.save(autoencoder.state_dict(), 'autoencoder_final.pt')
