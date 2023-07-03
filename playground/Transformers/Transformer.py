import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import math
import pickle


def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


# train_inputs, val_inputs, train_targets, val_targets = Dataset("dataset/encoded_tensor.pickle")
# data = load_tensor_from_pickle("dataset/encoded_tensor.pickle")
data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle")

# Flatten each data point
data = data.view(data.shape[0], -1)

# Split into sequences of 11 (10 for input, 1 for target)
sequences = [data[i - 11:i] for i in range(11, len(data))]

# Split sequences into inputs and targets
inputs = [seq[:-1] for seq in sequences]
targets = [seq[-1] for seq in sequences]

# Convert to PyTorch tensors (saved pickle already is torch tensor)
inputs = torch.stack(inputs)
targets = torch.stack(targets)

# Split into training and validation
split_idx = int(len(inputs) * 0.8)  # 80% for training
train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
train_targets, val_targets = targets[:split_idx], targets[split_idx:]



# Preprocess the data


# Hyperparameters
input_dim = train_inputs.shape[-1]
output_dim = train_targets.shape[-1]

hidden_dim = 64
num_layers = 2
dropout = 0.1
lr = 0.001


class TransformerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(d_model=input_dim, nhead=1,
                                       num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers,
                                       dim_feedforward=hidden_dim, dropout=dropout)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc(output)


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
# Create the model
model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

# Training loop
num_epochs = 2  # Can be adjusted
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_inputs, train_inputs)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(val_inputs, val_inputs)
        val_loss = criterion(val_output, val_targets)

    # Print statistics
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}, Val Loss: {val_loss.item()}')