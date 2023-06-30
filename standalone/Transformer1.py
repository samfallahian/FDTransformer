import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Transformer
import torch.optim as optim
import pickle


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


def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data = load_tensor_from_pickle(r"/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle")
data = load_tensor_from_pickle("dataset/encoded_tensor.pickle")

# Flatten each data point -> shape: 111000, 8, 6 => 111000, 48
# data = data.view(data.shape[0], -1)

# Or reshape data to have one feature per data point?
data = data.view(-1, 1)

# Split into sequences of 11 (10 for input, 1 for target)
sequences = [data[i - 11:i] for i in range(11, len(data))]

# Split sequences into inputs and targets
inputs = [seq[:-1] for seq in sequences]
targets = [seq[-1] for seq in sequences]

# Convert to PyTorch tensors
inputs = torch.stack(inputs).squeeze()
targets = torch.stack(targets).squeeze()

# Split into training and validation
split_idx = int(len(inputs) * 0.8)  # 80% for training
train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
train_targets, val_targets = targets[:split_idx], targets[split_idx:]

# Create DataLoaders
batch_size = 528
train_data = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data = TensorDataset(val_inputs, val_targets)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)

# Hyperparameters
input_dim = train_inputs.shape[-1]
output_dim = train_targets.shape[-1]
hidden_dim = 64
num_layers = 2
dropout = 0.1
lr = 0.001

# Define the model


# Create the model
model = TransformerModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        # Reshape inputs and targets for transformer model
        inputs = inputs.view(-1, batch_size, input_dim)
        targets = targets.view(-1, batch_size, output_dim)
        optimizer.zero_grad()
        output = model(inputs, inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Reshape inputs and targets for transformer model
            inputs = inputs.view(-1, batch_size, input_dim)
            targets = targets.view(-1, batch_size, output_dim)
            output = model(inputs, inputs)
            loss = criterion(output, targets)
            val_loss += loss.item()

    print(
        f'Epoch {epoch + 1}/{num_epochs}: train loss {train_loss / len(train_loader)}, val loss {val_loss / len(val_loader)}')

# https://github.com/KasperGroesLudvigsen/influenza_transformer/tree/main