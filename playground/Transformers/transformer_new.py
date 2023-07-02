import torch
from torch.nn import Transformer
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set your hyperparameters
batch_size = 10  # reduce batch size
n_epochs = 10
seq_length = 10
input_dim = train_inputs.shape[-1]
output_dim = input_dim
embedding_dim = 64
nhead = 2
n_hidden = 128
n_layers = 2
dropout = 0.2
lr = 0.001

# instantiate your Transformer model
model = TransformerModel(input_dim, output_dim, nhead, embedding_dim, n_hidden, n_layers, dropout).to(device)

# setup the loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Convert the train and test datasets into DataLoader
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

accumulation_steps = 4  # Increase this number to have larger virtual batch size

# training loop
model.train()
for epoch in range(n_epochs):
    hidden = model.init_hidden(batch_size)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute model output
        output = model(inputs, hidden)
        loss = criterion(output, targets)

        # Normalize the loss because it is averaged over all observations in the batch
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step
            model.zero_grad()  # Reset gradients tensors

    print("Epoch: {}/{}.............".format(epoch, n_epochs), end=' ')
    print("Loss: {:.4f}".format(loss.item()))


import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, embedding_dim, n_hidden, n_layers, dropout):
        super(TransformerModel, self).__init__()

        # Input embedding layer
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        # Transformer block
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=n_hidden)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        # Output layer
        self.decoder = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)

        return x

    def init_hidden(self, batch_size):
        # Since the Transformer model doesn't require a hidden state, we return a dummy tensor
        return torch.zeros(1)
