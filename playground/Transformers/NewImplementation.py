import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import pickle

class CustomDataset(Dataset):
    def __init__(self, data, source_size = 8, target_size = 2):
        self.data = data
        self.source_size = source_size
        self.target_size = target_size
        self.chunk_size = source_size + target_size

    def __len__(self):
        # Number of chunks we can make
        return len(self.data) // self.chunk_size

    def __getitem__(self, idx):
        # Start index of chunk
        start_idx = idx * self.chunk_size

        # Splitting the chunk into source and target
        source = self.data[start_idx:start_idx + self.source_size]
        target = self.data[start_idx + self.source_size:start_idx + self.chunk_size]

        return source, target

class TransformerSeq2Seq(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerSeq2Seq, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)  # Simple output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc(output)









def load_tensor_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_tensor = pickle.load(f)
    return loaded_tensor




# Hyperparameters
d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2
learning_rate = 0.001
epochs = 50
batch_size = 48
data_path= "/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle"
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)

data_tensor = load_tensor_from_pickle(data_path).view(-1, 48)
print(f"Data Shape {data_tensor.shape}")

dataset = CustomDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print(len(dataloader))


# Model, Loss, Optimizer
model = TransformerSeq2Seq(d_model, nhead, num_encoder_layers, num_decoder_layers).to(device)
criterion = nn.MSELoss()  # You can change the loss function based on your needs
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_idx, (source, target) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(source, target)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print logs
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

