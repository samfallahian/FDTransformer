import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from datetime import datetime
import os
from torch.cuda.amp import autocast, GradScaler
import pickle
import math
from torch.optim.lr_scheduler import StepLR
import wandb
import pandas as pd
from utils import helpers


def load_data(file_pattern, num_files):
    data_by_coords = defaultdict(list)

    for i in range(1, num_files + 1):
        filename = file_pattern.format(i)
        data = torch.load(filename)

        for entry in data:
            coords = tuple(entry['coordinates'])
            answer = entry['answer'].squeeze(0)  # Reshape [1, 8, 6] to [8, 6]
            data_by_coords[coords].append(answer)

    return data_by_coords


data = load_data("/mnt/d/sources/cgan/playground/dataset/3p6_time_{}.torch", 10)

print(len(data))
print(len(data[-117, -76, -25]))
print(data[-117, -76, -25][0].shape)


class SequenceDataset(Dataset):
    def __init__(self, data_by_coords, source_len=8, target_len=2):
        self.data_by_coords = data_by_coords
        self.source_len = source_len
        self.target_len = target_len
        self.total_len = source_len + target_len

        # Flatten all the data into sequences and keep track of the coordinates
        self.sequences = []
        for coord, time_series in self.data_by_coords.items():
            for i in range(len(time_series) - self.total_len + 1):
                self.sequences.append((coord,
                                       time_series[i:i + self.source_len],
                                       time_series[i + self.source_len:i + self.source_len + target_len]))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        coords, source_sequence, target_sequence = self.sequences[idx]

        # Convert each sequence into a single tensor
        source = torch.stack(source_sequence).view(self.source_len, -1)
        target = torch.stack(target_sequence).view(self.target_len, -1)

        # Convert coordinates to tensor
        coords_tensor = torch.tensor(coords).float()

        return coords_tensor, source, target




dataset = SequenceDataset(data)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# i = 0
#
# for coords_batch, source, target in data_loader:
#     i += 1
#     print(i)
#     if i == 1:
#         print(len(coords_batch))
#         print(coords_batch[0])
#         print(coords_batch)
#         print(source.shape)
#         print(target.shape)
#         print(source.size(), target.size())
#         break



class TransformerModel(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True
                                          , dropout=dropout)
        self.fc = nn.Linear(d_model, d_model)  # Output layer

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc(output)


class TrainTransformer:
    # Trainer class for the Transformer model

    def load_tensor_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_tensor = pickle.load(f)
        return loaded_tensor

    def __init__(self, model, device,data_loader,
                 lr=0.001, epochs=10, log_interval=50,
                 source_size=8, target_size=2, scheduler_step=1000,
                 batch_size=48, lr_gamma=0.95, is_wandb=True, save_directory="saved_models"):

        if is_wandb:
            wandb.init(project='Transformers')
            config = wandb.config
            config.batch_size = batch_size
            config.lr = lr

        self.debug = True
        self.model = model
        print("Model initialized.")
        self.device = device
        # self.data = self.load_tensor_from_pickle(data_path).view(-1, 48)
        # print(f"Data Shape {self.data.shape}")
        # self.dataset = CustomDataset(self.data, source_size, target_size)
        # self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.dataloader = data_loader
        self.logger = helpers.Log("transformer")

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.criterion = nn.MSELoss()  # Define loss and optimizer
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=lr_gamma)  # initialize the scheduler

        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        self.df_result = pd.DataFrame(
            columns=["epoch", "batch", "batch_data_point", "data_point", "loss", "time"])

        # self.decoder = DecodeData(device=device)
        self.is_wandb = is_wandb

    def train(self):
        data_point_count = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.
            running_loss = 0.
            start_time = time.time()
            print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')

            batch_data_point_count = 0
            for batch_idx, (source, target) in enumerate(self.dataloader):
                target = target.to(self.device)
                source = source.to(self.device)

                self.optimizer.zero_grad()

                output = self.model(source, target)

                # torch.autograd.set_detect_anomaly(True)

                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                batch_data_point_count += 1
                data_point_count += 1

                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time

                    print('| epoch {:3d}/{:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.6f} | ms/batch {:5.2f} | '
                          'loss {:6.5f}'.format(
                        epoch + 1, self.epochs, batch_idx, len(self.dataloader), self.scheduler.get_last_lr()[0],
                        elapsed * 1000 / self.log_interval,
                        loss.item()))
                    start_time = time.time()
            running_loss /= len(self.dataloader)
            if self.is_wandb:
                wandb.log({"loss": running_loss, "lr": self.scheduler.get_last_lr()[0]})

            print(f'End of epoch {epoch + 1} / {self.epochs}, Running loss {running_loss:.5f}')
            # Save the model
            if epoch + 1 % self.save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': running_loss
                }, os.path.join(self.save_directory, f"transformer_checkpoint_{epoch + 1}.pth"))
        print("===============================================")
        print(f'End of training at {datetime.now().time().strftime("%H:%M:%S")}')
        # print(self.df_result.head(20))
        # self.logger.save_result(self.df_result)
        if self.is_wandb:
            wandb.finish()
        return running_loss

d_model = 48
nhead = 6
num_encoder_layers = 2
num_decoder_layers = 2
learning_rate = 0.001
epochs = 301
batch_size = 48
dropout = 0.1
lr = 0.001
log_interval = 50
scheduler_step = 5000
lr_gamma = 0.97
source_size = 8
target_size = 2
is_wandb = False

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(d_model, nhead, num_encoder_layers, num_decoder_layers, dropout).to(device)
trainer = TrainTransformer(model, device, data_loader=data_loader,
                 lr=lr, epochs=epochs, log_interval=log_interval,
                 source_size = source_size, target_size = target_size, scheduler_step=scheduler_step,
                 batch_size = batch_size, lr_gamma=lr_gamma, is_wandb=is_wandb)


# loss = trainer.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

i=0
# Training loop
for epoch in range(20):
    for coords_batch, source, target in data_loader:
        source, target = source.to(device), target.to(device)
        i +=1

        if i == 1:
            print("dataloader length", len(data_loader))
            print("batch length", len(source))
            print("src size", source.shape)
            print("tgt size", target.shape)

        optimizer.zero_grad()
        outputs = model(source, target)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.6f}")
