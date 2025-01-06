import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import time
from datetime import datetime
import os
from torch.cuda.amp import autocast, GradScaler
import pickle
from torch.optim.lr_scheduler import StepLR
from TransformerDataLoader import CustomDataset
from DataDecoder import DecodeData
import pandas as pd
from utils import helpers


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.ninp)
        output = self.transformer_encoder(src, src_mask)
        return output

class CustomDataset(Dataset):
    def __init__(self, data, seq_len, batch_src_seq, batch_tgt_seq):
        self.data = data
        self.seq_len = seq_len
        self.total_seq = batch_src_seq + batch_tgt_seq
        self.batch_src_seq = batch_src_seq

    def __len__(self):
        return (len(self.data) - self.total_seq * self.seq_len) + 1

    def __getitem__(self, idx):
        src = self.data[idx + self.seq_len: idx + (self.batch_src_seq - 1) * self.seq_len]
        tgt = self.data[idx + self.batch_src_seq * self.seq_len: idx + self.total_seq * self.seq_len]
        return src, tgt

class TrainTransformer:

    def load_tensor_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_tensor = pickle.load(f)
        return loaded_tensor

    def __init__(self, model, device, data_path="encoded_tensor.pickle",
                 lr=0.001, seq_len=48, epochs=10, log_interval=200,
                 batch_src_seq=9, batch_tgt_seq=1, scheduler_step=1000,
                 lr_gamma=0.95, is_wandb=True):

        self.debug = True
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = self.load_tensor_from_pickle(data_path).view(-1, 48)
        print(f"Data Shape {self.data.shape}")
        self.dataset = CustomDataset(self.data, seq_len, batch_src_seq, batch_tgt_seq)
        self.dataloader = DataLoader(self.dataset, batch_size=seq_len, shuffle=False)
        self.logger = helpers.Log("transformer")

        self.epochs = epochs
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.batch_src_seq = batch_src_seq
        self.batch_tgt_seq = batch_tgt_seq

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.criterion = nn.MSELoss() # Define loss and optimizer
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=lr_gamma) # initialize the scheduler

        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = "saved_models"  # Directory to save the models
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        self.df_result = pd.DataFrame(
            columns=["epoch", "batch", "batch_data_point", "data_point", "loss", "time"])

        self.decoder = DecodeData(device=device)

    def train(self):
        data_point_count = 0
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.
            running_loss = 0.
            start_time = time.time()
            print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')
            src_mask = self.model.generate_square_subsequent_mask(self.seq_len).to(self.device)
            batch_data_point_count = 0
            for batch, (src, tgt) in enumerate(self.dataloader):
                tgt = tgt.to(self.device)
                self.optimizer.zero_grad()
                src = src.to(self.device)
                if src.size(0) != self.seq_len:
                    src_mask = self.model.generate_square_subsequent_mask(src.size(0)).to(self.device)
                print("Target Size: ", tgt.size())
                print("Source Size: ", src.size())
                print("Mask Size: ", src_mask.size())

                output = self.model(src, src_mask)
                print("output Size: ", output.size())
                torch.autograd.set_detect_anomaly(True)
                loss = self.criterion(output, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                batch_data_point_count += 1
                data_point_count += 1

                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                running_loss += loss.item()
                if batch % self.log_interval == 0 and batch > 0:
                    cur_loss = total_loss / self.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.6f} | ms/batch {:5.2f} | '
                          'loss {:5.2f}'.format(
                        epoch + 1, batch, len(self.data) // self.seq_len, self.scheduler.get_last_lr()[0],
                        elapsed * 1000 / self.log_interval,
                        cur_loss))
                    total_loss = 0
                    start_time = time.time()
            running_loss /= len(self.dataloader)

            print(f'End of epoch {epoch + 1}, Running loss {running_loss:.2f}')
            # Save the model
            if epoch % self.save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': running_loss
                }, os.path.join(self.save_directory, f"transformer_checkpoint_{epoch}.pth"))
        print("===============================================")
        print(f'End of training at {datetime.now().time().strftime("%H:%M:%S")}')
        print(self.df_result.head(20))
        self.logger.save_result(self.df_result)
        return running_loss

# Define constants
seq_len = 48
epochs = 10 # 301
ninp = 48  # The dimension of your input feature
nhid = 128  # 200  # Dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # Number of heads in nn.MultiheadAttention models
dropout = 0.1
lr = 0.001
log_interval = 50
batch_src_seq = 9
batch_tgt_seq = 1
scheduler_step = 5000
lr_gamma = 0.97
data_path= "/mnt/d/sources/cgan/playground/convolutional/dataset/encoded_tensor.pickle"
is_wandb = False

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(device)
model = TransformerModel(ninp, nhead, nhid, nlayers, dropout).to(device)
trainer = TrainTransformer(model, device, data_path=data_path,
                 lr=lr, seq_len=seq_len, epochs=epochs, log_interval=log_interval,
                 batch_src_seq=9, batch_tgt_seq=batch_tgt_seq, scheduler_step=scheduler_step,
                 lr_gamma=lr_gamma, is_wandb=is_wandb)
loss = trainer.train()