import time
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import pickle
import torch.nn as nn
import math
from torch.optim.lr_scheduler import StepLR
import wandb
from TransformersDataLoader import CustomDataset
from TransformerModel import TransformerModel


class Train_Transformer:
    # Trainer class for the Transformer model

    def load_tensor_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_tensor = pickle.load(f)
        return loaded_tensor

    def __init__(self, model, device, data_path="encoded_tensor.pickle",
                 lr=0.001, seq_len=48, epochs=10, log_interval=200,
                 batch_src_seq=9, batch_tgt_seq=1, scheduler_step=1000,
                 lr_gamma=0.95):
        # Initializes the model and the necessary parameters for training
        wandb.init(project='Transformers')
        config = wandb.config
        config.batch_size = (batch_src_seq + batch_tgt_seq) * seq_len
        config.lr = lr

        self.debug = True
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = self.load_tensor_from_pickle(data_path).view(-1, 48)
        print(f"Data Shape {self.data.shape}")
        self.dataset = CustomDataset(self.data, seq_len, batch_src_seq, batch_tgt_seq)
        self.dataloader = DataLoader(self.dataset, batch_size=seq_len, shuffle=False)

        self.epochs = epochs
        self.seq_len = seq_len
        self.log_interval = log_interval
        self.batch_src_seq = batch_src_seq
        self.batch_tgt_seq = batch_tgt_seq

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.criterion = nn.MSELoss() # Define loss and optimizer
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=lr_gamma) # initialize the scheduler

        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = "saved_models"  # Directory to save the models
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist

    def train(self):
        # Function to perform the training of the model
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.
            running_loss = 0.
            start_time = time.time()
            print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')
            src_mask = self.model.generate_square_subsequent_mask(self.seq_len).to(self.device)
            for batch, (src_batch, tgt) in enumerate(self.dataloader):
                # print("batch no ", batch, " len src", len(src_batch), " * ",len(src_batch[0]), " len target", len(tgt))
                tgt = tgt.to(self.device)
                self.optimizer.zero_grad()

                # with autocast():
                #     if src.size(0) != seq_len:
                #         src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
                #     output = model(src, src_mask)
                #     loss = criterion(output, tgt)
                # # Backward pass and optimization
                # scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # # Unscales the gradients of optimizer's assigned params in-place
                # scaler.unscale_(optimizer)
                # # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                # # Unscales gradients and calls or skips optimizer.step()
                # scaler.step(optimizer)
                # # Updates the scale for next iteration
                # scaler.update()
                # scheduler.step()

                # if src.size(0) != seq_len:
                #     src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)
                # output = model(src, src_mask)
                # loss = criterion(output, tgt)
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                for src in src_batch:
                    src = src.to(self.device)
                    if src.size(0) != self.seq_len:
                        src_mask = self.model.generate_square_subsequent_mask(src.size(0)).to(self.device)
                    output = self.model(src, src_mask)
                    loss = self.criterion(output, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
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
            wandb.log({"loss": running_loss})
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
        wandb.finish()
        return running_loss
