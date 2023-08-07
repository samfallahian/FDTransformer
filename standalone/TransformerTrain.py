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
from DataDecoder import DecodeData
import pandas as pd
from utils import helpers

class TrainTransformer:
    # Trainer class for the Transformer model

    def load_tensor_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            loaded_tensor = pickle.load(f)
        return loaded_tensor

    def __init__(self, model, device, data_path="encoded_tensor.pickle",
                 lr=0.001, epochs=10, log_interval=50,
                 source_size = 8, target_size = 2, scheduler_step=1000,
                 batch_size = 48, lr_gamma=0.95, is_wandb=True):

        if is_wandb:
            wandb.init(project='Transformers')
            config = wandb.config
            config.batch_size = batch_size
            config.lr = lr

        self.debug = True
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = self.load_tensor_from_pickle(data_path).view(-1, 48)
        print(f"Data Shape {self.data.shape}")
        self.dataset = CustomDataset(self.data, source_size, target_size)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.logger = helpers.Log("transformer")

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval

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
                # target = target.to(self.device)
                # source = source.to(self.device)

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

                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                          'lr {:02.6f} | ms/batch {:5.2f} | '
                          'loss {:5.4f}'.format(
                        epoch + 1, batch_idx, len(self.dataloader), self.scheduler.get_last_lr()[0],
                        elapsed * 1000 / self.log_interval,
                        loss.item()))
                    start_time = time.time()
            running_loss /= len(self.dataloader)
            if self.is_wandb:
                wandb.log({"loss": running_loss, "lr": self.scheduler.get_last_lr()[0]})

            print(f'End of epoch {epoch + 1}, Running loss {running_loss:.4f}')
            # Save the model
            # if epoch % self.save_interval == 0:
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'loss': running_loss
            #     }, os.path.join(self.save_directory, f"transformer_checkpoint_{epoch}.pth"))
        print("===============================================")
        print(f'End of training at {datetime.now().time().strftime("%H:%M:%S")}')
        # print(self.df_result.head(20))
        # self.logger.save_result(self.df_result)
        if self.is_wandb:
            wandb.finish()
        return running_loss
