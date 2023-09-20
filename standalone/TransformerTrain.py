import time
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import wandb
import pandas as pd
from utils import helpers


class TrainTransformer:
    def __init__(self, model, device, dataset, lr=0.001, epochs=10, log_interval=50,
                 source_size=8, target_size=2, scheduler_step=1000,
                 batch_size=48, lr_gamma=0.95, is_wandb=True, save_directory="saved_models"):
        if is_wandb:
            wandb.init(project='Transformers')
            config = wandb.config
            config.batch_size = batch_size
            config.lr = lr

        # Helpers and Logging
        self.is_wandb = is_wandb
        self.debug = True
        self.logger = helpers.Log("transformer")
        self.log_interval = log_interval
        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        self.df_result = pd.DataFrame(
            columns=["epoch", "batch", "batch_data_point", "data_point", "loss", "time"])

        # Data
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        # Training
        self.epochs = epochs
        self.model = model
        print("Model initialized.")
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.criterion = nn.MSELoss()  # Define loss and optimizer
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=lr_gamma)


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.
            start_time = time.time()
            print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')

            for batch_idx, (coords_batch, source, target) in enumerate(self.dataloader):
                source, target = source.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(source, target)

                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                if batch_idx % self.log_interval == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time

                    print(f"| epoch {epoch+1}/{self.epochs} | {batch_idx} batches | lr {self.scheduler.get_last_lr()[0]:.6f} "
                          f"| ms/batch {elapsed * 1000 / self.log_interval:.2f} | loss {loss.item():.6f}")

            running_loss /= len(self.dataloader)
            if self.is_wandb:
                wandb.log({"loss": running_loss, "lr": self.scheduler.get_last_lr()[0]})

            print(f"End of epoch {epoch + 1} / {self.epochs}, Running loss {running_loss:.5f}, at {datetime.now().time().strftime('%H:%M:%S')}")
            print("-------------------------------------------------------------------")
            # Save the model
            if (epoch + 1) % self.save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': running_loss
                }, os.path.join(self.save_directory, f"transformer_checkpoint_{epoch + 1}.pth"))
        print("=================================================================")
        print(f'End of training at {datetime.now().time().strftime("%H:%M:%S")}')
        # print(self.df_result.head(20))
        # self.logger.save_result(self.df_result)
        if self.is_wandb:
            wandb.finish()
        # Save the final trained model
        torch.save(self.model.state_dict(), os.path.join(self.save_directory, f"transformer_final_saved_model_{datetime.now().date().strftime('%m%d%Y')}.pth"))
        return running_loss
