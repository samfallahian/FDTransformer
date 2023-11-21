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
import torch.nn.functional as F


class TrainTransformer:
    def __init__(self, model, device, dataset, lr=0.001, epochs=10, log_interval=50,
                 scheduler_step=1000, batch_size=256, lr_gamma=0.95,
                 is_wandb=True, save_directory="saved_models", kind=1):
        if is_wandb:
            wandb.init(project='Transformers')
            config = wandb.config
            config.batch_size = batch_size
            config.lr = lr

        # Helpers and Logging
        self.kind = kind
        self.is_wandb = is_wandb
        self.debug = True
        self.logger = helpers.Log("transformer")
        self.log_interval = log_interval
        self.save_interval = 100  # Save the model every 100 epochs
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)  # Create the save directory if it doesn't exist
        # self.df_result = pd.DataFrame(
        #     columns=["epoch", "batch", "batch_data_point", "data_point", "loss", "time"])

        # Data
        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Training
        self.epochs = epochs
        self.model = model
        print("Model initialized.")
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.mse = nn.MSELoss()  # Define loss and optimizer
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=lr_gamma)

    def loss_function(self, output, target):
        MSE = self.mse(output, target)
        output_normalized = F.softmax(output, dim=1)
        target_normalized = F.softmax(target, dim=1)
        # KLD = torch.sum(target * (torch.log(target) - torch.log(output)))
        KLD = torch.sum(

            target_normalized * (torch.log(target_normalized + 1e-10) - torch.log(output_normalized + 1e-10)))

        total_loss = MSE + self.beta * KLD
        return total_loss

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.

            start_time = time.time()
            print(f'start of epoch {epoch + 1} at {datetime.now().time().strftime("%H:%M:%S")}')
            i = 0
            for coord, sequences in self.dataloader:
                loss1 = 0.
                i+=1
                print(sequences.shape)
                sequences = sequences.to(self.device)
                for sequence in sequences:
                    loss2 = 0.
                    for s in sequence:

                        if self.kind == 1:
                            src_seq = s[:-1]
                            tgt_seq = s[-1]

                            self.optimizer.zero_grad()
                            output = self.model(src_seq)
                            # output = model(src_seq[-1])
                            loss = self.mse(output.view(-1), tgt_seq.view(-1))

                        elif self.kind == 2:
                            src_seq = s[:-1]
                            tgt_seq = s[-1]

                            self.optimizer.zero_grad()
                            output = self.model(src_seq[-1], tgt_seq)
                            loss = self.mse(output[-1].view(-1), tgt_seq.view(-1))

                        elif self.kind == 3:
                            src_seq = s[:-1]
                            tgt_seq = s[-1]

                            self.optimizer.zero_grad()
                            output = self.model(src_seq)
                            loss = self.mse(output[-1].view(-1), tgt_seq.view(-1))

                        elif self.kind == 4:
                            src_seq = s[:-1]
                            tgt_seq = s[-1].unsqueeze(0)

                            self.optimizer.zero_grad()
                            output = self.model(src_seq, tgt_seq)
                            # output = model(src_seq[-1])
                            loss = self.mse(output, tgt_seq)

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                        self.optimizer.step()
                        self.scheduler.step()
                        loss2 += loss.item()
                    loss1 += (loss2 / len(sequence))
                epoch_loss += (loss1 / len(sequences))
                print(i, self.log_interval, i % self.log_interval)
                if i % self.log_interval == 0:
                    elapsed = time.time() - start_time

                    print(f"| {i} steps | epoch {epoch + 1}/{self.epochs} | lr {self.scheduler.get_last_lr()[0]:.6f} "
                          f"| ms/batch {elapsed * 1000 / self.log_interval:.2f} | loss {loss1 / len(sequences):.6f}")
            running_loss = epoch_loss / len(self.dataloader)
            if self.is_wandb:
                wandb.log({"loss": running_loss, "lr": self.scheduler.get_last_lr()[0]})

            print(
                f"End of epoch {epoch + 1} / {self.epochs}, Running loss {running_loss:.6f}, at {datetime.now().time().strftime('%H:%M:%S')}")
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
        if self.is_wandb:
            wandb.finish()
        # Save the final trained model
        torch.save(self.model.state_dict(), os.path.join(self.save_directory, f"transformer_final_saved_model_{datetime.now().date().strftime('%m%d%Y')}.pth"))
