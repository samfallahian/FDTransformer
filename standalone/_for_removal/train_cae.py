import torch.optim as optim
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from ContractiveAutoencoder import ContractiveAutoencoder




class Driver:
    def split_data(self, data, split_ratio=0.8):
        train_len = int(len(data) * split_ratio)
        val_len = len(data) - train_len
        train_data, val_data = random_split(data, [train_len, val_len])
        return train_data, val_data
    def __init__(self, model, device, data_path="_data_train_autoencoder_flat.pickle", batch_size=10000, lr=0.0001):
        self.model = model
        print("Model initialized.")
        self.device = device
        self.data = pickle.load(open(data_path, "rb"))
        print(f"Data loaded. Total samples: {len(self.data)}")

        self.batch_size = batch_size
        #self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scaler = GradScaler()  # For mixed precision training
        self.train_data, self.val_data = self.split_data(self.data)
        self.epochs = 10000
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=1000, shuffle=True)  # data loader for training set
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=1000, shuffle=True)  # data loader for validation set
        self.model.to(self.device)
        print("Model moved to the selected device.")

    def train(self):
        # Use automatic mixed precision if available
        scaler = torch.cuda.amp.GradScaler(enabled=self.device == 'cuda')

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_error = 0.0
            for i, batch in enumerate(self.train_loader, 0):
                # Ensure the data is float
                batch = batch.float().to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize with automatic mixed precision
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    outputs, encoded = self.model(batch)
                    loss = self.model.loss_function(batch, outputs, encoded)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # print statistics
                running_loss += loss.item()
                running_error += torch.mean((outputs - batch) ** 2).item()
                if i % 1000 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.8f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    print('[%d, %5d] error: %.8f' %
                          (epoch + 1, i + 1, running_error / 1000))
                    running_loss = 0.0
                    running_error = 0.0

            # save model every 10 epochs
            if epoch % 1000 == 9:
                torch.save(self.model.state_dict(), f'model_epoch_{epoch + 1}.pth')
        print('Finished Training')


if __name__ == "__main__":
    # Define your model
    model = ContractiveAutoencoder(input_size=375, latent_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    device = torch.device("mps")
    # Create the driver
    driver = Driver(data_path="_data_train_autoencoder_flat.pickle", device=device, model=model)

    # Start training
    driver.train()