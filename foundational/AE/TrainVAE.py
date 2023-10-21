import torch
import torch.optim as optim
from foundational.h5DataLoader import HDF5DataLoader
from VAE import VAE, loss_function
from foundational.h5DataLoader import DataLoader


class Trainer:
    def __init__(self, dataset_path, batch_size=1000, learning_rate=1e-3, num_epochs=500):
        # Initialize the data loader
        self.dataloader = DataLoader(HDF5DataLoader(dataset_path), batch_size=batch_size, shuffle=True)

        # Initialize the VAE and the optimizer
        self.model = VAE()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Check if CUDA is available and if so, move the model to the GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.model.to(self.device)

        self.num_epochs = num_epochs



    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.dataloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} [{batch_idx}/{len(self.dataloader)}]\tLoss: {loss.item() / len(data)}")
            print(f"====> Epoch {epoch}: Average Loss: {train_loss / len(self.dataloader.dataset)}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # Reconstruction + KL divergence losses summed over all elements and batch


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)


if __name__ == "__main__":
    dataset_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.hd5'
    trainer = Trainer(dataset_path)
    trainer.train()
    trainer.save_model('vae_model.pth')
