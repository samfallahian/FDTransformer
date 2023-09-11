from torch.utils.data import DataLoader, random_split
import torch
from HybridAutoencoder import HybridAutoencoder, TrainHA, loss_function
from standalone.HDF4Dataset import HDF5Dataset

def train_conv(data_path="/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_subset_corrected.hdf"):
    # Create an instance of HDF5Dataset
    data_tensor = HDF5Dataset(data_path)

    # Split the data_tensor 50/50 for train/validation
    num_train = len(data_tensor) // 2
    num_val = len(data_tensor) - num_train
    train_dataset, val_dataset = random_split(data_tensor, [num_train, num_val])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

    # Initialize model and train
    model = HybridAutoencoder(latent_size=(8, 6))
    TrainHA(model, train_loader, epochs=100, learning_rate=0.001, beta=.75)

    # Evaluate on validation set (optional)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in val_loader:
            reconstructed, mu, logvar = model(data)
            loss = loss_function(reconstructed, data, mu, logvar, beta=1.0)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation loss: {avg_loss:.4f}")

    return model

model = train_conv(data_path="/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_subset_corrected.hdf")
