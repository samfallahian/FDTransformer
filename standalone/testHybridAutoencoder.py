import pytest
import torch
from torch.utils.data import DataLoader
# Import your dataset class. For this example, I'm assuming it's named `MyDataset`.
# Adjust the import path according to your project structure.

from torch.utils.data import Dataset

class ListOfTensorsDataset(Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        return self.tensor_list[idx]



DATA_PATH = "/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_subset.hdf"

@pytest.fixture
def setup_dataloader():
    # Load tensor directly from file
    dataset = torch.load(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloader


def test_dataloader_output(setup_dataloader):
    dataloader = setup_dataloader
    for i, batch in enumerate(dataloader):
        # Check type of batch
        assert isinstance(batch, torch.Tensor), f"Expected output type: torch.Tensor, but got: {type(batch)}"

        # Check shape of batch, except for the last one
        if i < len(dataloader) - 1:  # If it's not the last batch
            assert batch.shape == (100, 125, 3), f"Expected output shape: (100, 125, 3), but got: {batch.shape}"
        else:
            assert batch.shape[1:] == (
            125, 3), f"Expected last batch shape to have dimensions (x, 125, 3), but got: {batch.shape}"


