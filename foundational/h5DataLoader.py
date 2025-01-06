import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class HDF5DataLoader(Dataset):
    def __init__(self, file_path, debug=False):
        self.file = h5py.File(file_path, 'r')
        self.keys = list(self.file.keys())
        self.debug = debug

    def __del__(self):
        self.file.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]

        if key not in self.file:
            raise KeyError(f"Key {key} not found in HDF5 file.")

        data = torch.tensor(self.file[key][:], dtype=torch.float32).T

        # Using the dataset name (key) as the attribute name
        attribute_name = key
        if not attribute_name:
            raise ValueError(f"No valid attribute name for dataset at index {index}.")

        if self.debug:
            print(f"Index: {index}, Key: {key}, Data Shape: {data.shape}")

        return data, attribute_name


def print_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[...]
            df = pd.DataFrame(data)

            # If columns attribute is available in the dataset, set DataFrame columns
            if 'columns' in obj.attrs:
                df.columns = [str(col) for col in obj.attrs['columns']]

            print(f"Dataset Name: {name}")
            print(df.head(25))
            print()

        except Exception as e:
            print(f"Could not print dataset {name} due to {str(e)}")

def test_hdf5_dataloader():
    # Define test hdf5 file name
    test_file_name = '/home/kkreth_umassd_edu/DL-PTV/_combined/3p6.hd5'
    test_file_name = '/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/4p6.hd5'


    # Test DataLoader with debug mode enabled
    dataloader = DataLoader(HDF5DataLoader(test_file_name, debug=True), batch_size=1, shuffle=False)

    #hdf5_loader = HDF5DataLoader(test_file_name, debug=True)
    for idx, item in enumerate(dataloader):
        if idx >= 10000:
            break
        if idx % 1000 == 0:
            assert item is not None, f"Found a None item at index {idx} in the dataset!"

    for idx, (tensor, attribute_name) in enumerate(dataloader):
        if idx >= 10000:
            break
        assert tensor.shape == (1, 3, 125)  # Batch size is 1


if __name__ == "__main__":
    test_hdf5_dataloader()
