import h5py
import torch


DATA_PATH="/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.hdf"
NEW_FILE="/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_corrected.hdf"

data_list  = torch.load(DATA_PATH)

with h5py.File(NEW_FILE, 'w') as f:
    # Create a dataset for your list of tensors
    dset = f.create_dataset('my_data', (len(data_list),) + data_list[0].shape, dtype='f', compression="gzip", compression_opts=9)

    # Store each tensor in the dataset
    for i, tensor in enumerate(data_list):
        dset[i] = tensor.numpy()
