import pickle
import torch

# Load data
with open('_data_train_autoencoder.pickle', 'rb') as f:
    data_list = pickle.load(f)

# Flatten tensors
flat_data_list = [tensor.reshape(tensor.size(0), -1) for tensor in data_list]

# Save to new pickle file
with open('_data_train_autoencoder_flat.pickle', 'wb') as f:
    pickle.dump(flat_data_list, f)