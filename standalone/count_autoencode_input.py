import pickle
import numpy as np

# Specify the path to the pickle file
pickle_file = "_data_train_autoencoder.pickle"

# Read the pickle file
with open(pickle_file, "rb") as f:
    data = pickle.load(f)

# Count the number of tensors and their shapes
tensor_count = 0
unique_tensors = set()

for tensor in data:
    if tensor.shape == (1, 125, 3):
        tensor_count += 1
        tensor_cpu = tensor.cpu()
        unique_tensors.add(tuple(np.array(tensor_cpu).flatten()))

# Print the count of unique tensors
print(f"Number of unique tensors: {tensor_count}")
