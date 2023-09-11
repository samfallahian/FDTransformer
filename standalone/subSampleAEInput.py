import torch


def save_subset(data, output_path, fraction=0.1):
    """
    Save a subset of the data.

    Args:
    - data (torch.Tensor or dict): Data loaded using torch.load. Can be a tensor or dictionary of tensors.
    - output_path (str): Path to save the subset data.
    - fraction (float): Fraction of data to retain.
    """
    if isinstance(data, torch.Tensor):
        subset_len = int(fraction * len(data))
        subset = data[:subset_len]
    elif isinstance(data, dict):
        subset = {}
        for key, value in data.items():
            subset_len = int(fraction * len(value))
            subset[key] = value[:subset_len]
    else:
        raise ValueError("Unsupported data type for subsetting.")

    torch.save(subset, output_path)
    print(f"Subset saved to: {output_path}")


# Specify the paths
input_path = '/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.hdf'
output_path = '/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_subset.hdf'

# Attempt to load the file using PyTorch
try:
    data = torch.load(input_path)
    print("File loaded successfully using PyTorch.")

    # Create and save the subset
    save_subset(data, output_path, fraction=0.1)

except Exception as e:
    print(f"Error when trying to load using PyTorch: {e}")
