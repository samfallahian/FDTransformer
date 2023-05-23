import pandas as pd
import torch
import os
import sys

def normalize_data(input_file):
    # Load the input DataFrame from a pickled file
    df = pd.read_pickle(input_file, compression="zip")

    # Drop the columns that we don't need
    df = df.drop(columns=['px','py','pz','distance'])

    # Select the columns we want to normalize
    df_subset = df[['vx','vy','vz']]

    # Convert the columns to float32 data type
    df_subset = df_subset.astype('float32')

    # First try to create the tensor on the CUDA device
    try:
        tensor_device = torch.device('cuda')
        tensor = torch.tensor(df_subset.values, device=tensor_device, dtype=torch.float32)
        print(f"Created tensor on {tensor_device} device")

    # If CUDA is not available, try to create the tensor on the MPS device
    except AssertionError as e1:
        if 'not compiled with CUDA' in str(e1):
            try:
                tensor_device = torch.device('mps')
                tensor = torch.tensor(df_subset.values, device=tensor_device, dtype=torch.float32)
                print(f"Created tensor on {tensor_device} device")

            # If MPS is not available, fall back to creating the tensor on the CPU
            except RuntimeError as e2:
                tensor_device = torch.device('cpu')
                tensor = torch.tensor(df_subset.values, device=tensor_device, dtype=torch.float32)
                print(f"Created tensor on {tensor_device} device")

        # If the error is not related to CUDA out of memory, raise the error
        else:
            raise e1

    # Create a tensor from the DataFrame using PyTorch and move it to the CUDA device
    tensor_cuda = torch.tensor(df_subset.values, device=tensor_device)

    # Calculate the minimum and maximum values of the tensor
    min_val = torch.min(tensor_cuda)
    max_val = torch.max(tensor_cuda)

    # Perform min-max normalization using tensor operations
    tensor_normalized = (tensor_cuda - min_val) / (max_val - min_val)

    # Move the normalized tensor back to the CPU device and extract the normalized values as numpy arrays
    tensor_cpu = tensor_normalized.cpu()
    x_norm, y_norm, z_norm = tensor_cpu.numpy()[:,0], tensor_cpu.numpy()[:,1], tensor_cpu.numpy()[:,2]

    # Add the normalized values to the original DataFrame as new columns
    df['vx_norm'] = x_norm
    df['vy_norm'] = y_norm
    df['vz_norm'] = z_norm

    # Save the DataFrame to disk with gzip compression
    output_file = os.path.splitext(input_file)[0] + '_with_normalized_values.pkl.gz'
    df.to_pickle(output_file, compression='gzip', protocol=5, storage_options=None)

    # Save a sample of the DataFrame with 100,000 rows to disk with gzip compression
    output_file_sample = os.path.splitext(input_file)[0] + '_with_normalized_values_100000_sample.pkl.gz'
    df.sample(100000).to_pickle(output_file_sample, compression='gzip', protocol=5, storage_options=None)

    return output_file, output_file_sample

# Example usage:
input_file = sys.argv[1]
output_file, output_file_sample = normalize_data(input_file)
print(f"Normalized DataFrame saved to {output_file}")
print(f"Sampled DataFrame saved to {output_file_sample}")
