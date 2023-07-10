# Import necessary libraries
import os
import pandas as pd
import torch
import concurrent.futures
import json


# Define the function to get the desired values
def get_desired_values(tensor):
    # Sort the tensor
    sorted_tensor = torch.sort(tensor.unique())[0]  # use unique() to get unique values

    # Get the smallest and largest values
    smallest = sorted_tensor[0].item()
    largest = sorted_tensor[-1].item()

    # Get the fourth smallest and fourth largest values
    fourth_smallest = sorted_tensor[3].item() if sorted_tensor.size(0) > 3 else None
    fourth_largest = sorted_tensor[-4].item() if sorted_tensor.size(0) > 3 else None

    return smallest, largest, fourth_smallest, fourth_largest


# Define the function to process a file
def process_file(filename):
    # Construct the full file path
    file_path = os.path.join(read_dir, filename)

    # Open the pickle file and load into a pandas DataFrame
    with open(file_path, 'rb') as f:
        df = pd.read_pickle(f, compression="zip")

    # Create an empty dictionary for this file's metadata
    file_metadata = {}

    # Iterate over the columns of interest
    for col in cols_of_interest:
        # Check if the column exists in the dataframe
        if col in df.columns:
            # Convert the column to a PyTorch tensor
            tensor = torch.from_numpy(df[col].values)

            # Get the desired values and store them in the file's metadata
            file_metadata[col] = get_desired_values(tensor)

    return filename, file_metadata


# Define the directory to read from
read_dir = '/Users/kkreth/PycharmProjects/data/DL-PTV/'

# Define the output file
output_file = '/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt'

# Define the columns of interest and sort them alphabetically
cols_of_interest = sorted(['x', 'y', 'z', 'time', 'distance'])

# Get all .pkl files in the directory
files = [file for file in os.listdir(read_dir) if file.endswith('.pkl')]

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Use the executor to map the process_file function to the files
    results = executor.map(process_file, files)

    # Create an empty dictionary to store the metadata
    metadata = dict(results)

# Write the metadata to the output file as JSON
with open(output_file, 'w') as f:
    json.dump(metadata, f)
