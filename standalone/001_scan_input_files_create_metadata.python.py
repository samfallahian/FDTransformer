import os
import pandas as pd
import torch
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Target directory containing .pkl files
target_dir = '/Users/kkreth/PycharmProjects/data/DL-PTV.backup/'
# Destination directory for JSON output
output_dir = '/Users/kkreth/PycharmProjects/cgan/configs/'

def process_file(file_path):
    # Open the pickle file and load into a pandas DataFrame
    with open(file_path, 'rb') as f:
        df = pd.read_pickle(f, compression="gzip")

    # Initialize a dictionary to store meta-data
    metadata_dict = {}

    # Define the columns of interest
    columns_of_interest = ['x', 'y', 'z', 'time', 'distance']

    for col in columns_of_interest:
        # Convert column to PyTorch tensor
        tensor = torch.from_numpy(df[col].to_numpy())

        # Remove duplicates and sort the tensor
        tensor = torch.sort(torch.unique(tensor)).values

        # Store min, max, 4th min, 4th max
        metadata_dict[col + '_min'] = float(tensor[0])
        metadata_dict[col + '_max'] = float(tensor[-1])
        metadata_dict[col + '_4th_min'] = float(tensor[3]) if tensor.shape[0] > 3 else float('nan')
        metadata_dict[col + '_4th_max'] = float(tensor[-4]) if tensor.shape[0] > 3 else float('nan')

        # Enumerate the sorted tensor and convert to list
        metadata_dict[col + '_enumerated'] = list(map(float, tensor.tolist()))

    return metadata_dict


# Use multithreading to process files in parallel
with ThreadPoolExecutor() as executor:
    # Create list of absolute paths for each .pkl file
    file_paths = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if file.endswith('.pkl')]

    # Initialize a dictionary to hold metadata for all files
    all_files_metadata = {}

    # Process each file
    for file_path, metadata in zip(file_paths, executor.map(process_file, file_paths)):
        # Use filename as key for metadata
        all_files_metadata[os.path.basename(file_path)] = metadata

    # Write all metadata to a JSON file
    with open(os.path.join(output_dir, 'Umass_experiments.txt'), 'w') as f:
        json.dump(all_files_metadata, f, indent=4)
