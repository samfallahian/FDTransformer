import os
import json
import pandas as pd
import torch
from threading import Thread

# Function to read pickle file into a dataframe
def read_file(file_path):
    # Open the pickle file and load into a pandas DataFrame
    with open(file_path, 'rb') as f:
        df = pd.read_pickle(f, compression="zip")
    return df

# Function to get the smallest, largest, 4th smallest and 4th largest values
def get_values(df, column):
    # Convert the pandas series to a PyTorch tensor
    tensor = torch.from_numpy(df[column].values)
    # Sort the tensor
    sorted_tensor, indices = torch.sort(tensor)
    # Get the smallest, largest, 4th smallest and 4th largest values
    smallest = sorted_tensor[0].item()
    largest = sorted_tensor[-1].item()
    fourth_smallest = sorted_tensor[3].item() if len(sorted_tensor) > 3 else None
    fourth_largest = sorted_tensor[-4].item() if len(sorted_tensor) > 3 else None
    # Get the enumerated list of unique sorted values
    enumerated = list(set(sorted_tensor.tolist()))
    enumerated.sort() # Sort the list after converting to set
    return smallest, largest, fourth_smallest, fourth_largest, enumerated

# Function to process each file
def process_file(file_path, result_dict):
    # Read the file into a dataframe
    df = read_file(file_path)
    # Store results for each required column in the result dictionary
    for column in ["x", "y", "z", "time", "distance"]:
        values = get_values(df, column)
        # Store the results in the dictionary, indexed by the filename and column
        result_dict[file_path][f"{column}_min"] = values[0]
        result_dict[file_path][f"{column}_max"] = values[1]
        result_dict[file_path][f"{column}_fourth_min"] = values[2]
        result_dict[file_path][f"{column}_fourth_max"] = values[3]
        result_dict[file_path][f"{column}_enumerated"] = values[4]

# Main function to loop through all files in the directory
def process_all_files(directory):
    # Initialize result dictionary
    result_dict = {}
    # Get all files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pkl')]
    # Start a thread for each file
    threads = []
    for file_path in files:
        # Initialize subdictionary for each file
        result_dict[file_path] = {}
        thread = Thread(target=process_file, args=(file_path, result_dict))
        threads.append(thread)
        thread.start()
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    return result_dict

# Start processing files
result = process_all_files("/Users/kkreth/PycharmProjects/data/DL-PTV/")

# Save the results into a json file
with open("/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt", 'w') as f:
    json.dump(result, f)
