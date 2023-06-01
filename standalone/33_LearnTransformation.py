import os
import torch
import pandas as pd

# Define the directory path containing the files
directory_path = "/home/kkreth_umassd_edu/data_pi/raw_input"

# List all files in the directory
file_list = os.listdir(directory_path)

# Initialize min and max values for columns
min_values = None
max_values = None

# Iterate over each file
for file_name in file_list:
    file_path = os.path.join(directory_path, file_name)

    # Read the file or perform any necessary file loading operation
    # Assuming you have a function to load the file and obtain a PyTorch tensor called "data_tensor"
    print("Now working on " + file_path)
    df = pd.read_pickle(file_path, compression="zip")

    #Now to see if we can look accross columns and get the same answer(s)
    columns_of_interest = ['vx','vy','vz']
    df_subset = df[columns_of_interest]
    # Create a tensor from the DataFrame using PyTorch and move it to the CUDA device
    data_tensor = torch.tensor(df_subset.values, device='cuda')

    # Find the minimum and maximum values of columns
    column_mins, _ = torch.min(data_tensor, dim=0)
    column_maxs, _ = torch.max(data_tensor, dim=0)

    # Update the overall min and max values if needed
    if min_values is None:
        min_values = column_mins
        max_values = column_maxs
    else:
        min_values = torch.min(min_values, column_mins)
        max_values = torch.max(max_values, column_maxs)

# Print the overall min and max values of columns
print("Min values:", min_values)
print("Max values:", max_values)

'''
For vx, here was the output:
/home/kkreth_umassd_edu/.virtualenvs/cgan/bin/python /home/kkreth_umassd_edu/cgan/standalone/33_LearnTransformation.py 
Min values: tensor(-1.1100, device='cuda:0', dtype=torch.float64)
Max values: tensor(2.6400, device='cuda:0', dtype=torch.float64)

Process finished with exit code 0


Here it is for vy:
Min values: tensor(-1.9800, device='cuda:0', dtype=torch.float64)
Max values: tensor(2.2000, device='cuda:0', dtype=torch.float64)


For vx, vy, and vz:
Min values: tensor([-1.1100, -1.9800, -1.2200], device='cuda:0', dtype=torch.float64)
Max values: tensor([2.6400, 2.2000, 1.1000], device='cuda:0', dtype=torch.float64)

For the above, that should be for roughly 60M rows per file. ~10 files, so 600M rows.
There are 3 values per row, so 1.8B rows were scanned approximatly. 

High of 2.6400
Low of -1.9800

'''