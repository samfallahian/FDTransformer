import os
import pandas as pd
import json
import random
import time
import torch
import h5py  # ADDED: Importing the h5py library
from CoordinateAnalyzer_V2 import CoordinateAnalyzer
import sys

'''
I'm going to just completely clean room this solution (Including CoordinateAnalyzer). I am getting inconsistent data like this:

Dataset Name: 85,-64,-9
        x      y      z        vx        vy        vz
0    77.0  184.0  239.0  0.543457  0.424316  0.439453
1    77.0  184.0  243.0  0.541016  0.422119  0.439453
2    77.0  184.0  247.0  0.541016  0.422119  0.439453
3    77.0  184.0  251.0  0.541016  0.422119  0.439453
4    77.0  184.0  255.0  0.539062  0.424316  0.439453
..    ...    ...    ...       ...       ...       ...
120  93.0  200.0  239.0  0.523926  0.432861  0.439453
121  93.0  200.0  243.0  0.521484  0.432861  0.439453
122  93.0  200.0  247.0  0.521484  0.435059  0.439453
123  93.0  200.0  251.0  0.523926  0.435059  0.439453
124  93.0  200.0  255.0  0.523926  0.435059  0.437256

Where the "y" and "z" coordinates here are clearly NOT correct. These should be values just below and just
above -64 and -9 respectively here, and they are most certainly not.

'''

# Define a global variable for the number of iterations
ITERATED = 100


def process_file(filename, META_DATA_FILE_PATH):
    """
    Process the HDF file and generate velocity tensors.

    Args:
        filename (str): Path to the HDF file.
        META_DATA_FILE_PATH (str): Path to the metadata file.

    Returns:
        None
    """
    # Open and load the metadata file as a dictionary
    with open(META_DATA_FILE_PATH, 'r') as f:
        experiment_dict = json.load(f)

    # Mark the start time to measure the duration of the entire process
    start_time = time.time()

    # Generate an output file path, if this file already exists delete it
    output_file_path = filename.replace('.hdf', '_tensors.hdf')
    if os.path.isfile(output_file_path):
        print(f"Deleting existing file: {output_file_path}")
        os.remove(output_file_path)

    # Read in the hdf file as a DataFrame
    df = pd.read_hdf(filename, key='processed_data')

    # Sort the DataFrame by x, y, z
    df.sort_values(by=['x', 'y', 'z'], inplace=True)

    # Initialize an empty list to store the velocity tensors
    velocity_tensors = []

    # Iterate over the range defined by ITERATED
    for _ in range(ITERATED):
        # Trim 2 lowest and 2 highest values from x, y, z arrays
        x_enumerated = df['x'].unique()
        y_enumerated = df['y'].unique()
        z_enumerated = df['z'].unique()

        x_enumerated_trimmed = x_enumerated[2:-2]
        y_enumerated_trimmed = y_enumerated[2:-2]
        z_enumerated_trimmed = z_enumerated[2:-2]

        # Assert that the trimmed arrays are of expected size
        assert len(x_enumerated_trimmed) == len(x_enumerated) - 4
        assert len(y_enumerated_trimmed) == len(y_enumerated) - 4
        assert len(z_enumerated_trimmed) == len(z_enumerated) - 4

        # Randomly choose a coordinate from the trimmed arrays
        x_random = random.choice(x_enumerated_trimmed)
        y_random = random.choice(y_enumerated_trimmed)
        z_random = random.choice(z_enumerated_trimmed)

        # Create a subset DataFrame for analysis
        df_subset = df

        try:
            # Instantiate the CoordinateAnalyzer and get the nearest values to the random coordinate
            analyzer = CoordinateAnalyzer(df_subset)
            result, combos = analyzer.get_nearest_values(x_random, y_random, z_random)
            #Where we keep the meta-data for what combinations we are using
            #combos = analyzer.get_nearest_values(x_random, y_random, z_random)

            # Assert that the result contains the expected number of data points
            assert len(result) == 125, "Analyzer does not contain 125 data points"

            # Convert the result to a tensor and append to the velocity_tensors list
            velocity_tensor = torch.tensor(result[['x', 'y', 'z', 'vx', 'vy', 'vz']].values).unsqueeze(0)
            velocity_tensors.append((velocity_tensor, combos))

        except Exception as e:
            print(f"Error processing data: {e}")

        # CHANGED: Save the list of velocity tensors as an hdf5 file using h5py
        # After all tensors are accumulated, write them to the HDF5 file
        with h5py.File(output_file_path, 'a') as hf:
            for tensor, combos in velocity_tensors:  # storing combos in the velocity_tensors
                coordinates_str = '_'.join([f'{x},{y},{z}' for x, y, z in combos])
                unique_name = f'{x_random},{y_random},{z_random}'
                dataset_name = f'{unique_name}'

                if dataset_name in hf:
                    print(f"Dataset {dataset_name} already exists")
                    continue  # Skip to the next iteration if dataset already exists

                ds = hf.create_dataset(dataset_name, data=tensor.numpy())
                ds.attrs['columns'] = ['x', 'y', 'z', 'vx', 'vy', 'vz']
                ds.attrs['centroid'] = coordinates_str
                # Sort and add the surrogate coordinates as attributes
                sorted_combos = sorted(combos)


        # Clear the velocity_tensors list after writing to the file
        velocity_tensors.clear()

        print(f"File processed and saved in: {output_file_path}")
        print(f"Total run time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python 050_gather_training_data_for_AE.py <filename> <META_DATA_FILE_PATH>")
        sys.exit(1)

    # Extract the command-line arguments
    filename = sys.argv[1]
    META_DATA_FILE_PATH = sys.argv[2]

    # Call the function with the provided arguments
    process_file(filename, META_DATA_FILE_PATH)
