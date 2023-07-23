import os
import pandas as pd
import json
import random
import time
import torch
from CoordinateAnalyzer import CoordinateAnalyzer

# Define a global variable for the number of iterations
ITERATED = 1000


def process_file(filename='/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/359.hdf',
                 META_DATA_FILE_PATH="/home/kkreth_umassd_edu/cgan/configs/Umass_experiments.txt"):
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
            result = analyzer.get_nearest_values(x_random, y_random, z_random)

            # Assert that the result contains the expected number of data points
            assert len(result) == 125, "Analyzer does not contain 125 data points"

            # Convert the result to a tensor and append to the velocity_tensors list
            velocity_tensor = torch.tensor(result[['vx', 'vy', 'vz']].values).unsqueeze(0)

            velocity_tensors.append(velocity_tensor)

        except Exception as e:
            print(f"Error processing data: {e}")

    # Save the list of velocity tensors as an hdf5 file
    torch.save(velocity_tensors, output_file_path)

    # Print a confirmation that the file has been processed and saved, along with the total runtime
    print(f"File processed and saved in: {output_file_path}")
    print(f"Total run time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    process_file()
