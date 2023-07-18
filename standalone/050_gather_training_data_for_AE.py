import os
import pandas as pd
import json
import random
import time
import torch
from CoordinateAnalyzer import CoordinateAnalyzer

# 0) Meta-data file path as GLOBAL variable
META_DATA_FILE_PATH = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
ITERATED = 1000

# Read in the meta-data file
with open(META_DATA_FILE_PATH, 'r') as f:
    experiment_dict = json.load(f)


def process_file(filename='/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/359.hdf'):
    start_time = time.time()

    # 0.5) Delete file if it exists
    output_file_path = filename.replace('.hdf', '_tensors.hdf')
    if os.path.isfile(output_file_path):
        print(f"Deleting existing file: {output_file_path}")
        os.remove(output_file_path)

    # 2) Read file
    df = pd.read_hdf(filename, key='processed_data')

    # 5) Sort dataframe by x, y, z
    df.sort_values(by=['x', 'y', 'z'], inplace=True)

    velocity_tensors = []

    # 3) Iterations
    for _ in range(ITERATED):
        # 6) Trim 2 lowest and 2 highest values from x, y, z arrays
        x_enumerated = df['x'].unique()
        y_enumerated = df['y'].unique()
        z_enumerated = df['z'].unique()

        x_enumerated_trimmed = x_enumerated[2:-2]
        y_enumerated_trimmed = y_enumerated[2:-2]
        z_enumerated_trimmed = z_enumerated[2:-2]

        # ensure trimmed arrays are of expected size
        assert len(x_enumerated_trimmed) == len(x_enumerated) - 4
        assert len(y_enumerated_trimmed) == len(y_enumerated) - 4
        assert len(z_enumerated_trimmed) == len(z_enumerated) - 4

        # 7) Random coordinates
        x_random = random.choice(x_enumerated_trimmed)
        y_random = random.choice(y_enumerated_trimmed)
        z_random = random.choice(z_enumerated_trimmed)

        # 8) Use analyzer to find nearest values
        df_subset = df

        try:
            # Assuming CoordinateAnalyzer is a defined class that takes a DataFrame and provides a method get_nearest_values
            analyzer = CoordinateAnalyzer(df_subset)
            result = analyzer.get_nearest_values(x_random, y_random, z_random)

            assert len(result) == 125, "Analyzer does not contain 125 data points"

            velocity_tensor = torch.tensor(result[['vx', 'vy', 'vz']].values).unsqueeze(0)

            velocity_tensors.append(velocity_tensor)

        except Exception as e:
            print(f"Error processing data: {e}")

    # 9) Gather tensors and write them to hdf5
    torch.save(velocity_tensors, output_file_path)

    print(f"File processed and saved in: {output_file_path}")
    print(f"Total run time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    process_file()
