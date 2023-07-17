import json
import pandas as pd
import numpy as np
import os
import random
import torch
from CoordinateAnalyzer import CoordinateAnalyzer
from multiprocessing import Pool, cpu_count

def process_file(filename):
    df = pd.read_hdf(os.path.join('/Users/kkreth/PycharmProjects/data/DL-PTV', filename), key='processed_data')

    # sort dataframe by x, y, z, and time
    df.sort_values(by=['x', 'y', 'z', 'time'], inplace=True)

    # Load the arrays from the JSON file and convert to integers
    x_enumerated = [int(x) for x in experiment_dict[filename]['x_enumerated']]
    y_enumerated = [int(y) for y in experiment_dict[filename]['y_enumerated']]
    z_enumerated = [int(z) for z in experiment_dict[filename]['z_enumerated']]
    time_enumerated = [int(t) for t in experiment_dict[filename]['time_enumerated']]

    # trim 2 lowest and 2 highest values from x, y, z arrays
    x_enumerated_trimmed = x_enumerated[2:-2]
    y_enumerated_trimmed = y_enumerated[2:-2]
    z_enumerated_trimmed = z_enumerated[2:-2]

    # ensure trimmed arrays are of expected size
    assert len(x_enumerated_trimmed) == len(x_enumerated) - 4
    assert len(y_enumerated_trimmed) == len(y_enumerated) - 4
    assert len(z_enumerated_trimmed) == len(z_enumerated) - 4

    counter = 0
    batch_size = 1000

    with open("/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.JSON", 'ab') as f:
        for _ in range(1000000):
            # select random x, y, z, and time
            x_random = random.choice(x_enumerated_trimmed)
            y_random = random.choice(y_enumerated_trimmed)
            z_random = random.choice(z_enumerated_trimmed)
            time_random = random.choice(time_enumerated)

            # create a new dataframe subset
            df_subset = df[(df['time'] == time_random)]

            # define analyzer variable
            try:
                analyzer = CoordinateAnalyzer(df_subset)
                result = analyzer.get_nearest_values(x_random, y_random, z_random)

                # assert that there are 125 datapoints
                assert len(result) == 125, "Analyzer does not contain 125 data points"

                # extract velocity information
                velocity_tensor = torch.tensor(result[['vx', 'vy', 'vz']].values).unsqueeze(0)

                torch.save(velocity_tensor, f)
                f.write(b'\n')

                counter += 1

                if counter % batch_size == 0:
                    print(f"{counter} lines written for file {filename}")

            except Exception as e:
                print(f"Failed to process file {filename} with error: {str(e)}")

# read in the meta-data file
with open("/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt", 'r') as f:
    experiment_dict = json.load(f)

# Get the list of .pkl files in the directory
file_list = [filename for filename in os.listdir('/Users/kkreth/PycharmProjects/data/DL-PTV') if filename.endswith('.pkl')]

# Create a multiprocessing Pool
pool = Pool(processes=cpu_count())

# Process the files in parallel
pool.map(process_file, file_list)

# Close the multiprocessing Pool
pool.close()
pool.join()
