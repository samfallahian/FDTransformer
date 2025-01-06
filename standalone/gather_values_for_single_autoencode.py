import os
import pickle
import pandas as pd
import torch
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from CoordinateAnalyzer import CoordinateAnalyzer
from standalone import TransformLatent

# Loop for 2 million iterations
num_iterations = 100000
print_interval = 100

# Function to check float bounds
def check_float_bounds(value, lower_bound, upper_bound):
    if value < lower_bound or value > upper_bound:
        raise ValueError(f"Float value {value} is out of bounds [{lower_bound}, {upper_bound}]")

# Check if the pickle file exists
pickle_file = '_data_train_autoencoder.pickle'
if os.path.exists(pickle_file):
    # Load existing data from the pickle file
    load_start_time = time.time()
    with open(pickle_file, 'rb') as f:
        existing_data = pickle.load(f)
    load_time = time.time() - load_start_time
else:
    # Create a new empty list for data
    existing_data = []
    load_time = 0

# Define bounds
x_bound_lower = -113
x_bound_upper = 113
y_bound_lower = -72
y_bound_upper = 75
z_bound_lower = -21
z_bound_upper = 22

# Read the initial data from disk
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6.pkl', compression="zip")

# Global variables
randomx = None
randomy = None
randomz = None
arandomx = None
arandomy = None
arandomz = None

# Define the sample_with_retry function
def sample_with_retry(df):
    global randomx, randomy, randomz, arandomx, arandomy, arandomz
    while True:
        try:
            random_indices = np.random.choice(df.index, size=1)
            # Retrieve the corresponding rows
            randomSingleton = df.loc[random_indices]
            randomx = randomSingleton.x
            randomy = randomSingleton.y
            randomz = randomSingleton.z
            arandomx = randomx.iloc[0]
            arandomy = randomy.iloc[0]
            arandomz = randomz.iloc[0]
            check_float_bounds(arandomx, x_bound_lower, x_bound_upper)
            check_float_bounds(arandomy, y_bound_lower, y_bound_upper)
            check_float_bounds(arandomz, z_bound_lower, z_bound_upper)
            return randomSingleton
        except Exception as e:
            #Do nothing
            a = None
            #print(f"An exception occurred: {str(e)}")


# Create a torch device
device = torch.device("mps")



# Track the start time
start_time = time.time()

# Timer for sampling and data processing
sample_process_time = 0
# Timer for saving pickle data
save_time = 0

# Append the converted_values_tensor to the existing_data list
def append_data(converted_values_tensor):
    if converted_values_tensor.size() == torch.Size([1, 125, 3]):
        existing_data.append(converted_values_tensor)

# Variables for exception count
exception_count = 0
non_exception_count = 0

def process_iteration(i):
    global sample_process_time, exception_count, non_exception_count
    # Sample a random data point
    sample_start_time = time.time()
    randomSingleton = None
    try:
        randomSingleton = sample_with_retry(df)
        non_exception_count += 1
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        exception_count += 1
    sample_time = time.time() - sample_start_time
    sample_process_time += sample_time

    # Create a subset based on the random time
    df_subset = df[df['time'] == randomSingleton.time.iloc[0]]

    # Analyze coordinates and convert values
    analysis_start_time = time.time()
    analyzer = CoordinateAnalyzer(df_subset)
    result = analyzer.get_nearest_values(randomSingleton.x.iloc[0], randomSingleton.y.iloc[0], randomSingleton.z.iloc[0])
    result_vxVYvz = result.loc[:, ['vx', 'vy', 'vz']]

    converter = TransformLatent.FloatConverter()
    converted_values = converter.convert(result_vxVYvz)
    analysis_time = time.time() - analysis_start_time

    # Convert result_vxVYvz DataFrame to a NumPy array
    converted_values = result_vxVYvz.to_numpy()

    # Reshape converted_values to [1, 125, 3]
    converted_values_tensor = torch.from_numpy(converted_values).unsqueeze(0).to(device)

    # Append the converted_values_tensor to the existing_data list
    append_start_time = time.time()
    append_data(converted_values_tensor)
    append_time = time.time() - append_start_time

    # Print progress every 1000 iterations
    if (i + 1) % print_interval == 0:
        elapsed_time = time.time() - start_time
        print(f"Iteration: {i+1}, Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Time for sampling and data processing: {sample_process_time:.2f} seconds")
        print(f"Time for analysis: {analysis_time:.2f} seconds")
        print(f"Time for loading pickle data: {load_time:.2f} seconds")
        print(f"Time for appending data: {append_time:.2f} seconds")
        print(f"Time for saving pickle data: unknown seconds")
        print(f"Exception Count: {exception_count}, Non-Exception Count: {non_exception_count}")
        print("-" * 30)
        # Save the updated existing_data to the pickle file (overwrite mode)
        save_start_time = time.time()
        with open(pickle_file, 'wb') as f:
            pickle.dump(existing_data, f)
        save_time = time.time() - save_start_time

# Create a thread pool with 10 threads
with ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(process_iteration, i) for i in range(num_iterations)]
    # Wait for all threads to complete
    for future in futures:
        future.result()


# Print the final elapsed time
elapsed_time = time.time() - start_time
print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")
