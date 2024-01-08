import h5py
import numpy as np
import dask.array as da
import os
import pickle
import concurrent.futures
import time
import sys

# Set a boolean flag to control the record limit
limit_records = True  # Set to True to limit records to 10,000


def read_hdf5(file_path):
    data = []
    with h5py.File(file_path, 'r') as f:
        def read_group(group):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    if limit_records:
                        # Read the first 10,000 records if available
                        num_records_to_read = min(item.shape[0], 10000)
                        data.append(item[:num_records_to_read])
                    else:
                        data.append(item[:])  # Read all records
                elif isinstance(item, h5py.Group):
                    read_group(item)
        read_group(f)
    return data



# Directory containing your HDF5 files
directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined'

# List all HDF5 files in the directory
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.hd5')]


def process_hdf5_file(file):
    print(f"Processing {file}...")
    sys.stdout.flush()  # Force immediate console output
    data = read_hdf5(file)
    dataset_count = sum(len(item) for item in data)

    if dataset_count < 1000000:
        print(f"Error: Only {dataset_count} datasets found in {file}. Expected at least 1,000,000 datasets.")
        return

    # Assuming your data is tabular and can be concatenated
    # Adjust this part based on the actual structure of your data
    combined_array = np.concatenate(data, axis=0)

    # Convert the combined array to a Dask array
    dask_array = da.from_array(combined_array, chunks=1000)  # Adjust chunk size as needed

    # Pickle the Dask array
    output_file = file.replace('.hd5', '.dask')
    with open(output_file, 'wb') as f:
        pickle.dump(dask_array, f)

    print(f"Completed {file} - Processed {dataset_count} datasets.")
    sys.stdout.flush()  # Force immediate console output


# Callback function to print progress
def progress_callback(future):
    file = future.args[0]
    print(f"Progress for {file}: {future.result()} datasets processed.")
    sys.stdout.flush()  # Force immediate console output


# Create a ThreadPoolExecutor with a maximum of 11 worker threads (one per file)
with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
    # Submit the processing tasks for each HDF5 file and specify the callback
    futures = [executor.submit(process_hdf5_file, file) for file in files]

    while not all(future.done() for future in futures):
        for future in concurrent.futures.as_completed(futures):
            if future.done():
                continue
            progress_callback(future)

        time.sleep(30)  # Wait for 30 seconds before checking progress again

print("All files processed.")
