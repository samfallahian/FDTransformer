import h5py
import numpy as np
import dask.array as da
import os
import pickle

def read_hdf5(file_path):
    data = []
    with h5py.File(file_path, 'r') as f:
        def read_group(group):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    data.append(item[:])  # Read dataset into memory
                elif isinstance(item, h5py.Group):
                    read_group(item)
        read_group(f)
    return data

# Directory containing your HDF5 files
directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined'

# List all HDF5 files in the directory
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.hd5')]

# Read and combine data from all HDF5 files
combined_data = []
for file in files:
    combined_data.extend(read_hdf5(file))

# Assuming your data is tabular and can be concatenated
# Adjust this part based on the actual structure of your data
combined_array = np.concatenate(combined_data, axis=0)

# Convert the combined array to a Dask array
dask_array = da.from_array(combined_array, chunks=1000)  # Adjust chunk size as needed

# Pickle the Dask array
with open('/Users/kkreth/PycharmProjects/data/DL-PTV/combined_external_links.dask', 'wb') as f:
    pickle.dump(dask_array, f)
