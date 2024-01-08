import h5py
import pandas as pd

def read_hdf5_to_dataframe(file_path, num_entries=1000):
    data = []
    dataset_names = []
    with h5py.File(file_path, 'r') as f:
        # Recursive function to read data from groups and datasets
        def read_data(group, prefix=''):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    dataset_name = f"{prefix}/{key}"
                    dataset_names.append(dataset_name)
                    data.append(item[:num_entries])

        # Initialize an empty DataFrame with 125 columns
        df = pd.DataFrame(columns=[f'Column_{i}' for i in range(125)])

        # Recursive function to print details of groups and datasets
        def inspect_group(group, prefix=''):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    dataset_name = f"{prefix}/{key}"
                    dataset_names.append(dataset_name)
                    data.append(item[:num_entries])

                elif isinstance(item, h5py.Group):
                    inspect_group(item, prefix=f"{prefix}/{key}")

        inspect_group(f)

        for i, dataset_name in enumerate(dataset_names):
            df[dataset_name] = data[i]

    return df

if __name__ == "__main__":
    hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.hd5'  # Change this to your file's path
    df = read_hdf5_to_dataframe(hdf5_file_path, num_entries=1000)

    if df is not None:
        print(df.head())  # Print the first few rows of the DataFrame
    else:
        print('No data found in the HDF5 file.')
