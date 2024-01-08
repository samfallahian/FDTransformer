import os
import pandas as pd
import dask.dataframe as dd
import h5py
from multiprocessing import Pool
from tqdm import tqdm

def read_and_combine_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        all_dataframes = []
        for dataset_name in h5file.keys():
            data = h5file[dataset_name][:]
            df = pd.DataFrame(data)
            all_dataframes.append(df)
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df

def process_file(file):
    print(f"Processing {file}...")
    temp_df = read_and_combine_hdf5(file)
    print(f"Read {len(temp_df)} records from {file}")
    return temp_df

def combine_hdf5_files(directory, output_directory):
    hdf5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.hd5')]

    # Using multiprocessing to process each file
    with Pool() as pool:
        all_dfs = list(tqdm(pool.imap(process_file, hdf5_files), total=len(hdf5_files)))

    # Combining all DataFrames into one Dask DataFrame
    combined_df = dd.from_pandas(pd.concat(all_dfs, ignore_index=True), npartitions=len(hdf5_files))

    # Convert all column names to strings to satisfy Parquet requirements
    combined_df.columns = combined_df.columns.astype(str)

    # Saving the combined DataFrame as Parquet
    output_file_path = os.path.join(output_directory, 'combined_data.parquet')
    combined_df.to_parquet(output_file_path)

    print(f"Successfully combined files into {output_file_path}")

if __name__ == "__main__":
    input_directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined'
    input_directory = '/home/kkreth_umassd_edu/DL-PTV/_combined'
    output_directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/'
    output_directory = '/home/kkreth_umassd_edu/DL-PTV/'
    combine_hdf5_files(input_directory, output_directory)
