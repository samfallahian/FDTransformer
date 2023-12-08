import os
import pandas as pd
import dask.dataframe as dd
import h5py

def read_hdf5_to_pandas(file_path):
    with h5py.File(file_path, 'r') as h5file:
        all_dataframes = []
        for dataset_name in h5file.keys():
            data = h5file[dataset_name][:]
            df = pd.DataFrame(data)
            all_dataframes.append(df)
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df

def combine_hdf5_files(directory, output_directory):
    hdf5_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.hd5')]
    combined_df = None

    for file in hdf5_files:
        print(f"Processing {file}...")
        # Read each file into a combined Pandas DataFrame
        temp_df = read_hdf5_to_pandas(file)

        print(f"Read {len(temp_df)} records from {file}")

        # Convert Pandas DataFrame to Dask DataFrame
        dask_df = dd.from_pandas(temp_df, npartitions=1)

        if combined_df is None:
            combined_df = dask_df
        else:
            combined_df = dd.concat([combined_df, dask_df])

    if combined_df is not None:
        # Convert all column names to strings to satisfy Parquet requirements
        combined_df.columns = combined_df.columns.astype(str)

        # Saving the combined DataFrame as Parquet
        output_file_path = os.path.join(output_directory, 'combined_data.parquet')
        combined_df.to_parquet(output_file_path)

        print(f"Successfully combined files into {output_file_path}")
    else:
        print("No data was combined.")

if __name__ == "__main__":
    input_directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined'
    output_directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/'
    combine_hdf5_files(input_directory, output_directory)
