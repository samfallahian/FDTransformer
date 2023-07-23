import os
import pandas as pd
import numpy as np

base_path = '/Users/kkreth/PycharmProjects/data/DL-PTV'

# Read all pkl files from base directory
for filename in os.listdir(base_path):
    if filename.endswith('.pkl'):
        file_path = os.path.join(base_path, filename)
        df = pd.read_hdf(file_path, key='processed_data')

        # Assert there are exactly 1200 unique time values
        assert len(df['time'].unique()) == 1200, f"File {filename} does not have exactly 1200 unique time values."

        # Create a new directory for each pkl file
        new_dir_path = os.path.join(base_path, filename[:-4])  # remove '.pkl' from filename
        os.makedirs(new_dir_path, exist_ok=True)

        # Split dataframe by 'time' value and save as hdf files
        for time_val, group in df.groupby('time'):
            output_file_path = os.path.join(new_dir_path, str(time_val) + '.hdf')
            group.to_hdf(path_or_buf=output_file_path, key='processed_data', mode='w', format='table', complib='zlib',
                         complevel=9, data_columns=True)
