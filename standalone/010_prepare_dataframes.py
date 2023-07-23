import pandas as pd
import os
import json
import zipfile
import concurrent.futures
from standalone import TransformLatent
import numpy as np

# Load the JSON file into a dictionary
with open("/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt", 'r') as f:
    experiment_dict = json.load(f)

# Specify the directories for input and output
input_dir = "/Users/kkreth/PycharmProjects/data/DL-PTV.backup/"
output_dir = "/Users/kkreth/PycharmProjects/data/DL-PTV-2/"

# Initialize the converter
converter = TransformLatent.FloatConverter()

def process_dataframe(df):
    # Ensure that vx, vy, vz, time, distance, x, y, and z are 16-bit signed integers
    columns_to_convert = ['time', 'distance', 'x', 'y', 'z']
    df[columns_to_convert] = df[columns_to_convert].astype('int16')

    # Ensure that vx, vy, vz are float32
    df[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']].astype('float32')

    # Create additional columns named vx.original, vy.original, and vz.original
    # and populate them with duplicate values from vx, vy, and vz respectively
    df['vx_original'] = df['vx']
    df['vy_original'] = df['vy']
    df['vz_original'] = df['vz']

    # Apply the transformation to vx, vy, vz
    df['vx'] = df['vx'].apply(converter.convert)
    df['vy'] = df['vy'].apply(converter.convert)
    df['vz'] = df['vz'].apply(converter.convert)

    # Drop columns px, py, and pz
    df.drop(['px', 'py', 'pz'], axis=1, inplace=True)

    return df

# Iterate over all experiments
for key in experiment_dict:
    # Construct the full file paths for input and output
    input_file_path = os.path.join(input_dir, key)
    output_file_path = os.path.join(output_dir, key)

    # Open the zip archive and extract the pickle file
    with zipfile.ZipFile(input_file_path, 'r') as zip_file:
        # Get the list of file names in the zip archive
        file_names = zip_file.namelist()

        # Assuming there's only one file in the zip archive, extract it
        extracted_file_name = file_names[0]
        extracted_file = zip_file.extract(extracted_file_name)

    # Read the extracted pickle file using pd.read_pickle
    df = pd.read_pickle(extracted_file)

    # Split the DataFrame into chunks for parallel processing
    chunks = np.array_split(df, 100)

    # Process each chunk using multiple workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        processed_chunks = list(executor.map(process_dataframe, chunks))

    # Concatenate the processed chunks back into a single DataFrame
    df_processed = pd.concat(processed_chunks)

    # Print the first 5 rows of the processed dataframe
    print(df_processed.describe())

    df_processed.to_hdf(path_or_buf=output_file_path, key='processed_data', mode='w', format='table', complib='zlib',
                        complevel=9,
                        data_columns=True)

    # Remove the extracted file
    os.remove(extracted_file)
