import dask.dataframe as dd
import pandas as pd
import pickle
import os
pd.set_option('display.max_columns', None)

# Define the file path
csv_file = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_data_for_training_AE.csv'

# Set the chunksize
chunksize = 1000

# Create an iterator object for chunks
chunk_iterator = pd.read_csv(csv_file, chunksize=chunksize)

# Read and process the first chunk separately
first_chunk = next(chunk_iterator)

# print the first five rows
print(first_chunk.head(5))

# Initialize an empty dataframe
df_pandas = pd.DataFrame()

# Loop through the rest of the chunks
for chunk in chunk_iterator:

    # Define the data types - setting the first column to object (text) and rest to float16
    dtypes = {chunk.columns[0]: 'object'}
    dtypes.update({col: 'float16' for col in chunk.columns[1:]})

    # concatenate the chunk to df_pandas with appropriate dtypes
    df_pandas = pd.concat([df_pandas, chunk.astype(dtypes)], ignore_index=True)

# Rename the "Dataset Name" column to "DatasetName"
df_pandas.rename(columns={'Dataset Name': 'DatasetName'}, inplace=True)

# Define the path for the output pickle file
output_pickle_file = os.path.join(os.path.dirname(csv_file), 'combined_data_for_training_AE.dataframe.pkl')

# Save the Pandas DataFrame to disk using pickle
with open(output_pickle_file, 'wb') as file:
    pickle.dump(df_pandas, file)

print(f"DataFrame saved to {output_pickle_file}")