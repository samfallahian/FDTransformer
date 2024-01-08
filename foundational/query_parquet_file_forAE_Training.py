import os
import pyarrow.parquet as pq
import pandas as pd
import psutil
import time

def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Current memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

# The directory containing the Parquet files
parquet_directory_path = '/home/kkreth_umassd_edu/DL-PTV/combined_data.parquet'

# Define the batch size
batch_size = 10000  # Adjust based on your system's memory capacity

# Initialize variables
total_records = 0
sampled_records = []

# Iterate over each Parquet file in the directory
for file in os.listdir(parquet_directory_path):
    if file.endswith(".parquet"):
        file_path = os.path.join(parquet_directory_path, file)

        try:
            parquet_file = pq.ParquetFile(file_path)

            # Iterate over batches in each Parquet file
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()
                total_records += len(df)
                sampled_records.append(df.sample(min(len(df), 10)))  # Sample 10 or fewer records from each batch
                print_memory_usage()
                time.sleep(10.5)  # Wait for 500 ms
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Concatenate the sampled records
random_records = pd.concat(sampled_records).sample(10)  # Further sample down to 10 records if needed
random_records['row_number'] = random_records.index

# Print results
print(f"Total number of records: {total_records}")
print(random_records)
