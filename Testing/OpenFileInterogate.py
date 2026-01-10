import pickle
import gzip
import pandas as pd
import os

# Set pandas options to display all columns and wide output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_path = '/Users/kkreth/PycharmProjects/cgan/Testing/8p4_final_results.pkl'

def try_open(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    # Try as gzipped pickle
    try:
        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
            print("Successfully loaded as gzipped pickle.")
            return data
    except Exception as e:
        # Silently fail if not gzipped, or print for debugging if needed
        pass

    # Try as regular pickle
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            print("Successfully loaded as regular pickle.")
            return data
    except Exception as e:
        print(f"Failed to load as regular pickle: {e}")
    
    return None

data = try_open(file_path)

if data is not None:
    print(f"Data type: {type(data)}")
    if isinstance(data, pd.DataFrame):
        print("First 10 rows (all columns):")
        print(data.head(10))
    elif isinstance(data, list):
        print("First 10 elements:")
        for item in data[:10]:
            print(item)
    elif isinstance(data, dict):
        print("First 10 keys:")
        for key in list(data.keys())[:10]:
            print(f"{key}: {data[key]}")
    else:
        print("Data content:")
        print(data)
else:
    print("Could not load the file.")
