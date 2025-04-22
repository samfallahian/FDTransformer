'''
This script adds 375 coordinate columns (x_1 through x_125, y_1 through y_125, z_1 through z_125)
and 47 latent columns (latent_1 through latent_47) to the pickle files. 
All new columns are initialized as float32 data type.
'''

from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class CoordinateProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent class (HostPreferences). Check your configuration.")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' is set but contains None value. A valid path must be provided.")

    def read_pickle_file(self, file_path):
        """Read a pickle file with different compression methods."""
        for compression in [None, 'zip', 'gzip']:
            try:
                with open(file_path, 'rb') as f:
                    if compression:
                        df = pd.read_pickle(f, compression=compression)
                    else:
                        df = pd.read_pickle(f)
                    return df
            except Exception as e:
                if compression == 'gzip':  # If we've tried all methods
                    print(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue

    def add_coordinate_columns(self, df):
        """Add coordinate columns vx_1 through vx_125, vy_1 through vy_125, vz_1 through vz_125,
        and latent_1 through latent_47."""
        if df is None:
            return None

        # Create dictionaries to hold the new columns
        new_cols = {}
        
        # Create coordinate columns
        for i in range(1, 126):
            new_cols[f'vx_{i}'] = np.zeros(len(df), dtype=np.float32)
            new_cols[f'vy_{i}'] = np.zeros(len(df), dtype=np.float32)
            new_cols[f'vz_{i}'] = np.zeros(len(df), dtype=np.float32)
        
        # Create latent columns
        for i in range(1, 48):
            new_cols[f'latent_{i}'] = np.zeros(len(df), dtype=np.float32)
        
        # Create a DataFrame from the new columns
        new_df = pd.DataFrame(new_cols, index=df.index)
        
        # Concatenate the original DataFrame with the new columns
        result_df = pd.concat([df, new_df], axis=1)
        
        return result_df

    def process_file(self, file_path):
        """Process a single file: read and add coordinate columns."""
        df = self.read_pickle_file(file_path)
        if df is not None:
            processed_df = self.add_coordinate_columns(df)
            if processed_df is not None:
                # Save the processed dataframe with compression
                output_file = os.path.join(self.output_directory, os.path.basename(file_path))
                processed_df.to_pickle(output_file, compression='gzip')
                return True
        return False

    def run(self):
        print(f"\nPath Configuration:")
        print(f"Input: {self.output_directory}")
        print(f"Output: {self.output_directory}")
        print(f"Metadata Location: {self.metadata_location}")

        # Verify metadata location exists and is readable
        if not os.path.exists(self.metadata_location):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_location}")
        if not os.access(self.metadata_location, os.R_OK):
            raise PermissionError(f"Metadata file is not readable: {self.metadata_location}")

        # Ensure output directory exists and is writable
        try:
            os.makedirs(self.output_directory, exist_ok=True)
            if not os.access(self.output_directory, os.W_OK):
                raise PermissionError(f"Directory {self.output_directory} is not writable")
            print(f"Verified output directory exists and is writable: {self.output_directory}")
        except Exception as e:
            raise RuntimeError(f"Failed to setup output directory: {str(e)}")

        # Create list of absolute paths for each .pkl file
        file_paths = [os.path.join(self.output_directory, file)
                      for file in os.listdir(self.output_directory)
                      if file.endswith('.pkl')]

        print(f"Found {len(file_paths)} .pkl files to process")

        processed_count = 0
        error_count = 0

        # Process files using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(self.process_file, file_paths)

            for file_path, result in zip(file_paths, results):
                if result:
                    processed_count += 1
                    print(f"Successfully processed: {os.path.basename(file_path)}")
                else:
                    error_count += 1
                    print(f"Failed to process: {os.path.basename(file_path)}")

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(file_paths)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors: {error_count}")


if __name__ == "__main__":
    processor = CoordinateProcessor()
    processor.run()