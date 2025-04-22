'''
This script processes pickle files by examining x_enumerated, y_enumerated, and z_enumerated coordinates,
identifies the two lowest and two highest values from each list, and removes rows containing any of these
extreme values. It maintains the existing structure while focusing on cleaning the data by removing outliers.
'''

from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor


class ExtremeValueProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent class (HostPreferences).")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' must contain a valid path.")

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
                if compression == 'gzip':
                    print(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue

    def process_file(self, file_path):
        """Process a single file: read, remove extreme values, and save."""
        # Read the metadata file
        try:
            with open(self.metadata_location, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error reading metadata file: {str(e)}")
            return False

        # Get the base filename
        filename = os.path.basename(file_path)
        if filename not in metadata:
            print(f"No metadata found for {filename}")
            return False

        # Read the DataFrame
        df = self.read_pickle_file(file_path)
        if df is None:
            return False

        initial_rows = len(df)
        file_metadata = metadata[filename]

        # Get extreme values for each coordinate
        extreme_values = set()
        for coord in ['x', 'y', 'z']:
            if f'{coord}_enumerated' in file_metadata:
                values = file_metadata[f'{coord}_enumerated']
                # Add two lowest and two highest values
#TODO Increase this to 3 for the below to values to avoid errors we were seeing (look at 4/21 8:33 AM commit logs for details)
                extreme_values.update(values[:3])  # Two lowest
                extreme_values.update(values[-3:])  # Two highest

        # Create a boolean mask for rows to keep
        mask = ~(df['x'].isin(extreme_values) |
                 df['y'].isin(extreme_values) |
                 df['z'].isin(extreme_values))

        # Apply the mask to get the cleaned DataFrame
        df_cleaned = df[mask]
        final_rows = len(df_cleaned)

        # Save the processed dataframe
        output_file = os.path.join(self.output_directory, filename)
        df_cleaned.to_pickle(output_file, compression='gzip')

        print(f"Processed {filename}:")
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Removed rows: {initial_rows - final_rows}")

        return True

    def run(self):
        print(f"\nPath Configuration:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")
        print(f"Metadata Location: {self.metadata_location}")

        # Ensure output directory exists and is writable
        try:
            os.makedirs(self.output_directory, exist_ok=True)
            if not os.access(self.output_directory, os.W_OK):
                raise PermissionError(f"Directory {self.output_directory} is not writable")
        except Exception as e:
            raise RuntimeError(f"Failed to setup output directory: {str(e)}")

        # Get list of pickle files
        file_paths = [os.path.join(self.output_directory, file)
                      for file in os.listdir(self.output_directory)
                      if file.endswith('.pkl')]

        print(f"Found {len(file_paths)} .pkl files to process")

        processed_count = 0
        error_count = 0

        # Process files using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(self.process_file, file_paths)

            for file_path, result in zip(file_paths, results):
                if result:
                    processed_count += 1
                else:
                    error_count += 1

        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(file_paths)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors: {error_count}")


if __name__ == "__main__":
    processor = ExtremeValueProcessor()
    processor.run()