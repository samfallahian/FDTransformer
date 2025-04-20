from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class DataFrameDefragmenter(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        """
        Initialize the defragmenter with HostPreferences configuration.

        Args:
            filename (str): Configuration file name
        """
        super().__init__(filename)
        if not hasattr(self, 'output_directory'):
            raise AttributeError(
                "'output_directory' is required but not set in the parent class (HostPreferences).")
        if self.output_directory is None:
            raise ValueError("'output_directory' is set but contains None value. A valid path must be provided.")

    def read_pickle_file(self, file_path):
        """Read a pickle file with different compression methods."""
        for compression in [None, 'zip', 'gzip']:
            try:
                return pd.read_pickle(file_path, compression=compression)
            except Exception as e:
                if compression == 'gzip':  # If we've tried all methods
                    print(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue

    def defragment_dataframe(self, df):
        """
        Defragment a DataFrame by creating an optimized copy.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Defragmented DataFrame
        """
        if df is None:
            return None

        # Create a list of all columns and their data
        columns_data = []
        for col in df.columns:
            columns_data.append(pd.Series(df[col], name=col))

        # Create new DataFrame at once using concat
        defragmented_df = pd.concat(columns_data, axis=1)
        return defragmented_df

    def process_file(self, file_path):
        """Process a single file: read, defragment, and save."""
        try:
            print(f"Processing: {file_path}")
            df = self.read_pickle_file(file_path)
            if df is not None:
                # Defragment the DataFrame
                defragmented_df = self.defragment_dataframe(df)
                if defragmented_df is not None:
                    # Save back to the same location with compression
                    defragmented_df.to_pickle(file_path, compression='gzip')
                    return True
            return False
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False

    def run(self):
        """Process all pickle files in the output directory."""
        print(f"\nPath Configuration:")
        print(f"Directory: {self.output_directory}")

        # Ensure output directory exists and is writable
        try:
            if not os.path.exists(self.output_directory):
                raise FileNotFoundError(f"Directory not found: {self.output_directory}")
            if not os.access(self.output_directory, os.W_OK):
                raise PermissionError(f"Directory {self.output_directory} is not writable")
            print(f"Verified directory exists and is writable: {self.output_directory}")
        except Exception as e:
            raise RuntimeError(f"Directory validation failed: {str(e)}")

        # Get list of pickle files
        pickle_files = [os.path.join(self.output_directory, f)
                        for f in os.listdir(self.output_directory)
                        if f.endswith('.pkl')]

        if not pickle_files:
            print("No pickle files found in the specified directory.")
            return

        print(f"Found {len(pickle_files)} pickle files to process.")

        # Process files using ThreadPoolExecutor
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(self.process_file, pickle_files)

            for file_path, result in zip(pickle_files, results):
                if result:
                    success_count += 1
                    print(f"Successfully defragmented: {os.path.basename(file_path)}")
                else:
                    error_count += 1
                    print(f"Failed to process: {os.path.basename(file_path)}")

        # Print summary
        print("\nProcessing Summary:")
        print(f"Total files processed: {len(pickle_files)}")
        print(f"Successfully defragmented: {success_count}")
        print(f"Errors: {error_count}")


if __name__ == "__main__":
    defragmenter = DataFrameDefragmenter()
    defragmenter.run()