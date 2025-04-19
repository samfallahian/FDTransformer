'''
OK, beware, if your system has <64 GB of RAM, I would highly recommend that you
limit the threads to 2 or even 1.
'''


from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import TransformLatent
from concurrent.futures import ThreadPoolExecutor

class CleanFilesProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent class (HostPreferences). Check your configuration.")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' is set but contains None value. A valid path must be provided.")
        self.converter = TransformLatent.FloatConverter()

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

    def process_dataframe(self, df):
        """Process a single dataframe according to the requirements."""
        if df is None:
            return None

        # Ensure that vx, vy, vz, time, distance, x, y, and z are 16-bit signed integers
        columns_to_convert = ['time', 'distance', 'x', 'y', 'z']
        df[columns_to_convert] = df[columns_to_convert].astype('int32')

        # Create additional columns for original velocity values
        df['vx_original'] = df['vx']
        df['vy_original'] = df['vy']
        df['vz_original'] = df['vz']

        # Apply the transformation to vx, vy, vz
        df['vx'] = df['vx'].apply(self.converter.convert)
        df['vy'] = df['vy'].apply(self.converter.convert)
        df['vz'] = df['vz'].apply(self.converter.convert)

        # Ensure that vx, vy, vz are float32
        df[['vx', 'vy', 'vz']] = df[['vx', 'vy', 'vz']].astype('float32')

        # Drop columns px, py, and pz
        df.drop(['px', 'py', 'pz'], axis=1, inplace=True)

        return df

    def process_file(self, file_path):
        """Process a single file: read and transform its dataframe."""
        df = self.read_pickle_file(file_path)
        if df is not None:
            processed_df = self.process_dataframe(df)
            if processed_df is not None:
                # Save the processed dataframe with compression
                output_file = os.path.join(self.output_directory, os.path.basename(file_path))
                processed_df.to_pickle(output_file, compression='gzip')
                return True
        return False

    def run(self):
        print(f"\nPath Configuration:")
        print(f"Input: {self.raw_input}")
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
        file_paths = [os.path.join(self.raw_input, file) 
                     for file in os.listdir(self.raw_input) 
                     if file.endswith('.pkl')]
    
        print(f"Found {len(file_paths)} .pkl files to process")
        
        processed_count = 0
        error_count = 0

        # Process files sequentially for now (parallel processing can be added later)
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
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
    processor = CleanFilesProcessor()
    processor.run()