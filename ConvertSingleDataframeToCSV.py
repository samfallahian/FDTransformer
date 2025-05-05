import os
import pandas as pd
import time
import logging
import psutil


class PickleToCsvConverter:
    """
    A class to convert a compressed pickle (.pkl) dataframe to a CSV file.

    This utility handles different compression methods (None, zip, gzip).
    """

    def __init__(self, debug=False):
        """Initialize the converter."""
        # Set up logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        self.logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

    def read_pickle_file(self, file_path):
        """
        Read a pickle file with different compression methods.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            pandas.DataFrame or None: The dataframe from the pickle file or None if reading fails.
        """
        self.logger.debug(f"Starting to read pickle file: {file_path}")
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        self.logger.debug(f"File size: {file_size_mb:.2f} MB")
        
        # Try to determine compression method from file extension first
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.gz':
            compressions = ['gzip']
        elif file_ext == '.zip':
            compressions = ['zip']
        else:
            compressions = [None, 'zip', 'gzip']
            
        for compression in compressions:
            try:
                start_time = time.time()
                self.logger.debug(f"Trying to read with compression: {compression}")
                
                # Open the file and read it
                with open(file_path, 'rb') as f:
                    if compression:
                        df = pd.read_pickle(f, compression=compression)
                    else:
                        df = pd.read_pickle(f)
                
                # Log performance information
                duration = time.time() - start_time
                self.logger.debug(f"Successfully read pickle file with compression: {compression}")
                self.logger.debug(f"Time taken to read: {duration:.2f} seconds")
                self.logger.debug(f"DataFrame shape: {df.shape}")
                self._log_memory_usage()
                return df
                
            except Exception as e:
                self.logger.debug(f"Failed to read with compression {compression}: {str(e)}")
                if compression == compressions[-1]:  # If we've tried all methods
                    self.logger.error(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue

    def convert(self, pickle_file_path, csv_file_path=None, index=False, chunksize=100000):
        """
        Convert a pickle file to CSV.

        Args:
            pickle_file_path (str): Path to the pickle file.
            csv_file_path (str, optional): Path where the CSV file will be saved.
                          If None, it will use the same name as the pickle file with .csv extension.
            index (bool, optional): Whether to include the index in the CSV file. Default is False.
            chunksize (int, optional): Number of rows to write at once. Default is 100,000.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
        self.logger.info(f"Starting conversion of {pickle_file_path}")
        start_total = time.time()
        
        # Get the dataframe from the pickle file
        df = self.read_pickle_file(pickle_file_path)

        if df is None:
            return False

        # If csv_file_path is not provided, generate one
        if csv_file_path is None:
            base_name = os.path.splitext(os.path.basename(pickle_file_path))[0]
            output_dir = os.path.dirname(pickle_file_path)
            csv_file_path = os.path.join(output_dir, f"{base_name}.csv")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(csv_file_path)), exist_ok=True)

        # Write to CSV - optimize for large dataframes
        try:
            # Check if dataframe is very large and needs chunking
            if len(df) > chunksize:
                self.logger.info(f"Large dataframe detected ({len(df)} rows). Using chunked writing.")
                
                # Write the header
                df.iloc[:0].to_csv(csv_file_path, index=index)
                
                # Append chunks without header
                for i in range(0, len(df), chunksize):
                    start_chunk = time.time()
                    chunk = df.iloc[i:i+chunksize]
                    chunk.to_csv(csv_file_path, mode='a', header=False, index=index)
                    self.logger.debug(f"Wrote chunk {i//chunksize + 1} of {(len(df) + chunksize - 1)//chunksize} "
                                     f"({i}-{min(i+chunksize, len(df))}) in {time.time() - start_chunk:.2f} seconds")
                    self._log_memory_usage()
            else:
                # For smaller dataframes, write all at once
                start_write = time.time()
                df.to_csv(csv_file_path, index=index)
                self.logger.debug(f"Wrote CSV in {time.time() - start_write:.2f} seconds")
                
            total_time = time.time() - start_total
            self.logger.info(f"Successfully converted {pickle_file_path} to {csv_file_path}")
            self.logger.info(f"Total conversion time: {total_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to CSV {csv_file_path}: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    converter = PickleToCsvConverter(debug=True)  # Enable debug logging

    # Convert a pickle file to CSV
    pickle_path = "/Users/kkreth/PycharmProjects/data/all_data_broken_down_1200_each_directory/6p4/1200.pkl"

    # Without specifying CSV path (will use same name as pickle file but with .csv extension)
    converter.convert(pickle_path)