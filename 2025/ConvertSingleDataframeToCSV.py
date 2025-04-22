import os
import pandas as pd


class PickleToCsvConverter:
    """
    A class to convert a compressed pickle (.pkl) dataframe to a CSV file.

    This utility handles different compression methods (None, zip, gzip).
    """

    def __init__(self):
        """Initialize the converter."""
        pass

    def read_pickle_file(self, file_path):
        """
        Read a pickle file with different compression methods.

        Args:
            file_path (str): Path to the pickle file.

        Returns:
            pandas.DataFrame or None: The dataframe from the pickle file or None if reading fails.
        """
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

    def convert(self, pickle_file_path, csv_file_path=None, index=False):
        """
        Convert a pickle file to CSV.

        Args:
            pickle_file_path (str): Path to the pickle file.
            csv_file_path (str, optional): Path where the CSV file will be saved.
                          If None, it will use the same name as the pickle file with .csv extension.
            index (bool, optional): Whether to include the index in the CSV file. Default is False.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
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

        # Write to CSV
        try:
            df.to_csv(csv_file_path, index=index)
            print(f"Successfully converted {pickle_file_path} to {csv_file_path}")
            return True
        except Exception as e:
            print(f"Error writing to CSV {csv_file_path}: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    converter = PickleToCsvConverter()

    # Convert a pickle file to CSV
    pickle_path = "/Users/kkreth/PycharmProjects/data/all_data_cleaned_dtype_correct/7p8.pkl"
    csv_path = "/Users/kkreth/PycharmProjects/data/all_data_cleaned_dtype_correct/7p8.csv"  # Optional

    # Without specifying CSV path (will use same name as pickle file but with .csv extension)
    converter.convert(pickle_path)

    # Or with explicit CSV path
    # converter.convert(pickle_path, csv_path)