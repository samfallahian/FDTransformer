from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd


class TestPickleAnalyzer(HostPreferences):
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

    def analyze_pickle_files(self):
        """Analyze all pickle files in the input directory."""
        print(f"\nAnalyzing pickle files in: {self.output_directory}")

        # Get list of pickle files
        file_paths = [os.path.join(self.output_directory, file)
                      for file in os.listdir(self.output_directory)
                      if file.endswith('.pkl')]

        print(f"Found {len(file_paths)} .pkl files to analyze")

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            print(f"\nAnalyzing file: {filename}")

            df = self.read_pickle_file(file_path)
            if df is not None:
                print("\nColumns and their dtypes:")
                for col, dtype in df.dtypes.items():
                    print(f"Column: {col:<20} dtype: {dtype}")

                print(f"\nTotal rows: {len(df)}")
            else:
                print(f"Failed to read file: {filename}")

    def run(self):
        self.analyze_pickle_files()


if __name__ == "__main__":
    analyzer = TestPickleAnalyzer()
    analyzer.run()