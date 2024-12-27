import os
import sys
from data_helper import DataHelper
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

class ExperimentDataWrapper:
    """
    Wrapper class for managing and preprocessing experiment data.
    """

    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
        self.cleaned_data = None

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_and_clean_data(self):
        """
        Load data from the file, drop unused columns, and clean column types.
        """
        print("Loading and cleaning data...")
        data = pd.read_pickle(f"/work/pi_bseyedaghazadeh_umassd_edu/DL-PTV.backup/3p6.pkl", compression='zip')

        # Drop unnecessary columns
        data.drop(columns=["px", "py", "pz"], errors="ignore", inplace=True)

        # Define column data types
        column_types = {
            "x": "int32",
            "y": "int32",
            "z": "int32",
            "vx": "float32",
            "vy": "float32",
            "vz": "float32",
            "time": "int32",
            "distance": "int32",
        }

        for col, dtype in column_types.items():
            if col in data.columns:
                data[col] = data[col].astype(dtype)

        self.cleaned_data = data
        print(self.cleaned_data.head())
        print("Data loaded and cleaned.")

    def normalize_data(self):
        """
        Normalize velocity columns and store normalized values.
        """
        print("Normalizing data...")
        normalization_params = {
            "vx": (-1.11, 2.64),
            "vy": (-1.98, 2.20),
            "vz": (-1.22, 1.10),
        }

        for col, (min_val, max_val) in normalization_params.items():
            if col in self.cleaned_data.columns:
                self.cleaned_data[f"{col}_norm"] = DataHelper.normalize_column(
                    self.cleaned_data[col], min_val, max_val
                )
        print("Normalization complete.")

    def split_and_sample(self, sample_size=1000):
        """
        Split data by time steps, sample rows, and write to pickle files.
        """
        print("Splitting data by time and sampling...")
        unique_times = sorted(self.cleaned_data["time"].unique())

        for idx, time_step in enumerate(unique_times, start=1):
            time_data = self.cleaned_data[self.cleaned_data["time"] == time_step]
            print(time_data.head())

            if len(time_data) > sample_size:
                sampled_data = time_data.sample(n=sample_size, random_state=42)
            else:
                sampled_data = time_data
            print(sampled_data.head())
            output_file = os.path.join(self.output_dir, f"{idx}.pkl")
            DataHelper.save_to_pickle(sampled_data, output_file)

        print(f"Data split into {len(unique_times)} files with sampling. Saved to {self.output_dir}.")


if __name__ == "__main__":
    file_path = "/work/pi_bseyedaghazadeh_umassd_edu/DL-PTV.backup/3p6.pkl"
    output_dir = "../data/3p6/"
    wrapper = ExperimentDataWrapper(file_path, output_dir)
    wrapper.load_and_clean_data()
    wrapper.normalize_data()
    wrapper.split_and_sample(sample_size=1000)
