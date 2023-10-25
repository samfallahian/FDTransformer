import pandas as pd
import numpy as np

class Scintillator:

    def __init__(self, df_path, hdf_path):
        self.df = self._load_dataframe(df_path)
        self.hdf_raw_table = self._load_hdf_table(hdf_path)

    def _load_dataframe(self, file_path):
        return pd.read_pickle(file_path, compression="zip")

    def _load_hdf_table(self, file_path):
        return pd.read_hdf(file_path, key='processed_data/table')

    def _get_v_values(self, x, y, z):
        matching_rows = self.hdf_raw_table[(self.hdf_raw_table['x'] == x) &
                                           (self.hdf_raw_table['y'] == y) &
                                           (self.hdf_raw_table['z'] == z)]

        # Assert that there's only one matching row
        if len(matching_rows) > 1:
            raise Exception("Too many rows")
        elif len(matching_rows) == 0:
            raise Exception(f"No matching rows found for x, y, z of {x}, {y}, {z}")

        # Extract the values directly
        vx_scalar = float(matching_rows['vx'].iloc[0])
        vy_scalar = float(matching_rows['vy'].iloc[0])
        vz_scalar = float(matching_rows['vz'].iloc[0])

        return vx_scalar, vy_scalar, vz_scalar

    def process(self):
        exception_messages = []
        total_rows = len(self.df)

        for idx, (_, row) in enumerate(self.df.iterrows()):
            raw_input_tensor = np.zeros((3, 125))
            for i in range(125):
                x, y, z = row[f'centroid_vector_{i:03}']
                try:
                    raw_input_tensor[:, i] = self._get_v_values(x, y, z)
                except Exception as e:
                    exception_messages.append(str(e))

            # Print progress every 1,000 rows
            if (idx + 1) % 1000 == 0:
                percentage_complete = (idx + 1) / total_rows * 100
                print(f"Processed {idx + 1} rows ({percentage_complete:.2f}% complete)")

        if exception_messages:
            print("\nExceptions encountered during processing:")
            for exception in exception_messages:
                print(exception)


import argparse

def main(args):
    scintillator = Scintillator(args.df, args.hdf_raw_table)
    scintillator.process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Scintillator data")
    parser.add_argument('--df', required=True, help='Path to the .pkl.zip DataFrame.')
    parser.add_argument('--hdf_raw_table', required=True, help='Path to the HDF file with processed_data key.')
    args = parser.parse_args()
    main(args)
