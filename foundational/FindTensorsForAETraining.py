import os
import json
import random
import argparse
from typing import List, Tuple
import numpy as np
import h5py
from AnalyzerOfCoordinates import AnalyzerOfCoordinates


class FindTensorsForAETraining:
    def __init__(self, json_file_location: str, hdf_file: str):
        self.json_file_location = json_file_location
        self.hdf_file = hdf_file
        # Derive output_file_index from hdf_file name
        self.output_file_index = os.path.splitext(os.path.basename(hdf_file))[0]
        self.analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file)
        # Creating the output file path in the same directory as hdf_file
        self.output_file = os.path.join(os.path.dirname(hdf_file), f"tensor_{self.output_file_index}.hdf")

    def get_random_coordinates(self, num_rows: int = 1000) -> List[Tuple[int, int, int]]:
        directory_name = os.path.basename(os.path.dirname(self.hdf_file))

        # Loading the JSON file
        with open(self.json_file_location, 'r') as json_file:
            data = json.load(json_file)

        try:
            # Extracting the enumerated values from the JSON using the directory name as the key
            x_enumerated = sorted(data[directory_name]['x_enumerated'])[2:-2]
            y_enumerated = sorted(data[directory_name]['y_enumerated'])[2:-2]
            z_enumerated = sorted(data[directory_name]['z_enumerated'])[2:-2]
        except KeyError:
            print(f"The key {directory_name} does not exist in JSON data")
            return []

        # Generating a list of random coordinates
        coordinates_list = [(random.choice(x_enumerated), random.choice(y_enumerated), random.choice(z_enumerated))
                            for _ in range(num_rows)]
        return coordinates_list

    def run(self):
        random_coordinates = self.get_random_coordinates()

        # Opening the hdf file to read the dataset and output HDF file to write
        with h5py.File(self.hdf_file, 'r') as hdf, h5py.File(self.output_file, 'w') as hdf_out:
            dataset = hdf['processed_data/table']

            for idx, (x, y, z) in enumerate(random_coordinates):
                try:
                    # Analyzing the coordinates and storing the results as a new dataset in the output HDF5 file
                    result_df = self.analyzer.analyze(x, y, z)
                    tensor = result_df[['vx', 'vy', 'vz']].to_numpy()
                    dataset_name = f"{os.path.basename(os.path.dirname(self.hdf_file))}_{self.output_file_index}_{x}_{y}_{z}"
                    hdf_out.create_dataset(dataset_name, data=tensor)
                except Exception as e:
                    print(f"Error occurred with coordinates x: {x}, y: {y}, z: {z}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find Tensors for AE Training.')
    parser.add_argument('json_file_location', type=str, help='Location of the JSON file.')
    parser.add_argument('hdf_file', type=str, help='Location of the HDF file.')

    args = parser.parse_args()

    tensor_finder = FindTensorsForAETraining(args.json_file_location, args.hdf_file)
    tensor_finder.run()
