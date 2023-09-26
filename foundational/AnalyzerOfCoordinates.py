import pandas as pd
import json
import itertools
from typing import List
import os
import unittest


class AnalyzerOfCoordinates:
    def __init__(self, json_file_location: str, hdf_file: str):
        self.json_file_location = json_file_location
        self.hdf_file = hdf_file

        # Extract directory name from hdf_file
        dir_name = os.path.basename(os.path.dirname(hdf_file))
        self.key = dir_name.split('.')[0]

        with open(json_file_location, 'r') as file:
            json_data = json.load(file)

        if self.key not in json_data:
            raise ValueError(f"Key {self.key} not found in JSON file.")

        self.x_enumerated = json_data[self.key]['x_enumerated']
        self.y_enumerated = json_data[self.key]['y_enumerated']
        self.z_enumerated = json_data[self.key]['z_enumerated']

    def locate_adjacent_values(self, value: float, enumerated_list: List[float]) -> List[float]:
        sorted_list = sorted(enumerated_list)
        index = sorted_list.index(value)
        if index < 2 or index >= len(sorted_list) - 2:
            raise ValueError("coordinate not found")

        return sorted_list[index - 2:index + 3]

    def analyze(self, x: float, y: float, z: float) -> pd.DataFrame:
        x_values = self.locate_adjacent_values(x, self.x_enumerated)
        y_values = self.locate_adjacent_values(y, self.y_enumerated)
        z_values = self.locate_adjacent_values(z, self.z_enumerated)

        df = pd.read_hdf(self.hdf_file)

        all_combinations = list(itertools.product(x_values, y_values, z_values))
        if len(all_combinations) != 125:
            raise ValueError("coordinates not compatible, possible edge case")

        result_df = df[df.apply(lambda row: (row['x'], row['y'], row['z']) in all_combinations, axis=1)]

        if result_df.empty or len(result_df) != 125:
            raise ValueError("coordinates not compatible, possible edge case")

        return result_df.sort_values(by=['x', 'y', 'z'])


class TestAnalyzerOfCoordinates(unittest.TestCase):
    def test_analyzer(self):
        json_file_location = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
        hdf_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1.hdf"
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file)

        x, y, z = -78.0, 23.0, 3.0
        result_df = analyzer.analyze(x, y, z)

        expected_x_values = [-85, -81, -78, -74, -70]
        expected_y_values = [16, 20, 23, 27, 31]
        expected_z_values = [-5, -1, 3, 7, 11]

        self.assertTrue(all(value in result_df['x'].values for value in expected_x_values))
        self.assertTrue(all(value in result_df['y'].values for value in expected_y_values))
        self.assertTrue(all(value in result_df['z'].values for value in expected_z_values))


if __name__ == '__main__':
    unittest.main()

