import pandas as pd
import json
import itertools
from typing import List
import os
import unittest

pd.set_option('display.max_columns', None)

class AnalyzerOfCoordinates:
    def __init__(self, json_file_location: str, hdf_file: str = None, key: str = None):
        self.json_file_location = json_file_location
        self.hdf_file = hdf_file

        if hdf_file is not None and key is None:
            dir_name = os.path.basename(os.path.dirname(hdf_file))
            self.key = dir_name.split('.')[0]
        elif key is not None:
            self.key = key
        else:
            raise ValueError("Either hdf_file or key must be provided.")

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
        if self.hdf_file is None:
            raise ValueError("HDF file has not been set.")

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

    def provide_coordinates_ordered_list(self, x: float, y: float, z: float) -> List[List[float]]:
        x_values = self.locate_adjacent_values(x, self.x_enumerated)
        y_values = self.locate_adjacent_values(y, self.y_enumerated)
        z_values = self.locate_adjacent_values(z, self.z_enumerated)
        return list(itertools.product(x_values, y_values, z_values))

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

        # Columns will look like this:
        # (-85,16,-5), (-85,16,-1), (-85,16,3) ... (-70,31,7), (-70,31,11)
        # But these will be individual columns (with a naming convention) so:
        # -85 will be one column of data called x1
        # 16 would be y1
        # -5 would be z1
        # ...
        # -70 will be x125
        # 31 will be y125
        # 11 will be z125

        self.assertTrue(all(value in result_df['x'].values for value in expected_x_values))
        self.assertTrue(all(value in result_df['y'].values for value in expected_y_values))
        self.assertTrue(all(value in result_df['z'].values for value in expected_z_values))

    def test_analyze_10p4(self):
        json_file_location = "/path/to/your/json/file.txt"  # Adjust this path
        hdf_file = "/path/to/your/10p4/experiment/hdf/file.hdf"  # Adjust this path for the 10p4 experiment
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file=hdf_file, key='10p4')

        result_df = analyzer.analyze_10p4(-34.0, 4.0, -17.0)
        print(result_df)

    def test_provide_coordinates_ordered_list(self):
        json_file_location = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
        hdf_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/300.hdf"
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file)

        x, y, z = -113.0, -68.0, -17.0
        coordinates_ordered_list = analyzer.provide_coordinates_ordered_list(x, y, z)

        centroid_coordinates = (x, y, z)

        expected_coordinates = (-121.0, -68.0, -25.0)
        # Add other expected coordinates as needed...

        self.assertEqual(coordinates_ordered_list[10], expected_coordinates)

    def test_provide_coordinates_center_center(self):
        json_file_location = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
        hdf_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/300.hdf"
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file)

        x, y, z = -113.0, -68.0, -17.0
        coordinates_ordered_list = analyzer.provide_coordinates_ordered_list(x, y, z)

        centroid_coordinates = (x, y, z)
        # Add other expected coordinates as needed...

        expected_coordinates = (-113.0, -68.0, -17.0)
        # Add other expected coordinates as needed...

        self.assertEqual(coordinates_ordered_list[62], expected_coordinates)

        # Expecting a TypeError because the method is called without mandatory arguments
        with self.assertRaises(TypeError):
            coordinates_ordered_list = analyzer.provide_coordinates_ordered_list()

    def test_analyzer_edge_case_1(self):
        json_file_location = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
        hdf_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1.hdf"
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file=hdf_file)

        x, y, z = -46.0, -0.0, -33.0
        with self.assertRaises(ValueError) as context:
            analyzer.analyze(x, y, z)

        self.assertIn("coordinate not found", str(context.exception))

    def test_analyzer_edge_case_2(self):
        json_file_location = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
        hdf_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1.hdf"
        analyzer = AnalyzerOfCoordinates(json_file_location, hdf_file=hdf_file)

        x, y, z = -46.0, -0.0, -33.0
        with self.assertRaises(ValueError) as context:
            analyzer.provide_coordinates_ordered_list(x, y, z)

        self.assertIn("coordinate not found", str(context.exception))


if __name__ == '__main__':
    unittest.main()
