import argparse
import pandas as pd
import numpy as np
import os
from foundational.AnalyzerOfCoordinates import AnalyzerOfCoordinates

"""
For reasons that I don't get, this wouldn't run on the cluters (outside of Pycharm) without this:
export PYTHONPATH=$PYTHONPATH:/home/kkreth_umassd_edu/cgan/:/home/kkreth_umassd_edu/cgan/foundational

--json "/home/kkreth_umassd_edu/cgan/configs/Umass_experiments.txt" --hdf "/home/kkreth_umassd_edu/DL-PTV/3p6/1.hdf"
"""


class CoordinateIterator:
    DEFAULT_JSON_FILE_LOCATION = "/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt"
    #DEFAULT_JSON_FILE_LOCATION = "/home/kkreth_umassd_edu/cgan/configs/Umass_experiments.txt"
    DEFAULT_HDF_FILE = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/200.hdf"
    #DEFAULT_HDF_FILE = "/home/kkreth_umassd_edu/DL-PTV/3p6/200.hdf"
    """
    --json /Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt --hdf /Users/kkreth/PycharmProjects/data/DL-PTV/3p6/200.hdf
    """


    def __init__(self, json_file_location=None, hdf_file=None):
        self.json_file_location = json_file_location or self.DEFAULT_JSON_FILE_LOCATION
        self.hdf_file = hdf_file or self.DEFAULT_HDF_FILE
        self.df = pd.read_hdf(self.hdf_file)
        self.pruneCoordinates()

    def pruneCoordinates(self):
        analyzer = AnalyzerOfCoordinates(self.json_file_location, self.hdf_file)

        def get_coordinates(row):
            try:
                return analyzer.provide_coordinates_ordered_list(row['x'], row['y'], row['z'])
            except ValueError as e:
                if str(e) == "coordinate not found":
                    return np.nan
                raise

        self.df['centroid_vector'] = self.df.apply(get_coordinates, axis=1)
        self.df.dropna(subset=['centroid_vector'], inplace=True)

        # Transform centroid_vector into 125 separate columns
        expanded_df = self.df['centroid_vector'].apply(pd.Series)
        expanded_df.columns = [f'centroid_vector_{str(col).zfill(3)}' for col in expanded_df.columns]

        # Concatenate the new columns to the original DataFrame and drop the centroid_vector column
        self.df = pd.concat([self.df, expanded_df], axis=1).drop(columns='centroid_vector')

        # Saving the pruned DataFrame
        path, filename = os.path.split(self.hdf_file)
        new_filename = os.path.join(path, f"centroid_coordinates_from_{filename}.pkl.zip")
        #self.df.to_hdf(new_filename, key='df', mode='w', complevel=9, complib='blosc')
        self.df.to_pickle(new_filename, compression="zip")

    def basic_iterator(self, x, y, z):
        analyzer = AnalyzerOfCoordinates(self.json_file_location, self.hdf_file)
        coordinates_ordered_list = analyzer.provide_coordinates_ordered_list(x, y, z)
        print(coordinates_ordered_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinate Iterator")
    parser.add_argument('--json', default=CoordinateIterator.DEFAULT_JSON_FILE_LOCATION,
                        help='JSON file location. Default: ' + CoordinateIterator.DEFAULT_JSON_FILE_LOCATION)
    parser.add_argument('--hdf', default=CoordinateIterator.DEFAULT_HDF_FILE,
                        help='HDF file location. Default: ' + CoordinateIterator.DEFAULT_HDF_FILE)
    args = parser.parse_args()

    iterator = CoordinateIterator(args.json, args.hdf)
    x, y, z = -113.0, -68.0, -17.0
    iterator.basic_iterator(x, y, z)
