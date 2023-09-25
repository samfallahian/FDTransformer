import pandas as pd
import numpy as np


class CoordinateAnalyzer:
    def __init__(self, dataframe):
        """
        Initialize the CoordinateAnalyzer object with a dataframe.
        Args:
        dataframe (pd.DataFrame): Input dataframe with columns 'x', 'y', 'z', etc.
        """
        self.dataframe = dataframe
        # Dictionary to store the sorted_rows DataFrames with keys as (x, y, z) tuples.
        self.sorted_rows_dict = {}

    def get_nearest_values(self, x, y, z):
        """
        Get nearest values to a given x, y, z coordinate from the dataframe.
        Args:
        x, y, z (float): The x, y, z coordinates
        Returns:
        sorted_rows (pd.DataFrame): A sorted dataframe of the nearest values
        """
        # Get unique sorted values from the dataframe
        x_values = sorted(set(self.dataframe['x']))
        y_values = sorted(set(self.dataframe['y']))
        z_values = sorted(set(self.dataframe['z']))


        # Get index of the provided x, y, z in the sorted lists
        x_index = x_values.index(x)
        y_index = y_values.index(y)
        z_index = z_values.index(z)

        # Create slices around the indexes, handling edge cases
        x_range = slice(max(x_index - 2, 0), min(x_index + 3, len(x_values)))
        y_range = slice(max(y_index - 2, 0), min(y_index + 3, len(y_values)))
        z_range = slice(max(z_index - 2, 0), min(z_index + 3, len(z_values)))

        # Select rows from dataframe that fall within the slices
        selected_rows = self.dataframe[
            (self.dataframe['x'].isin(x_values[x_range])) &
            (self.dataframe['y'].isin(y_values[y_range])) &
            (self.dataframe['z'].isin(z_values[z_range]))
            ]

        # Check the number of selected rows
        if selected_rows.shape[0] != 125:
            raise Exception(f"Failed to find 125 nearest values. Instead, found {len(selected_rows)}")

        # Sort the selected rows and convert types for space efficiency
        sorted_rows = selected_rows.sort_values(by=['x', 'y', 'z'])

        #Need Integers for the coordinates
        sorted_rows['x'] = sorted_rows['x'].astype(np.uint8)
        sorted_rows['y'] = sorted_rows['y'].astype(np.uint8)
        sorted_rows['z'] = sorted_rows['z'].astype(np.uint8)

        sorted_rows['vx'] = sorted_rows['vx'].astype(np.float16)
        sorted_rows['vy'] = sorted_rows['vy'].astype(np.float16)
        sorted_rows['vz'] = sorted_rows['vz'].astype(np.float16)
        sorted_rows['distance'] = sorted_rows['distance'].astype(np.uint8)

        # Adding 'x', 'y', 'z' columns to sorted_rows DataFrame
        sorted_rows['x_coord'] = sorted_rows['x']
        sorted_rows['y_coord'] = sorted_rows['y']
        sorted_rows['z_coord'] = sorted_rows['z']

        # Store the sorted_rows DataFrame in the dictionary with key as (x, y, z) tuple.
        self.sorted_rows_dict[(x, y, z)] = sorted_rows


        return sorted_rows, self.sorted_rows_dict
