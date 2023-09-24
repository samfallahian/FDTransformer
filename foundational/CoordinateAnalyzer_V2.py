import pandas as pd
import numpy as np

class CoordinateAnalyzer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def get_nearest_values(self, x, y, z):
        x_values = sorted(set(self.dataframe['x']))
        y_values = sorted(set(self.dataframe['y']))
        z_values = sorted(set(self.dataframe['z']))

        x_index = x_values.index(x)
        y_index = y_values.index(y)
        z_index = z_values.index(z)

        x_range = slice(max(x_index - 2, 0), min(x_index + 3, len(x_values)))
        y_range = slice(max(y_index - 2, 0), min(y_index + 3, len(y_values)))
        z_range = slice(max(z_index - 2, 0), min(z_index + 3, len(z_values)))

        selected_rows = self.dataframe[
            (self.dataframe['x'].isin(x_values[x_range])) &
            (self.dataframe['y'].isin(y_values[y_range])) &
            (self.dataframe['z'].isin(z_values[z_range]))
            ]

        if selected_rows.shape[0] != 125:
            raise Exception("Failed to find 125 nearest values. Instead, we only found " + str(len(selected_rows)))

        # Sort the selected rows based on x, y, and z values
        sorted_rows = selected_rows.sort_values(by=['x', 'y', 'z'])

        # Convert `vx`, `vy`, and `vz` to 16-bit floats
        sorted_rows['vx'] = sorted_rows['vx'].astype(np.float16)
        sorted_rows['vy'] = sorted_rows['vy'].astype(np.float16)
        sorted_rows['vz'] = sorted_rows['vz'].astype(np.float16)

        # Convert `distance` to uint8
        sorted_rows['distance'] = sorted_rows['distance'].astype(np.uint8)

        return sorted_rows

    def get_all_combinations(self):
        unique_xyz = self.dataframe[['x', 'y', 'z']].drop_duplicates()
        result = pd.DataFrame(columns=self.dataframe.columns)

        for _, row in unique_xyz.iterrows():
            nearest_values = self.get_nearest_values(row['x'], row['y'], row['z'])
            result = pd.concat([result, nearest_values])

        return result
