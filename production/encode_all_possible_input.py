import os
import pandas as pd
import json
import argparse
from encoder.model_WAE_01 import WAE
import logging
import torch
import warnings
from multiprocessing import Pool, cpu_count
from production.adjacent_coordinate import adjacent_coordinate
logging.basicConfig(level=logging.ERROR)

'''
Should create a single output from a single input
--input_hd5_file=/Users/kkreth/PycharmProjects/data/DL-PTV/5p2/1.hdf
--model_path=/Users/kkreth/PycharmProjects/cgan/saved_models/WAE_0043_Latent_model_at_epoch_100.pth
--meta_project_data_JSON=/Users/kkreth/PycharmProjects/cgan/configs/Umass_experiments.txt
'''

warnings.filterwarnings("ignore", "To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).")
warnings.filterwarnings("ignore", category=UserWarning, module='torch.tensor')
LATENT_SIZE = 47


class InsufficientCoordinatesException(Exception):
    """Exception raised when coordinate array does not contain sufficient number of elements."""

    def __init__(self, coordinate):
        self.message = f"Insufficient coordinates. Expected 3, but received {len(coordinate)}: {coordinate}."
        super().__init__(self.message)


class TooManyRowsException(Exception):
    """Exception raised when more than one row matches a given criteria"""

    def __init__(self, x, y, z):
        self.message = f"More than one row found for coordinates x={x}, y={y}, z={z}. Expected only one row."
        super().__init__(self.message)


class NoRowsFoundException(Exception):
    """Exception raised when no rows match a given criteria."""

    def __init__(self, x, y, z):
        self.message = f"No rows found for coordinates x={x}, y={y}, z={z}. At least one row is expected."
        super().__init__(self.message)


class Hd5ReaderForAE:
    def __init__(self, input_hd5_file: str, model_path: str, meta_project_data_JSON: str, hdf_key: str = 'processed_data'):
        self.input_hd5_file = input_hd5_file
        self.meta_project_data_JSON = meta_project_data_JSON
        self.model_path = model_path
        self.meta_project_data = self._load_json(meta_project_data_JSON)
        self.hdf_raw_table =  self.read_hdf_file(hdf_key)
        self.hdf_raw_table['x'] = self.hdf_raw_table['x'].astype(int)
        self.hdf_raw_table['y'] = self.hdf_raw_table['y'].astype(int)
        self.hdf_raw_table['z'] = self.hdf_raw_table['z'].astype(int)
        self.new_complete_df = pd.DataFrame(columns=['x', 'y', 'z'] + [f'latent_{i}' for i in range(LATENT_SIZE)])
        self.model = WAE()
        self.experiment_name = os.path.basename(os.path.dirname(input_hd5_file))
        self.finder = adjacent_coordinate(meta_project_data_JSON, self.experiment_name)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()

    @staticmethod
    def _load_json(json_path: str) -> dict:
        with open(json_path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict

    def read_hdf_file(self, hdf_key: str):
        if not os.path.isfile(self.input_hd5_file):
            raise FileNotFoundError(f'{self.input_hd5_file} does not exist.')
        df = pd.read_hdf(self.input_hd5_file, key=hdf_key)
        return df

    def _get_v_values(self, x, y, z):
        matching_rows = self.hdf_raw_table[(self.hdf_raw_table['x'] == x) &
                                           (self.hdf_raw_table['y'] == y) &
                                           (self.hdf_raw_table['z'] == z)]
        if len(matching_rows) > 1:
            raise TooManyRowsException(x, y, z)
        if len(matching_rows) == 0:
            raise NoRowsFoundException(x, y, z)
        vx_scalar = float(matching_rows['vx'].iloc[0])
        vy_scalar = float(matching_rows['vy'].iloc[0])
        vz_scalar = float(matching_rows['vz'].iloc[0])
        return_keys = self.finder.find_adjacent_coordinates(x, y, z)
        return_values = []
        for key in return_keys:
            matching_rows = self.hdf_raw_table[(self.hdf_raw_table['x'] == int(key[0])) &
                                               (self.hdf_raw_table['y'] == int(key[1])) &
                                               (self.hdf_raw_table['z'] == int(key[2]))]
            if not matching_rows.empty:
                vx = matching_rows['vx'].values[0]
                vy = matching_rows['vy'].values[0]
                vz = matching_rows['vz'].values[0]
                return_values.append([vx, vy, vz])
        v = torch.tensor(return_values, dtype=torch.float32)
        return v

    def encode_and_save_row(self, row):
        self.finder = adjacent_coordinate(self.meta_project_data_JSON, self.experiment_name)
        try:
            coordinates_3p6 = self.finder.find_adjacent_coordinates(row.x.astype(int), row.y.astype(int), row.z.astype(int))
        except ValueError:
            return None

        for coordinate in coordinates_3p6:
            try:
                if len(coordinate) < 3:
                    raise InsufficientCoordinatesException(coordinate)
                v_tuple = self._get_v_values(coordinate[0], coordinate[1], coordinate[2])
                if v_tuple is None:
                    continue
                v = torch.flatten(v_tuple.clone().detach())

                v_decoded = self.model.decode(self.model.encode(v))
                logging.debug(f"v: {v}")
                logging.debug(f"v decoded: {v_decoded}")

                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    v_decoded = self.model.decode(self.model.encode(v))
                    logging.debug(f"v: {v}")
                    logging.debug(f"v decoded: {v_decoded}")

                if v.size() != torch.Size([375]):
                    raise IndexError(f'Invalid tensor size {v.size()}, expected size: (375)')
                latent_rep = self.model.encode(v)
                new_row = pd.Series([coordinate[0], coordinate[1], coordinate[2]] + latent_rep.tolist(), index=self.new_complete_df.columns)
                return pd.DataFrame(new_row).transpose()
            except (ValueError, InsufficientCoordinatesException, IndexError, ArithmeticError) as e:
                return None

    def process_entire_file(self):
        with Pool(cpu_count() - 1) as p:
            all_encoded_rows = p.map(self.encode_and_save_row, [row for _, row in self.hdf_raw_table.iterrows()])
        non_empty_rows = [df for df in all_encoded_rows if df is not None and not df.empty]
        if non_empty_rows:  # Added conditional here to check if non_empty_rows is not empty
            self.new_complete_df = pd.concat(non_empty_rows, ignore_index=True)
            base_file_name = os.path.splitext(os.path.basename(self.input_hd5_file))[0]
            output_filename = f"latent_representation_for_{base_file_name}.pkl"
            file_dir = os.path.dirname(self.input_hd5_file)
            self.new_complete_df.to_pickle(file_dir + '/' + output_filename)



def main():

    parser = argparse.ArgumentParser(description='Hd5ReaderForAE')
    parser.add_argument('--input_hd5_file', type=str, required=True, help='The path to the input HDF5 file.')
    parser.add_argument('--model_path', type=str, required=True, help='The path to the model.')
    parser.add_argument('--meta_project_data_JSON', type=str, required=True, help='The path to the Meta Project Data JSON.')
    args = parser.parse_args()

    reader = Hd5ReaderForAE(args.input_hd5_file, args.model_path, args.meta_project_data_JSON)
    reader.process_entire_file()  # Correct method name here
    #print(reader.hdf_raw_table)

    # Save the DataFrame as a pickle file in the same directory as input_hd5_file
    directory = os.path.dirname(os.path.realpath(args.input_hd5_file))
    #output_file = os.path.join(directory, 'pickle_file.pkl')
    #reader.hdf_raw_table.to_pickle(output_file)
    #print(f'Pickle file has been saved in {output_file}')


if __name__ == "__main__":
    main()