import pandas as pd
import numpy as np
import torch
from AE.VAE import VAE
import argparse

"""
This file scintilates or "pulls together" several items to produce a final table of valid targets for the VAE. All 
output should have sufficient 'bread crumbs' to completely recreate (or test) the output.

When testing locally I used:
--df /Users/kkreth/PycharmProjects/data/DL-PTV/3p6/centroid_coordinates_from_1199.hdf.pkl.zip --hdf_raw_table /Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1199.hdf --model_path /Users/kkreth/PycharmProjects/cgan/foundational/AE/vae_model_epoch_14.pth
On the cluster, it will look slightly different (with SLURM substitutions for numbers), but basically this:
--df /home/kkreth_umassd_edu/DL-PTV/3p6/centroid_coordinates_from_1199.hdf.pkl.zip 
--hdf_raw_table /home/kkreth_umassd_edu/DL-PTV/3p6/1199.hdf 
--model_path /home/kkreth_umassd_edu/cgan/foundational/AE/vae_model_epoch_14.pth

--df /home/kkreth_umassd_edu/DL-PTV/3p6/centroid_coordinates_from_1199.hdf.pkl.zip --hdf_raw_table /home/kkreth_umassd_edu/DL-PTV/3p6/1199.hdf  --model_path /home/kkreth_umassd_edu/cgan/foundational/AE/vae_model_epoch_14.pth

"""

class Scintillator:

    def __init__(self, df_path, hdf_path, model_path):
        self.df_path = df_path
        self.df = self._load_dataframe(df_path)
        self.hdf_raw_table = self._load_hdf_table(hdf_path)
        self.vae_model = self._load_model(model_path)

    def _load_dataframe(self, file_path):
        return pd.read_pickle(file_path, compression="zip")

    def _load_hdf_table(self, file_path):
        return pd.read_hdf(file_path, key='processed_data/table')

    def _load_model(self, model_path):
        model = VAE()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def _get_v_values(self, x, y, z):
        matching_rows = self.hdf_raw_table[(self.hdf_raw_table['x'] == x) &
                                           (self.hdf_raw_table['y'] == y) &
                                           (self.hdf_raw_table['z'] == z)]

        if len(matching_rows) > 1:
            raise Exception("Too many rows")
        elif len(matching_rows) == 0:
            raise Exception(f"No matching rows found for x, y, z of {x}, {y}, {z}")

        vx_scalar = float(matching_rows['vx'].iloc[0])
        vy_scalar = float(matching_rows['vy'].iloc[0])
        vz_scalar = float(matching_rows['vz'].iloc[0])

        return vx_scalar, vy_scalar, vz_scalar

    def get_latent_representation(self, input_tensor):
        with torch.no_grad():
            mu, logvar = self.vae_model.encode(input_tensor.view(1, -1))
            z = self.vae_model.reparameterize(mu, logvar)
        return z

    def process(self):
        exception_messages = []
        total_rows = len(self.df)
        latent_representations = []

        for idx, (_, row) in enumerate(self.df.iterrows()):
            raw_input_tensor = np.zeros((3, 125))
            for i in range(125):
                x, y, z = row[f'centroid_vector_{i:03}']
                try:
                    raw_input_tensor[:, i] = self._get_v_values(x, y, z)
                except Exception as e:
                    exception_messages.append(str(e))

            input_tensor = torch.tensor(raw_input_tensor, dtype=torch.float32)
            latent_representation = self.get_latent_representation(input_tensor)
            latent_representations.append(latent_representation.numpy())

            if (idx + 1) % 1000 == 0:
                percentage_complete = (idx + 1) / total_rows * 100
                print(f"Processed {idx + 1} rows ({percentage_complete:.2f}% complete)")

        if exception_messages:
            print("\nExceptions encountered during processing:")
            for exception in exception_messages:
                print(exception)

        # Add latent representations to dataframe and save
        self.df['latent_representation'] = latent_representations
        new_file_name = self.df_path.replace("centroid_coordinates_from_", "latent_representation_for_").replace(".hdf", "")
        self.df.to_pickle(new_file_name, compression="zip")


def main(args):
    scintillator = Scintillator(args.df, args.hdf_raw_table, args.model_path)
    scintillator.process()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Scintillator data")
    parser.add_argument('--df', required=True, help='Path to the .pkl.zip DataFrame.')
    parser.add_argument('--hdf_raw_table', required=True, help='Path to the HDF file with processed_data key.')
    parser.add_argument('--model_path', required=True, help='Path to the saved VAE model.')
    args = parser.parse_args()
    main(args)
