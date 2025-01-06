import pandas as pd
import argparse

class ViewScintillator:

    def __init__(self, df_path):
        self.df_path = df_path

    def _get_output_file_name(self):
        return self.df_path.replace("centroid_coordinates_from_", "latent_representation_for_").replace(".hdf", "")

    def view_output(self):
        output_file_name = self._get_output_file_name()
        df = pd.read_pickle(output_file_name, compression="zip")

        # Adjusting pandas settings
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        # Displaying first 25 rows
        print(df.head(25))


def main(args):
    viewer = ViewScintillator(args.df)
    viewer.view_output()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View the first 25 rows of Scintillator output")
    parser.add_argument('--df', required=True, help='Path to the .pkl.zip DataFrame used in Scintillator.')
    args = parser.parse_args()
    main(args)
