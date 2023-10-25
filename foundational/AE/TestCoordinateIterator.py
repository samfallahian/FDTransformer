import pandas as pd

import argparse
import pandas as pd


def extract_dataframe_from_zip(file_path):
    """
    Extract DataFrame from a .pkl.zip file.

    Args:
        file_path (str): Path to the .pkl.zip file.

    Returns:
        DataFrame: The extracted DataFrame.
    """
    return pd.read_pickle(file_path, compression="zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DataFrame from a .pkl.zip file")
    parser.add_argument('--file', required=True, help='Path to the .pkl.zip file.')
    args = parser.parse_args()

    df = extract_dataframe_from_zip(args.file)
    print(df.head())  # Print the first 5 rows to verify


