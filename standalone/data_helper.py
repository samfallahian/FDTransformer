import pandas as pd
import torch
import pickle

class DataHelper:
    """
    Helper class containing static methods for data preprocessing and sampling.
    """

    @staticmethod
    def normalize_column(series, min_val, max_val):
        """
        Normalize a pandas Series to a range between 0 and 1 using min and max values.
        """
        tensor = torch.tensor(series.values, dtype=torch.float32)
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor.numpy()

    @staticmethod
    def save_to_pickle(df, file_path):
        """
        Save a DataFrame to a pickle file.
        """
        df.to_pickle(file_path)
