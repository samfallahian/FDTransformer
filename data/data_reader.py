import pandas as pd
from sklearn import preprocessing
from utils import helpers


class DataReader:
    def __init__(self):
        # super.__init__()
        """ Load training configurations """
        config = helpers.Config()
        cfg = config.from_json("data")
        self.path = cfg.data_path


    def load_standardize_data(self, file_name):
        """ Read input file """
        df = pd.read_pickle(self.path + file_name + ".pkl", compression="zip")
        labels = df.drop(df.columns.difference(["label"]), axis=1).to_numpy()
        data = df.drop(["label"], axis=1).to_numpy()
        """ Standardize data """
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # labels = scalar.fit_transform(labels)
        data = scalar.fit_transform(data)
        return data, labels
    @staticmethod
    def de_standardize(transformed_values):
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        return scalar.inverse_transform(transformed_values)
