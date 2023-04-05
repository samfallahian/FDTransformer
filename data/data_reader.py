import pandas as pd
from sklearn import preprocessing
from utils import helpers
import numpy as np


class DataReader:
    def __init__(self):
        # super.__init__()
        """ Load training configurations """
        config = helpers.Config()
        cfg = config.from_json("data")
        self.path = cfg.data_path

    def load_standardize_data(self, file_name):
        """ Read input file """
        ### PTV DATA
        df = pd.read_pickle(self.path + file_name + ".pkl", compression="zip")
        labels = df.drop(df.columns.difference(["x", "y", "z", "time"]), axis=1).to_numpy()
        data = df.drop(["x", "y", "z", "time"], axis=1).to_numpy()

        ### Wind data
        # df = pd.read_csv(self.path + file_name + ".csv")
        # labels = df.drop(df.columns.difference(["time_frame", "hrs", "farm"]), axis=1).to_numpy()
        # data = df.drop(["time_frame", "hrs", "farm", "date"], axis=1).to_numpy()

        """ Standardize data """
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # labels = scalar.fit_transform(labels)
        data = scalar.fit_transform(data)
        return data, labels, scalar

    def load_standardize_data_test(self, file_name):
        """ Read input file """
        ### PTV DATA
        df = pd.read_pickle(self.path + file_name + ".pkl", compression="zip")
        labels = df.drop(df.columns.difference(["x", "y", "z", "time"]), axis=1).to_numpy()
        data = df.drop(["x", "y", "z", "time"], axis=1).to_numpy()

        """ Standardize data """
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # labels = scalar.fit_transform(labels)
        data = scalar.fit_transform(data)
        return data, labels, scalar