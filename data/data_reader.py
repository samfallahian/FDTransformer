import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils import helpers


class DataReader:
    def __init__(self):
        super.__init__()
        """ Load training configurations """
        config = helpers.Config()
        cfg = config.from_json("data")
        df = pd.read_pickle(cfg.data_path + ".pkl", compression="zip")

    def load_standardize_data(self):
        pass
