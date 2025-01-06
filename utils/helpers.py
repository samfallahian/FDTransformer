import json
from configs import config as cfg
from datetime import datetime
import os.path
import pandas as pd


class Config:
    def __init__(self):
        self.cfg = cfg.CFG

    def from_json(self, data_kind):
        """Creates config from json"""
        params = json.loads(json.dumps(self.cfg), object_hook=PythonObject)
        if data_kind == "data":
            return params.data
        elif data_kind == "model":
            return params.model
        else:
            return params.training


class PythonObject(object):
    """Helper class to convert json into Python object"""

    def __init__(self, dict_):
        self.__dict__.update(dict_)


class Log:

    def __init__(self, file_name="default"):
        self.file_name = file_name

    def write_file(self, data):
        with open(os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d %H%M%S')}-{self.file_name}.json"),
                  "w") as f:
            json.dump(data, f, indent=4)

    def save_result(self, df):
        df.to_csv(f"logs/{datetime.now().strftime('%Y-%m-%d %H%M%S')}-{self.file_name}.csv", index=False)
