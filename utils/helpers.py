import json
from configs import config as cfg


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
