'''
Remember, from the "33_Learn..." tranformations logic we found:
High of 2.6400
Low of -1.9800
We will use that here to transform on demand any nacent values
'''

import numpy as np
import pandas as pd

class FloatConverter:
    def __init__(self):
        self.min_value = -0.197745 #-1.9800
        self.max_value = 0.263599 #2.6400
        self.scale = 1.0 / (self.max_value - self.min_value)
        self.shift = -self.min_value * self.scale

    def convert(self, value):
        if isinstance(value, (int, float, np.float16)):
            # Single float value
            return value * self.scale + self.shift
        elif isinstance(value, np.ndarray):
            # Array of float values
            return value * self.scale + self.shift
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # DataFrame or Series of float values
            return value * self.scale + self.shift
        else:
            raise TypeError(f"Unsupported input type: {type(value)}, value: {value}")

    def unconvert(self, value):
        if isinstance(value, (int, float, np.float16)):
            # Single float value
            return (value - self.shift) / self.scale
        elif isinstance(value, np.ndarray):
            # Array of float values
            return (value - self.shift) / self.scale
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # DataFrame or Series of float values
            return (value - self.shift) / self.scale
        else:
            raise TypeError(f"Unsupported input type: {type(value)}, value: {value}")
