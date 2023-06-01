'''
Remember, from the "33_Learn..." tranformations logic we found:
High of 2.6400
Low of -1.9800
We will use that here to transform on demand any nacent values
'''
import numpy as np

class FloatConverter:
    def __init__(self):
        self.min_value = -1.9800
        self.max_value = 2.6400
        self.scale = 1.0 / (self.max_value - self.min_value)
        self.shift = -self.min_value * self.scale

    def convert(self, value):
        if isinstance(value, (int, float)):
            # Single float value
            converted_value = value * self.scale + self.shift
            if converted_value < self.min_value or converted_value > self.max_value:
                raise ValueError("Value out of range")
            return converted_value
        elif isinstance(value, np.ndarray):
            # Array of float values
            converted_values = value * self.scale + self.shift
            if np.any(converted_values < self.min_value) or np.any(converted_values > self.max_value):
                raise ValueError("Value out of range")
            return converted_values
        else:
            raise TypeError("Unsupported input type")

