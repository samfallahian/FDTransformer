'''
Remember, from the "33_Learn..." tranformations logic we found:
High of 2.6400
Low of -1.9800
We will use that here to transform on demand any nacent values
'''

class FloatConverter:
    def __init__(self):
        self.min_value = -1.9800
        self.max_value = 2.6400
        self.scale = 1.0 / (self.max_value - self.min_value)
        self.shift = -self.min_value * self.scale

    def convert(self, value):
        converted_value = value * self.scale + self.shift
        if converted_value < self.min_value or converted_value > self.max_value:
            raise ValueError("Value out of range")
        return converted_value
