import unittest
import numpy as np
from standalone import TransformLatent

class MyTestCase(unittest.TestCase):
    def test_upper(self):
        input_native = 2.54
        converter = TransformLatent.FloatConverter()
        converted_value = converter.convert(input_native)
        self.assertEqual(0.9783549783549783, converted_value)

    def test_lower(self):
        input_native = -1.95
        converter = TransformLatent.FloatConverter()
        converted_value = converter.convert(input_native)
        self.assertEqual(0.006493506493506496, converted_value)

    def test_multi(self):
        input_native = np.array([-1.95, 2.54], dtype=float)
        converter = TransformLatent.FloatConverter()
        converted_values = converter.convert(input_native)
        expected_values = np.array([0.006493506493506496, 0.9783549783549783])
        np.testing.assert_array_equal(expected_values, converted_values)

    def test_OOB(self):
        input_native = -11.95
        converter = TransformLatent.FloatConverter()
        with self.assertRaises(ValueError):
            converted_value = converter.convert(input_native)

if __name__ == '__main__':
    unittest.main()

