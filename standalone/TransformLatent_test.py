import unittest
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


    def test_OOB(self):
        input_native = -11.95
        converter = TransformLatent.FloatConverter()
        with self.assertRaises(ValueError):
            converted_value = converter.convert(input_native)

if __name__ == '__main__':
    unittest.main()

