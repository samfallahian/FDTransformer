import unittest
import numpy as np
import os
import h5py

class TestPrepareData(unittest.TestCase):
    def setUp(self):
        self.data_path = "transformer_neurIPS/data/train_40.h5"
        if not os.path.exists(self.data_path):
             self.skipTest(f"Data file {self.data_path} not found. Run prepare_data.py first.")
             
    def test_dtypes_and_content(self):
        with h5py.File(self.data_path, 'r') as f:
            data = f['data'][:]
            # data shape: (N, 40, 26, 52)
            
            # Check y and z coordinates (indices 48 and 49)
            # They should be integers (though stored in float32 array, they should be whole numbers)
            y_coords = data[:, :, :, 48]
            z_coords = data[:, :, :, 49]
            
            np.testing.assert_array_equal(y_coords, y_coords.astype(int), "Y coordinates should be integers")
            np.testing.assert_array_equal(z_coords, z_coords.astype(int), "Z coordinates should be integers")
            
            # Check for non-triviality: Latents (indices 0-46) should not be all zeros
            latents = data[:, :, :, :47]
            for i in range(min(10, len(data))): # Check a sample
                self.assertFalse(np.all(latents[i] == 0), f"Sequence {i} contains only zero latents")

            # Check that we don't have too many zeros in latents overall
            # I'm OK with a fraction of say 1% zeros...assert that and report out actual percentage found.
            zero_fraction = np.mean(latents == 0)
            print(f"\nActual zero fraction in latents: {zero_fraction:.4%}")
            self.assertLess(zero_fraction, 0.01, f"Way too many zeros in latents: {zero_fraction:.4%}")

if __name__ == "__main__":
    unittest.main()
