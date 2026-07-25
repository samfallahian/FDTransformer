import unittest
import numpy as np
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
sys.path.insert(0, PROJECT_ROOT)

from pySINDy.reproduce_all_params import get_vorticity_enstrophy, run_sindy_recovery

class TestEnstrophyChain(unittest.TestCase):
    def setUp(self):
        # Create some dummy velocity data (nx, ny, nz, 3)
        self.nx, self.ny, self.nz = 10, 10, 10
        self.x = np.linspace(0, 1, self.nx)
        self.y = np.linspace(0, 1, self.ny)
        self.z = np.linspace(0, 1, self.nz)
        
        # Simple velocity field: u = y, v = z, w = x
        self.V = np.zeros((self.nx, self.ny, self.nz, 3))
        yy, zz, xx = np.meshgrid(self.y, self.z, self.x, indexing='ij')
        # meshgrid with 'ij' returns (ny, nz, nx)
        # We want (nx, ny, nz)
        XX, YY, ZZ = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.V[..., 0] = YY
        self.V[..., 1] = ZZ
        self.V[..., 2] = XX

    def test_enstrophy_definition_consistency(self):
        """Test that get_vorticity_enstrophy implements the formula it is tested against."""
        wx, wy, wz, enstrophy = get_vorticity_enstrophy(self.V, self.x, self.y, self.z)
        
        # Expected enstrophy from the returned vorticity
        expected_enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
        
        # They should be identical because one is calculated from the other in the same function
        np.testing.assert_array_almost_equal(enstrophy, expected_enstrophy, decimal=15)
        print("\n[OK] Enstrophy is calculated directly from returned vorticity.")

    def test_sindy_recovery_on_perfect_data(self):
        """Test that SINDy recovers the 0.5 coefficients with near-zero MSE on self-consistent data."""
        wx, wy, wz, enstrophy = get_vorticity_enstrophy(self.V, self.x, self.y, self.z)
        results = run_sindy_recovery(wx, wy, wz, enstrophy)
        
        self.assertLess(results['MSE'], 1e-25)
        self.assertAlmostEqual(results['wx^2'], 0.5, places=10)
        self.assertAlmostEqual(results['wy^2'], 0.5, places=10)
        self.assertAlmostEqual(results['wz^2'], 0.5, places=10)
        print(f"\n[OK] SINDy MSE on perfect data: {results['MSE']:.2e}")

    def test_sindy_recovery_with_independent_enstrophy_noise(self):
        """
        Test SINDy when enstrophy has some independent noise.
        This simulates what would happen if enstrophy came from an independent 'measurement' 
        rather than being derived from the same vorticity components used as predictors.
        """
        wx, wy, wz, enstrophy = get_vorticity_enstrophy(self.V, self.x, self.y, self.z)
        
        # Add 1% noise to enstrophy
        noise_level = 0.01
        noise = np.random.normal(0, noise_level * np.mean(enstrophy), enstrophy.shape)
        noisy_enstrophy = enstrophy + noise
        
        results = run_sindy_recovery(wx, wy, wz, noisy_enstrophy)
        
        print(f"\n[INFO] SINDy MSE with 1% independent enstrophy noise: {results['MSE']:.2e}")
        # MSE should now be much higher than 1e-34
        self.assertGreater(results['MSE'], 1e-10)
        # Coefficients should still be around 0.5 if noise is zero-mean
        self.assertAlmostEqual(results['wx^2'], 0.5, delta=0.05)

    def test_sindy_recovery_with_vorticity_noise(self):
        """
        Test SINDy when vorticity components have noise but enstrophy is calculated from them.
        This represents the current state: even if the velocity/vorticity prediction is 'bad',
        as long as enstrophy is derived from it, SINDy will 'recover' the 0.5 coefficients 
        perfectly because the algebraic relationship is perfectly maintained in the data.
        """
        wx, wy, wz, _ = get_vorticity_enstrophy(self.V, self.x, self.y, self.z)
        
        # Add 10% noise to vorticity
        noise_level = 0.1
        wx_n = wx + np.random.normal(0, noise_level * np.abs(wx).max(), wx.shape)
        wy_n = wy + np.random.normal(0, noise_level * np.abs(wy).max(), wy.shape)
        wz_n = wz + np.random.normal(0, noise_level * np.abs(wz).max(), wz.shape)
        
        # Recalculate enstrophy from NOISY vorticity
        enstrophy_consistent = 0.5 * (wx_n**2 + wy_n**2 + wz_n**2)
        
        results = run_sindy_recovery(wx_n, wy_n, wz_n, enstrophy_consistent)
        
        print(f"\n[INFO] SINDy MSE with 10% vorticity noise (but consistent enstrophy): {results['MSE']:.2e}")
        # Even with 10% noise in the predictors, the MSE should still be near-zero
        # because the relationship is an identity in the data.
        self.assertLess(results['MSE'], 1e-25)
        self.assertAlmostEqual(results['wx^2'], 0.5, places=10)

    def test_data_identity_check(self):
        """Check if predicted data is actually different from raw data in the actual files."""
        raw_path = os.path.join(PROJECT_ROOT, "pySINDy/raw_data_grad.npz")
        pred_path = os.path.join(PROJECT_ROOT, "pySINDy/predicted_data_grad.npz")
        
        if not (os.path.exists(raw_path) and os.path.exists(pred_path)):
            print("\n[SKIP] Data files not found for identity check.")
            return
            
        raw = np.load(raw_path)
        pred = np.load(pred_path)
        
        # Check if V arrays are identical
        are_identical = np.array_equal(raw['V'], pred['V'])
        self.assertFalse(are_identical, "Predicted velocity data is IDENTICAL to raw velocity data!")
        
        diff = np.mean(np.abs(raw['V'] - pred['V']))
        print(f"\n[OK] Mean difference between Raw and Predicted Velocity: {diff:.2e}")
        self.assertGreater(diff, 0, "Predicted data should not be exactly the same as raw data.")

if __name__ == "__main__":
    unittest.main()
