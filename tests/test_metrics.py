import os
import torch
import sys
import unittest
import numpy as np

# Add project root and transformer_neurIPS to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_production_transformer import Config, mse_loss

class TestMetrics(unittest.TestCase):
    def test_mse_loss(self):
        # Basic MSE test
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(mse_loss(a, b).item(), 0.0)
        
        c = torch.tensor([2.0, 3.0, 4.0])
        # (1^2 + 1^2 + 1^2) / 3 = 1.0
        self.assertEqual(mse_loss(a, c).item(), 1.0)

    def test_persistence_logic(self):
        # Setup dummy data
        B, T_context, C = 2, 12, 47
        T_rollout = 28
        
        # Last context frame (step 12)
        last_frame = torch.ones(B, 1, C) * 5.0
        
        # Target that changes
        targets = torch.ones(B, T_rollout, C) * 6.0
        
        # Model predictions that are better than persistence
        model_preds = torch.ones(B, T_rollout, C) * 5.8
        
        # Persistence Baseline (repeat last_frame)
        persistence_preds = last_frame.repeat(1, T_rollout, 1)
        
        p_mse = mse_loss(persistence_preds, targets).item()
        m_mse = mse_loss(model_preds, targets).item()
        
        # p_mse = (6-5)^2 = 1.0
        # m_mse = (6-5.8)^2 = 0.2^2 = 0.04
        
        self.assertAlmostEqual(p_mse, 1.0, places=5)
        self.assertAlmostEqual(m_mse, 0.04, places=5)
        
        improvement = (p_mse - m_mse) / (p_mse + 1e-8) * 100
        # (1.0 - 0.04) / 1.0 * 100 = 96%
        self.assertAlmostEqual(improvement, 96.0, delta=1e-4)

if __name__ == "__main__":
    unittest.main()
