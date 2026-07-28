import os
import torch
import numpy as np
import h5py
import unittest
import glob
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_neurIPS.train_production_transformer import Config, TransformerDataset, mse_loss, l2_loss
from transformer_neurIPS.model_variants import get_model

class TestModelVsBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Find the best model
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
        model_files = glob.glob(os.path.join(model_dir, "*_rollout_best.pt"))
        if not model_files:
            raise unittest.SkipTest("No 'rollout_best' model found in " + model_dir)
        
        # Pick the most recent one
        cls.model_path = max(model_files, key=os.path.getmtime)
        print(f"\nUsing model: {cls.model_path}")

        # 2. Load checkpoint and config
        cls.checkpoint = torch.load(cls.model_path, map_location='cpu')
        
        # Update Config with saved parameters
        if 'config' in cls.checkpoint:
            print("Loading config from checkpoint...")
            for k, v in cls.checkpoint['config'].items():
                print(f"  {k}: {v}")
                setattr(Config, k, v)
        else:
            print("No config found in checkpoint!")
        
        # 3. Initialize model
        cls.model = get_model(Config)
        cls.model.load_state_dict(cls.checkpoint['model_state_dict'], strict=False)
        cls.model.eval()
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model.to(cls.device)

        # 4. Prepare data
        val_h5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/val_40.h5")
        if not os.path.exists(val_h5):
            raise unittest.SkipTest("Validation data not found at " + val_h5)
        
        cls.dataset = TransformerDataset(val_h5, subset_ratio=0.1)
        cls.val_loader = torch.utils.data.DataLoader(cls.dataset, batch_size=1, shuffle=False)

    def test_better_than_persistence_single_step(self):
        """
        Test that the model performs better than the 'persistence' baseline for a single step.
        """
        model_l2 = 0
        persistence_l2 = 0
        count = 0
        
        context_steps = getattr(Config, 'VAL_CONTEXT_STEPS', 12)
        num_x = getattr(Config, 'NUM_X', 26)
        latent_dim = getattr(Config, 'LATENT_DIM', 47)
        context_len = num_x * context_steps
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 10: break 
                batch = batch.to(self.device)
                
                # Input is the context
                inputs = batch[:, :context_len, :]
                # Target is the very next frame
                target = batch[:, context_len : context_len + 1, :latent_dim]
                
                # Model prediction
                output = self.model(inputs)
                pred = output[:, -1:, :]
                
                # Persistence: frame at current x-location from the PREVIOUS time step
                # In this flattened sequence, if context_len = 12 * 26, 
                # then target is time 12, x 0.
                # Its persistence baseline is time 11, x 0.
                # Which is at index: context_len - 26
                last_frame = inputs[:, context_len - 26 : context_len - 25, :latent_dim]
                
                model_l2 += l2_loss(pred, target).item()
                persistence_l2 += l2_loss(last_frame, target).item()
                count += 1
        
        if count > 0:
            model_l2 /= count
            persistence_l2 /= count
            improvement = (persistence_l2 - model_l2) / (persistence_l2 + 1e-8) * 100
            print(f"\nSingle-step Model L2: {model_l2:.6f}")
            print(f"Single-step Persistence L2: {persistence_l2:.6f}")
            print(f"Single-step Improvement: {improvement:.2f}%")
            
            self.assertLess(model_l2, persistence_l2, "Model failed to beat persistence even for a single step!")

    def test_better_than_persistence_rollout(self):
        """
        Test multi-step rollout performance against the persistence baseline.
        """
        rollout_l2 = 0
        persistence_l2 = 0
        rollout_count = 0
        
        context_steps = getattr(Config, 'VAL_CONTEXT_STEPS', 12)
        rollout_steps = getattr(Config, 'VAL_ROLLOUT_STEPS', 28)
        num_x = getattr(Config, 'NUM_X', 26)
        latent_dim = getattr(Config, 'LATENT_DIM', 47)
        
        context_len = num_x * context_steps
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 1: break 
                
                batch = batch.to(self.device)
                inputs = batch[:, :context_len, :]
                targets = batch[:, context_len : context_len + rollout_steps, :latent_dim]
                
                # Persistence Baseline: Last frame of context repeated
                last_frame = inputs[:, -num_x:, :latent_dim]
                num_repeats = (rollout_steps + num_x - 1) // num_x
                persistence_preds = last_frame.repeat(1, num_repeats, 1)
                persistence_preds = persistence_preds[:, :rollout_steps, :]
                
                persistence_l2 += l2_loss(persistence_preds, targets).item()

                # Model Rollout
                curr = inputs
                preds = []
                for _ in range(rollout_steps):
                    out = self.model(curr)
                    next_lat = out[:, -1:, :]
                    preds.append(next_lat)
                    
                    next_idx = curr.shape[1]
                    if next_idx >= batch.shape[1]: break
                    
                    next_tok = batch[:, next_idx : next_idx + 1, :].clone()
                    next_tok[:, :, :latent_dim] = next_lat
                    curr = torch.cat([curr, next_tok], dim=1)
                
                if len(preds) == rollout_steps:
                    preds = torch.cat(preds, dim=1)
                    rollout_l2 += l2_loss(preds, targets).item()
                    rollout_count += 1
        
        if rollout_count > 0:
            rollout_l2 /= rollout_count
            persistence_l2 /= rollout_count
            improvement = (persistence_l2 - rollout_l2) / (persistence_l2 + 1e-8) * 100
            print(f"\nRollout Model L2: {rollout_l2:.6f}")
            print(f"Rollout Persistence L2: {persistence_l2:.6f}")
            print(f"Rollout Improvement: {improvement:.2f}%")
            
            self.assertLess(rollout_l2, persistence_l2, 
                           f"Model failed to beat persistence baseline in rollout! Model L2: {rollout_l2:.6f}, Persistence L2: {persistence_l2:.6f}")
        else:
            self.fail("No rollouts were completed.")

if __name__ == '__main__':
    unittest.main()
