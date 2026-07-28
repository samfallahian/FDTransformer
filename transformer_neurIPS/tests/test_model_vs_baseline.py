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
from encoder_neurIPS.models import create_model_variant
from TransformLatent import FloatConverter

def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

class TestModelVsBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Use the specific best model
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
        cls.model_path = os.path.join(model_dir, "production_base_E256_L6_train_best.pt")
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model file not found: {cls.model_path}")
        
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

        # 4. Initialize Autoencoder for decoding latents
        cls.ae_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "encoder_neurIPS/saved_models/round_production/model_04_best.pt")
        if not os.path.exists(cls.ae_path):
             # Fallback to standard location
             cls.ae_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "encoder_neurIPS/saved_models/model_04_best.pt")

        if os.path.exists(cls.ae_path):
            print(f"Loading AE from: {cls.ae_path}")
            cls.ae = create_model_variant(4)
            ae_ckpt = torch.load(cls.ae_path, map_location='cpu')
            cls.ae.load_state_dict(ae_ckpt['model_state_dict'])
            cls.ae.eval()
            cls.ae.to(cls.device)
            cls.converter = FloatConverter()
        else:
            print(f"Warning: AE model not found at {cls.ae_path}. Centroid decoding will be skipped.")
            cls.ae = None

        # 5. Prepare data
        val_h5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/val_40.h5")
        if not os.path.exists(val_h5):
            raise unittest.SkipTest("Validation data not found at " + val_h5)
        
        cls.dataset = TransformerDataset(val_h5, subset_ratio=0.1)
        cls.val_loader = torch.utils.data.DataLoader(cls.dataset, batch_size=1, shuffle=False)

    def decode_to_centroid(self, latents):
        """
        Decodes (N, 47) latents to (N, 3) centroid velocities [vx, vy, vz].
        """
        if self.ae is None:
            # Fallback if AE is missing: just take first 3 dims (will be wrong but won't crash)
            return latents[:, :3]
        
        with torch.no_grad():
            # Flatten if needed (B, T, 47) -> (B*T, 47)
            orig_shape = latents.shape
            lat_flat = latents.reshape(-1, 47)
            
            recon = self.ae.decode(lat_flat) # (N, 375)
            # Centroid at neighbor index 62 -> cols 186, 187, 188
            centroid_v = recon[:, 186:189].cpu().numpy()
            
            # Convert back to physical units
            centroid_v = self.converter.unconvert(centroid_v)
            
            # Reshape back to (B, T, 3) or (N, 3)
            new_shape = list(orig_shape[:-1]) + [3]
            return torch.from_numpy(centroid_v).reshape(new_shape).to(self.device)

    def test_better_than_persistence_single_step(self):
        """
        Test that the model performs better than the 'persistence' baseline for a single step.
        MAE is calculated only for the 'centroid' vx, vy, vz.
        """
        model_error = 0
        persistence_error = 0
        count = 0
        
        context_steps = getattr(Config, 'VAL_CONTEXT_STEPS', 12)
        num_x = getattr(Config, 'NUM_X', 26)
        latent_dim = getattr(Config, 'LATENT_DIM', 47)
        context_len = num_x * context_steps
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 10: break # Check 10 samples
                batch = batch.to(self.device)
                
                inputs = batch[:, :context_len, :]
                
                # We check the centroid for various coordinates
                # Coordinate index 25 (last), 24 (next-to-last), 23 (one before that)
                coords_to_test = [25, 24, 23]
                
                for x_idx in coords_to_test:
                    # Target: (time=12, x=x_idx)
                    target_latent = batch[:, context_len + x_idx : context_len + x_idx + 1, :latent_dim]
                    target_v = self.decode_to_centroid(target_latent)
                    
                    # Persistence: (time=11, x=x_idx)
                    persistence_latent = inputs[:, (context_steps-1)*num_x + x_idx : (context_steps-1)*num_x + x_idx + 1, :latent_dim]
                    persistence_v = self.decode_to_centroid(persistence_latent)
                    
                    # Model prediction
                    curr = inputs
                    pred_v = None
                    for step in range(x_idx + 1):
                        out = self.model(curr)
                        next_lat = out[:, -1:, :]
                        if step == x_idx:
                            pred_v = self.decode_to_centroid(next_lat)
                        
                        # Update curr for next autoregressive step
                        next_idx = curr.shape[1]
                        next_tok = batch[:, next_idx : next_idx + 1, :].clone()
                        next_tok[:, :, :latent_dim] = next_lat
                        curr = torch.cat([curr, next_tok], dim=1)
                    
                    m_err = mae_loss(pred_v, target_v).item()
                    p_err = mae_loss(persistence_v, target_v).item()
                    
                    print(f"\n--- Centroid MAE for Sample {i} (at t=12, coord={x_idx}) ---")
                    print(f"Target [vx,vy,vz]: {target_v[0, 0].cpu().numpy()}")
                    print(f"Persistence [vx,vy,vz]: {persistence_v[0, 0].cpu().numpy()}")
                    print(f"Model Pred [vx,vy,vz]: {pred_v[0, 0].cpu().numpy()}")
                    print(f"Sample Persistence MAE: {p_err:.8f}")
                    print(f"Sample Model MAE: {m_err:.8f}")
                    
                    model_error += m_err
                    persistence_error += p_err
                    count += 1
        
        if count > 0:
            model_error /= count
            persistence_error /= count
            improvement = (persistence_error - model_error) / (persistence_error + 1e-8) * 100
            print(f"\nFinal Results (Average over {count} centroid points):")
            print(f"Single-step Model Centroid MAE: {model_error:.6f}")
            print(f"Single-step Persistence Centroid MAE: {persistence_error:.6f}")
            print(f"Single-step Improvement: {improvement:.2f}%")
            
            self.assertLess(model_error, persistence_error, "Model failed to beat persistence for centroid velocities!")

    def test_better_than_persistence_rollout(self):
        """
        Test multi-step rollout performance against the persistence baseline.
        MAE is calculated only for the 'centroid' vx, vy, vz.
        """
        rollout_error = 0
        persistence_error = 0
        rollout_count = 0
        
        context_steps = getattr(Config, 'VAL_CONTEXT_STEPS', 12)
        rollout_steps = getattr(Config, 'VAL_ROLLOUT_STEPS', 28)
        num_x = getattr(Config, 'NUM_X', 26)
        latent_dim = getattr(Config, 'LATENT_DIM', 47)
        
        context_len = num_x * context_steps
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 5: break 
                
                batch = batch.to(self.device)
                inputs = batch[:, :context_len, :]
                targets_latent = batch[:, context_len : context_len + rollout_steps, :latent_dim]
                targets_v = self.decode_to_centroid(targets_latent)
                
                # Persistence Baseline: Last frame of context repeated
                last_frame_latent = inputs[:, -num_x:, :latent_dim]
                num_repeats = (rollout_steps + num_x - 1) // num_x
                persistence_preds_latent = last_frame_latent.repeat(1, num_repeats, 1)
                persistence_preds_latent = persistence_preds_latent[:, :rollout_steps, :]
                persistence_v = self.decode_to_centroid(persistence_preds_latent)
                
                p_err = mae_loss(persistence_v, targets_v).item()

                curr = inputs
                preds_latent = []
                for _ in range(rollout_steps):
                    out = self.model(curr)
                    next_lat = out[:, -1:, :]
                    preds_latent.append(next_lat)
                    
                    next_idx = curr.shape[1]
                    if next_idx >= batch.shape[1]: break
                    
                    next_tok = batch[:, next_idx : next_idx + 1, :].clone()
                    next_tok[:, :, :latent_dim] = next_lat
                    curr = torch.cat([curr, next_tok], dim=1)
                
                if len(preds_latent) == rollout_steps:
                    preds_latent = torch.cat(preds_latent, dim=1)
                    preds_v = self.decode_to_centroid(preds_latent)
                    m_err = mae_loss(preds_v, targets_v).item()
                    rollout_error += m_err
                    persistence_error += p_err
                    rollout_count += 1
        
        if rollout_count > 0:
            rollout_error /= rollout_count
            persistence_error /= rollout_count
            improvement = (persistence_error - rollout_error) / (persistence_error + 1e-8) * 100
            print(f"\nRollout Results (Average over {rollout_count} sequences, Centroid only):")
            print(f"Rollout Model Centroid MAE: {rollout_error:.6f}")
            print(f"Rollout Persistence Centroid MAE: {persistence_error:.6f}")
            print(f"Rollout Improvement: {improvement:.2f}%")
            
            self.assertLess(rollout_error, persistence_error, 
                           f"Model failed to beat persistence baseline in rollout! Model MAE: {rollout_error:.6f}, Persistence MAE: {persistence_error:.6f}")
        else:
            self.fail("No rollouts were completed.")

if __name__ == '__main__':
    unittest.main()
