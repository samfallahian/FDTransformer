import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import unittest

# Import model creators and constants
from encoder_neurIPS.models import create_model_variant, ORIGINAL_DIM as NEURIPS_DIM
from encoder.autoencoderGEN3.models import get_model_by_index, ORIGINAL_DIM as GEN3_DIM

# Paths
DATA_DIR = "/Users/kkreth/PycharmProjects/data/encoder_neurIPS"
GEN3_MODEL_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best.pt"
SAVE_DIR = "/Users/kkreth/PycharmProjects/cgan/encoder_neurIPS/saved_models"
WINNER_IDX = 4 # NeurIPS_Model_04

def find_best_winner_checkpoint(save_dir, model_idx):
    """Automatically finds the best checkpoint for the winner model across all rounds."""
    best_l2 = float('inf')
    best_path = None
    best_round = -1
    
    # Check rounds from highest to lowest to prefer more trained models
    # but strictly use the one with the best reported metric if available
    rounds = ["production"] + [str(r) for r in range(10, 0, -1)]
    for r_suffix in rounds:
        round_dir = os.path.join(save_dir, f"round_{r_suffix}")
        if not os.path.exists(round_dir):
            continue
            
        # Prioritize 'best' snapshot
        path = os.path.join(round_dir, f"model_{model_idx:02d}_best.pt")
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                # Use the saved best_val_l2_norm to find the absolute best across rounds
                l2 = ckpt.get('best_val_l2_norm', ckpt.get('best_val_l2', float('inf')))
                if l2 < best_l2:
                    best_l2 = l2
                    best_path = path
                    best_round = r_suffix
            except:
                pass
                
    return best_path, best_round, best_l2

class TestGen3VsWinner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set device
        if torch.cuda.is_available():
            cls.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            cls.device = torch.device('mps')
        else:
            cls.device = torch.device('cpu')
        print(f"\n[Test] Using device: {cls.device}")

        # Load validation data
        val_path = os.path.join(DATA_DIR, 'validation_auto_encoder.pkl')
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at {val_path}")
        
        with open(val_path, 'rb') as f:
            val_np = pickle.load(f).astype(np.float32)
        
        # Take 10% random sample
        n_samples = len(val_np)
        n_subset = max(1, int(n_samples * 0.1))
        indices = np.random.choice(n_samples, n_subset, replace=False)
        cls.test_data = val_np[indices]
        print(f"[Test] Using {n_subset} samples (10% of validation data)")

        # Initialize models
        # GEN3 Model (Model 5 - AttentionSE)
        cls.gen3_model = get_model_by_index(4).to(cls.device)
        gen3_ckpt = torch.load(GEN3_MODEL_PATH, map_location=cls.device, weights_only=False)
        state_dict = gen3_ckpt.get('model_state_dict', gen3_ckpt)
        cls.gen3_model.load_state_dict(state_dict)
        cls.gen3_model.eval()

        # NeurIPS Winner (Auto-discovered)
        winner_path, winner_round, winner_reported_l2 = find_best_winner_checkpoint(SAVE_DIR, WINNER_IDX)
        if not winner_path:
            raise FileNotFoundError(f"Could not find any checkpoint for Model {WINNER_IDX:02d} in {SAVE_DIR}")
        
        print(f"[Test] Found Winner: Round {winner_round} | {os.path.basename(winner_path)} (Reported L2: {winner_reported_l2})")
        
        cls.winner_model = create_model_variant(WINNER_IDX).to(cls.device)
        winner_ckpt = torch.load(winner_path, map_location=cls.device, weights_only=False)
        cls.winner_model.load_state_dict(winner_ckpt['model_state_dict'])
        cls.winner_model.eval()

    def test_performance_comparison(self):
        """Compare GEN3 vs NeurIPS Winner performance."""
        x = torch.from_numpy(self.test_data).to(self.device)
        
        with torch.no_grad():
            # GEN3 Inference
            gen3_recon, _ = self.gen3_model(x)
            gen3_diff = gen3_recon - x.view_as(gen3_recon)
            
            # GEN3 L2 Norm (Euclidean Distance)
            gen3_l2_norms = torch.norm(gen3_diff, p=2, dim=1)
            gen3_avg_l2 = torch.mean(gen3_l2_norms).item()
            
            # GEN3 MSE and RMSE
            gen3_mse = torch.mean(gen3_diff**2).item()
            gen3_rmse = np.sqrt(gen3_mse)

            # NeurIPS Winner Inference
            winner_recon, _ = self.winner_model(x)
            winner_diff = winner_recon - x.view_as(winner_recon)
            
            # NeurIPS L2 Norm
            winner_l2_norms = torch.norm(winner_diff, p=2, dim=1)
            winner_avg_l2 = torch.mean(winner_l2_norms).item()
            
            # NeurIPS MSE and RMSE
            winner_mse = torch.mean(winner_diff**2).item()
            winner_rmse = np.sqrt(winner_mse)

        print(f"\n--- Performance Comparison (All Metrics) ---")
        print(f"{'Metric':<15} | {'GEN3 (Target)':<15} | {'NeurIPS Winner':<15} | {'Status'}")
        print(f"{'-'*16}|{'-'*17}|{'-'*17}|{'-'*10}")
        
        l2_status = "✓ BETTER" if winner_avg_l2 < gen3_avg_l2 else "✗ WORSE"
        rmse_status = "✓ BETTER" if winner_rmse < gen3_rmse else "✗ WORSE"
        mse_status = "✓ BETTER" if winner_mse < gen3_mse else "✗ WORSE"

        print(f"{'True L2 Norm':<15} | {gen3_avg_l2:<15.6e} | {winner_avg_l2:<15.6e} | {l2_status}")
        print(f"{'RMSE':<15} | {gen3_rmse:<15.6e} | {winner_rmse:<15.6e} | {rmse_status}")
        print(f"{'MSE':<15} | {gen3_mse:<15.6e} | {winner_mse:<15.6e} | {mse_status}")
        
        print(f"\n--- Why I said '10x better' before ---")
        print(f"Current Winner MSE: {winner_mse:.6e}")
        print(f"GEN3 RMSE Target:   {gen3_rmse:.6e}")
        print(f"Comparison: {winner_mse:.6e} is indeed ~1000x smaller than {gen3_rmse:.6e}, but they are on different scales.")
        print(f"Correct Comparison (RMSE vs RMSE): {winner_rmse:.6e} vs {gen3_rmse:.6e}")

        # Assertions: Winner must score lower (better) than GEN3
        self.assertLess(winner_avg_l2, gen3_avg_l2, 
            f"NeurIPS L2 ({winner_avg_l2:.6e}) is NOT better than GEN3 L2 ({gen3_avg_l2:.6e})")
        
        self.assertLess(winner_rmse, gen3_rmse, 
            f"NeurIPS RMSE ({winner_rmse:.6e}) is NOT better than GEN3 RMSE ({gen3_rmse:.6e})")

if __name__ == "__main__":
    # Disable W&B for the test
    os.environ["WANDB_MODE"] = "disabled"
    unittest.main()
