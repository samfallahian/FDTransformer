import os
import torch
import sys
import unittest
import time
import shutil

# Add project root and transformer_neurIPS to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_production_transformer import Config

class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for checkpoints during tests
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_saved_models")
        os.makedirs(self.test_dir, exist_ok=True)
        self.original_checkpoint_dir = Config.CHECKPOINT_DIR
        Config.CHECKPOINT_DIR = self.test_dir

    def tearDown(self):
        Config.CHECKPOINT_DIR = self.original_checkpoint_dir
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_checkpoint_creation(self):
        # Create a dummy model and save it using the logic from train_production_transformer
        from model_variants import get_model
        
        # Adjust config for small test model
        Config.EMBED_SIZE = 64
        Config.N_LAYERS = 2
        Config.N_HEADS = 4
        
        model = get_model(Config)
        run_name = "test_run"
        
        # 1. Test 'latest' checkpoint save
        cp_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")
        torch.save({
            'epoch': 0,
            'batch_idx': 0,
            'model_state_dict': model.state_dict(),
        }, cp_path)
        
        self.assertTrue(os.path.exists(cp_path), "Latest checkpoint was not created")
        
        # 2. Test 'rollout_best' checkpoint save
        save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_rollout_best.pt")
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'rollout_l2': 0.5,
            'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
        }, save_path)
        
        self.assertTrue(os.path.exists(save_path), "Rollout best checkpoint was not created")
        
        # 3. Verify content
        checkpoint = torch.load(save_path)
        self.assertIn('config', checkpoint)
        self.assertEqual(checkpoint['config']['EMBED_SIZE'], 64)

if __name__ == "__main__":
    unittest.main()
