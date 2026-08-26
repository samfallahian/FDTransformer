import os
import sys
import time
import pickle
import threading
import subprocess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
import logging

# Add current directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import create_model_variant, ORIGINAL_DIM, LATENT_DIM

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/kkreth/PycharmProjects/data/encoder_neurIPS"
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models")
PROD_SAVE_DIR = os.path.join(SAVE_DIR, "round_production")
UNIT_TEST_PATH = os.path.join(SCRIPT_DIR, "unit_tests", "test_gen3_vs_winner.py")
WINNER_IDX = 4  # NeurIPS_Model_04 (Switched to Model 4 after Model 20)
BATCH_SIZE = 4096  # Reduced from 12288 for stability (Choice 1)
LEARNING_RATE = 5e-5 # Reduced from 1e-4 for stability (Choice 1)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProductionTrain")

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def run_unit_test():
    """Background task to run unit test and log results in color."""
    try:
        # Set PYTHONPATH to project root for the subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
        env["WANDB_MODE"] = "disabled"
        
        result = subprocess.run(
            [sys.executable, UNIT_TEST_PATH],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # GREEN PASS
            logger.info("\033[92m[UNIT TEST] STATUS: PASS (Winner beats GEN3!)\033[0m")
        else:
            # RED FAIL
            logger.info("\033[91m[UNIT TEST] STATUS: FAIL (GEN3 still holds the crown)\033[0m")
            # Log the comparison table from the output if possible
            for line in result.stdout.split('\n'):
                if '|' in line or 'Metric' in line:
                    logger.info(f"  {line}")
    except Exception as e:
        logger.error(f"Error running unit test thread: {e}")

def main():
    device = get_device()
    logger.info(f"Starting Production Training for Model {WINNER_IDX:02d} on {device}")
    
    # Data Loading
    train_path = os.path.join(DATA_DIR, 'training_auto_encoder.pkl')
    val_path = os.path.join(DATA_DIR, 'validation_auto_encoder.pkl')
    with open(train_path, 'rb') as f:
        train_np = pickle.load(f).astype(np.float32)
    with open(val_path, 'rb') as f:
        val_np = pickle.load(f).astype(np.float32)
        
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Initialization
    model = create_model_variant(WINNER_IDX).to(device)
    os.makedirs(PROD_SAVE_DIR, exist_ok=True)
    
    # Seeding logic: Production > Scan Round 4 down to 1 (Winner 4 starts from R4)
    best_val_l2_norm = float('inf')
    seed_path = None
    prod_best = os.path.join(PROD_SAVE_DIR, f"model_{WINNER_IDX:02d}_best.pt")
    
    if os.path.exists(prod_best):
        seed_path = prod_best
        logger.info(f"Seeding from previous Production run: {seed_path}")
    else:
        # Scan rounds 4 down to 1 for Model 4
        for r in range(4, 0, -1):
            round_best = os.path.join(SAVE_DIR, f"round_{r}", f"model_{WINNER_IDX:02d}_best.pt")
            if os.path.exists(round_best):
                seed_path = round_best
                logger.info(f"Seeding from Round {r}: {seed_path}")
                break
    
    if seed_path:
        ckpt = torch.load(seed_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        # BUG FIX: Load the previous best metric to prevent overwriting with worse models on restart
        if 'best_val_l2_norm' in ckpt:
            best_val_l2_norm = ckpt['best_val_l2_norm']
            logger.info(f"\033[94m[RESTORE] Historical Best L2 Norm loaded: {best_val_l2_norm:.6e}\033[0m")
        elif 'val_l2_norm' in ckpt: # Fallback for different naming
            best_val_l2_norm = ckpt['val_l2_norm']
            logger.info(f"\033[94m[RESTORE] Historical L2 Norm loaded: {best_val_l2_norm:.6e}\033[0m")
    else:
        logger.warning("No Production or Round 5 checkpoint found. Starting from scratch/random.")
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Add Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # Increased aggression
    
    wandb.init(project="encoder_neurIPS_production", name=f"NeurIPS_Model_{WINNER_IDX:02d}_PROD", config={
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "optimizer": "AdamW",
        "best_val_l2_on_start": best_val_l2_norm,
        "scheduler": "ReduceLROnPlateau",
        "model_candidate": "Model 4 (Production)"
    })
    prev_val_l2_norm = None
    prev_val_rmse = None
    epoch = 0
    
    def get_colored_metric(current, previous):
        if previous is None:
            return f"{current:.6e}"
        if current < previous:
            return f"\033[92m{current:.6e}\033[0m" # GREEN for down
        elif current > previous:
            return f"\033[91m{current:.6e}\033[0m" # RED for up
        else:
            return f"{current:.6e}"

    try:
        while True: # Train until user interrupts
            model.train()
            train_l2_norm = 0.0
            batch_count = 0
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                recon_x, z = model(x)
                
                diff = recon_x - x.view_as(recon_x)
                l2_norms = torch.norm(diff, p=2, dim=1)
                loss = torch.mean(l2_norms) + 0.00005 * torch.mean(z**2)
                
                loss.backward()
                # Add Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_l2_norm += torch.mean(l2_norms).item()
                batch_count += 1
                
            # Validation
            model.eval()
            total_l2 = 0.0
            total_mse = 0.0
            sample_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon_x, _ = model(x)
                    diff = recon_x - x.view_as(recon_x)
                    total_l2 += torch.sum(torch.norm(diff, p=2, dim=1)).item()
                    total_mse += torch.sum(diff**2).item()
                    sample_count += x.size(0)
            
            val_l2_norm = total_l2 / sample_count
            val_rmse = np.sqrt(total_mse / (sample_count * ORIGINAL_DIM))
            
            # Step scheduler
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_l2_norm)
            current_lr = optimizer.param_groups[0]['lr']
            
            if current_lr < old_lr:
                logger.info(f"\033[93m[LR DECAY] Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}\033[0m")
            
            l2_str = get_colored_metric(val_l2_norm, prev_val_l2_norm)
            rmse_str = get_colored_metric(val_rmse, prev_val_rmse)
            
            logger.info(f"Epoch {epoch} | LR: {current_lr:.2e} | Val L2: {l2_str} | Val RMSE: {rmse_str}")
            
            prev_val_l2_norm = val_l2_norm
            prev_val_rmse = val_rmse
            
            # Save and check best
            if val_l2_norm < best_val_l2_norm:
                improvement = ((best_val_l2_norm - val_l2_norm) / best_val_l2_norm * 100) if best_val_l2_norm != float('inf') else 100.0
                logger.info(f"\033[92m[CHECKPOINT] New absolute best achieved! {val_l2_norm:.6e} < {best_val_l2_norm:.6e} ({improvement:.4f}% improvement)\033[0m")
                best_val_l2_norm = val_l2_norm
                save_path = os.path.join(PROD_SAVE_DIR, f"model_{WINNER_IDX:02d}_best.pt")
                torch.save({
                    'model_idx': WINNER_IDX,
                    'model_state_dict': model.state_dict(),
                    'best_val_l2_norm': val_l2_norm,
                    'val_rmse': val_rmse,
                    'epoch': epoch,
                    'timestamp': time.time()
                }, save_path)
                logger.info(f"\033[94m[SAVE] Model conserved at: {save_path}\033[0m")
            else:
                gap = ((val_l2_norm - best_val_l2_norm) / best_val_l2_norm * 100) if best_val_l2_norm != float('inf') else 0.0
                logger.info(f"\033[93m[STAGNANT] Current L2 ({val_l2_norm:.6e}) did not beat best ({best_val_l2_norm:.6e}). Gap: +{gap:.4f}%\033[0m")
            
            wandb.log({
                "epoch": epoch, 
                "val_l2_norm": val_l2_norm, 
                "val_rmse": val_rmse,
                "lr": current_lr
            })
            
            # Spawn unit test thread
            threading.Thread(target=run_unit_test, daemon=True).start()
            
            epoch += 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
