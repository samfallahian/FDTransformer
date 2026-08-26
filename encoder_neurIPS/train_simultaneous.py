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
SIM_SAVE_DIR = os.path.join(SAVE_DIR, "simultaneous_training")
UNIT_TEST_PATH = os.path.join(SCRIPT_DIR, "unit_tests", "test_gen3_vs_winner.py")

CANDIDATES = [4, 28]  # 4th and 5th place (Skipping Model 20 which was 3rd)
BATCH_SIZE = 4096
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimultaneousTrain")

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_l2 = 0
    count = 0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon_x, z = model(x)
        diff = recon_x - x
        l2_norms = torch.norm(diff, p=2, dim=1)
        loss = torch.mean(l2_norms) + 0.00005 * torch.mean(z**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_l2 += torch.mean(l2_norms).item()
        count += 1
    return total_l2 / count

def validate(model, loader, device):
    model.eval()
    total_l2 = 0
    total_mse = 0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon_x, _ = model(x)
            diff = recon_x - x
            l2_norms = torch.norm(diff, p=2, dim=1)
            total_l2 += torch.sum(l2_norms).item()
            total_mse += torch.sum(diff**2).item()
            sample_count += x.size(0)
    
    avg_l2 = total_l2 / sample_count
    rmse = np.sqrt(total_mse / (sample_count * ORIGINAL_DIM))
    return avg_l2, rmse

def main():
    device = get_device()
    logger.info(f"Starting Simultaneous Training for Models {CANDIDATES} on {device}")
    
    # Data Loading
    train_path = os.path.join(DATA_DIR, 'training_auto_encoder.pkl')
    val_path = os.path.join(DATA_DIR, 'validation_auto_encoder.pkl')
    
    with open(train_path, 'rb') as f: train_np = pickle.load(f).astype(np.float32)
    with open(val_path, 'rb') as f: val_np = pickle.load(f).astype(np.float32)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=BATCH_SIZE, shuffle=False)
    
    models = {}
    optimizers = {}
    schedulers = {}
    best_l2s = {}
    history = {c: [] for c in CANDIDATES}
    
    os.makedirs(SIM_SAVE_DIR, exist_ok=True)
    
    for midx in CANDIDATES:
        model = create_model_variant(midx).to(device)
        
        # Seed from highest available round (4 down to 1)
        seed_path = None
        for r in range(4, 0, -1):
            round_best = os.path.join(SAVE_DIR, f"round_{r}", f"model_{midx:02d}_best.pt")
            if os.path.exists(round_best):
                seed_path = round_best
                break
        
        if seed_path:
            logger.info(f"Seeding Model {midx} from {seed_path}")
            ckpt = torch.load(seed_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            logger.warning(f"No seed found for Model {midx}")
            
        models[midx] = model
        optimizers[midx] = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        schedulers[midx] = optim.lr_scheduler.ReduceLROnPlateau(optimizers[midx], mode='min', factor=0.5, patience=3)
        best_l2s[midx] = float('inf')
        
        wandb.init(project="encoder_neurIPS_simultaneous", name=f"Model_{midx:02d}_SIM", config={
            "model_idx": midx,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "AdamW"
        }, reinit=True)

    for epoch in range(NUM_EPOCHS):
        logger.info(f"--- Epoch {epoch} ---")
        epoch_stats = {}
        
        for midx in CANDIDATES:
            train_l2 = train_one_epoch(models[midx], train_loader, optimizers[midx], device)
            val_l2, val_rmse = validate(models[midx], val_loader, device)
            
            schedulers[midx].step(val_l2)
            curr_lr = optimizers[midx].param_groups[0]['lr']
            
            epoch_stats[midx] = (val_l2, val_rmse, curr_lr)
            
            if val_l2 < best_l2s[midx]:
                best_l2s[midx] = val_l2
                save_path = os.path.join(SIM_SAVE_DIR, f"model_{midx:02d}_best.pt")
                torch.save({'model_state_dict': models[midx].state_dict(), 'val_l2': val_l2, 'val_rmse': val_rmse}, save_path)
            
            # Individual wandb logging
            # Note: Since we have multiple wandb runs in one process, we need to be careful.
            # However, the user wants a side-by-side report in console.
            
        # Side-by-side Report
        logger.info(f"Comparison Epoch {epoch}:")
        logger.info(f"{'Metric':<15} | {'Model 4 (4th)':<20} | {'Model 28 (5th)':<20}")
        logger.info("-" * 65)
        
        m1, m2 = CANDIDATES
        l1, r1, lr1 = epoch_stats[m1]
        l2, r2, lr2 = epoch_stats[m2]
        
        # Color coding: Green for the winner of the pair this epoch
        c1_l2 = "\033[92m" if l1 < l2 else ""
        c2_l2 = "\033[92m" if l2 < l1 else ""
        reset = "\033[0m"
        
        logger.info(f"{'Val L2':<15} | {c1_l2}{l1:.6e}{reset:<20} | {c2_l2}{l2:.6e}{reset:<20}")
        logger.info(f"{'Val RMSE':<15} | {r1:<20.6e} | {r2:<20.6e}")
        logger.info(f"{'LR':<15} | {lr1:<20.2e} | {lr2:<20.2e}")

if __name__ == "__main__":
    main()
