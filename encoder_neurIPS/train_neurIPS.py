import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
from encoder_neurIPS.models import create_model_variant, ORIGINAL_DIM, LATENT_DIM

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/Users/kkreth/PycharmProjects/data/encoder_neurIPS"
OG_MODEL_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best.pt"
NUM_MODELS = 32
RUN_TIME_PER_MODEL = 5 * 60 # 5 minutes
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_models")
STATE_FILE = os.path.join(SAVE_DIR, "training_state.pkl")

def get_device():
    if torch.cuda.is_available(): 
        device = torch.device('cuda')
        print("\033[91mN\033[93mV\033[92mI\033[96mD\033[94mI\033[95mA\033[0m detected!")
        return device
    if torch.backends.mps.is_available(): 
        device = torch.device('mps')
        print("\033[91mM\033[93mP\033[92mS\033[0m \033[96md\033[94me\033[95mt\033[91me\033[93mc\033[92mt\033[96me\033[94md\033[95m!")
        return device
    device = torch.device('cpu')
    print("\033[91mC\033[93mP\033[92mU\033[0m \033[96md\033[94me\033[95mt\033[91me\033[93mc\033[92mt\033[96me\033[94md\033[95m!")
    return device

def load_data():
    train_path = os.path.join(DATA_DIR, 'training_auto_encoder.pkl')
    val_path = os.path.join(DATA_DIR, 'validation_auto_encoder.pkl')
    with open(train_path, 'rb') as f:
        train_np = pickle.load(f).astype(np.float32)
    with open(val_path, 'rb') as f:
        val_np = pickle.load(f).astype(np.float32)
    return train_np, val_np

def train_one_model(model_idx, device, train_loader, val_loader, round_num, og_performance=None, dry_run=False):
    """
    Trains a single model variant for a fixed amount of time.
    
    Args:
        model_idx: Integer index (0-31) representing the model architecture variant.
        device: Torch device (CPU, CUDA, or MPS).
        train_loader: DataLoader for training samples.
        val_loader: DataLoader for validation samples.
        round_num: Current round in the survival strategy (1, 2, etc.)
        og_performance: Reference RMSE from the 'OG' model for comparison.
        dry_run: If True, runs for a very short duration for testing.
        
    Returns:
        best_val_l2: The best True L2 Norm validation loss achieved during the run.
    """
    model = create_model_variant(model_idx).to(device)
    model_name = f"NeurIPS_Model_{model_idx:02d}_R{round_num}"
    
    # Path for weights from PREVIOUS round
    # Prioritize 'best' weight, but allow 'last' if needed
    prev_round_path = os.path.join(SAVE_DIR, f"round_{round_num-1}", f"model_{model_idx:02d}_best.pt")
    if round_num > 1 and not os.path.exists(prev_round_path):
        last_path = os.path.join(SAVE_DIR, f"round_{round_num-1}", f"model_{model_idx:02d}_last.pt")
        if os.path.exists(last_path):
            prev_round_path = last_path
    
    seeded_param_count = 0
    total_param_count = sum(p.numel() for p in model.parameters())

    # Check if we have weights from a previous round of THIS training session
    checkpoint_info = "Random Initialization"
    if round_num > 1:
        if os.path.exists(prev_round_path):
            try:
                ckpt = torch.load(prev_round_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                
                # Get file metadata
                mtime = os.path.getmtime(prev_round_path)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                file_size = os.path.getsize(prev_round_path) / (1024 * 1024)
                
                checkpoint_info = f"Loaded Round {round_num-1} Best: {os.path.basename(prev_round_path)} (Size: {file_size:.2f}MB, Saved: {timestamp})"
                print(f"[{model_name}] {checkpoint_info}")
                seeded_param_count = total_param_count 
            except Exception as e:
                print(f"[{model_name}] ERROR: Failed to load weights from Round {round_num-1}: {e}")
                checkpoint_info = f"Error loading Round {round_num-1}: {str(e)}"
        else:
            print(f"[{model_name}] CRITICAL WARNING: Round {round_num-1} weights missing at {prev_round_path}!")
            print(f"[{model_name}] Per instruction, NOT falling back to OG Model. Starting from Random Weights.")
            checkpoint_info = f"Missing Round {round_num-1} checkpoint (Started Random)"
            
    # ONLY if we didn't load from a previous round and it's Round 1, try OG seeding
    elif round_num == 1 and os.path.exists(OG_MODEL_PATH):
        try:
            og_ckpt = torch.load(OG_MODEL_PATH, map_location=device, weights_only=False)
            state_dict = og_ckpt.get('model_state_dict', og_ckpt)
            
            # Map OG state dict to new model with shape checking
            model_dict = model.state_dict()
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                        seeded_param_count += v.numel()
            
            model.load_state_dict(filtered_dict, strict=False)
            mtime = os.path.getmtime(OG_MODEL_PATH)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            checkpoint_info = f"OG Model Seeded (Matched {len(filtered_dict)} layers, {seeded_param_count/total_param_count:.1%} of params, Saved: {timestamp})"
            print(f"[{model_name}] {checkpoint_info}")
        except Exception as e:
            print(f"[{model_name}] Could not seed weights from OG Model: {e}")
            checkpoint_info = f"Error seeding OG: {str(e)}"

    seeded_pct = seeded_param_count / total_param_count

    # Configuration for optimizer
    # More aggressive learning rate for early rounds, decaying as pool shrinks
    # Round 1 & 2: 5e-4, Round 3: 2.5e-4, Round 4+: 1e-4
    base_lr = 5e-4 if round_num <= 2 else (2.5e-4 if round_num == 3 else 1e-4)
    
    # USE AdamW exclusively as requested
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5)

    wandb.init(project=f"encoder_neurIPS_v3_R{round_num}", name=model_name, config={
        "model_idx": model_idx, 
        "round": round_num,
        "base_lr": base_lr,
        "optimizer": "AdamW",
        "h1": getattr(model, 'h1', 'unknown'),
        "h2": getattr(model, 'h2', 'unknown'),
        "h3": getattr(model, 'h3', 'unknown'),
        "seeded_pct": seeded_pct,
        "checkpoint_source": checkpoint_info
    }, reinit=True)
    
    # Define metrics for scientific notation and axis grouping in UI
    wandb.define_metric("val_rmse", summary="min")
    wandb.define_metric("val_rmse_baseline", summary="mean")
    wandb.define_metric("val_l2_norm", summary="min")
    
    start_time = time.time()
    best_val_l2_norm = float('inf')
    best_val_rmse = float('inf')
    epoch = 0
    
    # Training Loop
    # Set training duration: 7 minutes for Round 2, 10 minutes for all subsequent rounds
    if round_num == 1:
        limit_seconds = 5 * 60
    elif round_num == 2:
        limit_seconds = 7 * 60
    else:
        limit_seconds = 10 * 60

    while time.time() - start_time < (10 if dry_run else limit_seconds):
        model.train()
        train_loss_total = 0.0
        train_loss_l2_norm = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Check time limit within batch loop
            if time.time() - start_time > (10 if dry_run else limit_seconds): break
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, z = model(x)
            
            # Reconstruction loss is computed as True L2 Norm (Euclidean distance) per sample
            # True L2 Norm = sqrt(sum((pred - target)^2))
            diff = recon_x - x.view_as(recon_x)
            l2_norm_per_sample = torch.norm(diff, p=2, dim=1)
            recon_loss = torch.mean(l2_norm_per_sample)
            
            # Latent regularization (original z^2 logic)
            l2_reg = torch.mean(z ** 2)
            loss = recon_loss + 0.00005 * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            train_loss_l2_norm += recon_loss.item()
            batch_count += 1
            if dry_run and batch_count > 5: break 
        
        if batch_count == 0: break

        # Validation Step
        model.eval()
        total_l2_norm = 0.0
        total_mse = 0.0
        sample_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, z = model(x)
                
                diff = recon_x - x.view_as(recon_x)
                # True L2 Norm per sample
                l2_norms = torch.norm(diff, p=2, dim=1)
                total_l2_norm += torch.sum(l2_norms).item()
                
                # Mean Squared Error for RMSE calculation
                total_mse += torch.sum(diff**2).item()
                sample_count += x.size(0)
        
        val_l2_norm = total_l2_norm / sample_count
        val_rmse = np.sqrt(total_mse / (sample_count * ORIGINAL_DIM))
        
        # Save if this is the best so far
        if val_l2_norm < best_val_l2_norm:
            best_val_l2_norm = val_l2_norm
            best_val_rmse = val_rmse
            # Save the "best" snapshot
            round_dir = os.path.join(SAVE_DIR, f"round_{round_num}")
            os.makedirs(round_dir, exist_ok=True)
            save_path = os.path.join(round_dir, f"model_{model_idx:02d}_best.pt")
            torch.save({
                'model_idx': model_idx,
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'best_val_l2_norm': best_val_l2_norm,
                'best_val_rmse': best_val_rmse,
                'epoch': epoch
            }, save_path)
            # print(f"[{model_name}] New best Val L2 Norm: {best_val_l2_norm:.4e} (Epoch {epoch})")

        # Logging to WandB with scientific notation formatting
        log_dict = {
            "epoch": epoch, 
            "val_l2_norm_scientific": f"{val_l2_norm:.4e}",
            "val_rmse_scientific": f"{val_rmse:.4e}", 
            "best_val_l2_norm_scientific": f"{best_val_l2_norm:.4e}",
            "best_val_rmse_scientific": f"{best_val_rmse:.4e}",
            "train_loss_total_scientific": f"{(train_loss_total / batch_count):.4e}",
            "train_loss_l2_norm_scientific": f"{(train_loss_l2_norm / batch_count):.4e}",
            # Standard keys
            "val_l2_norm": val_l2_norm,
            "val_rmse": val_rmse,
            "best_val_l2_norm": best_val_l2_norm,
            "best_val_rmse": best_val_rmse,
            "train_loss_total": train_loss_total / batch_count,
            "train_loss_l2_norm": train_loss_l2_norm / batch_count
        }
        if og_performance:
            log_dict["val_rmse_baseline"] = og_performance # Combined on same graph as val_rmse
            log_dict["val_rmse_baseline_scientific"] = f"{og_performance:.4e}"
        wandb.log(log_dict)
        epoch += 1

    # Final save (the "last" state)
    round_dir = os.path.join(SAVE_DIR, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)
    last_path = os.path.join(round_dir, f"model_{model_idx:02d}_last.pt")
    torch.save({
        'model_idx': model_idx,
        'round_num': round_num,
        'model_state_dict': model.state_dict(),
        'val_l2_norm': val_l2_norm,
        'val_rmse': val_rmse,
        'epoch': epoch
    }, last_path)
    
    wandb.finish()
    return best_val_l2_norm

def main(dry_run=False, force_restart=False):
    """
    Main execution logic for the survival-based training strategy.
    
    Survival Strategy:
    1. Start with 32 models (or num_models_to_train).
    2. In each round:
       a. Train each model for RUN_TIME_PER_MODEL (5 mins).
       b. Record best validation L2 score.
       c. Take the top 50% performers to the next round.
    3. Repeat until only 1 model remains.
    
    Restart Mechanism:
    - Current round and active models are saved in STATE_FILE.
    - If interrupted, the script resumes from the current round's first untrained model.
    - Checkpoints are stored in SAVE_DIR/round_X/model_Y.pt.
    """
    device = get_device()
    train_np, val_np = load_data()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=4096, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_np)), batch_size=4096, shuffle=False)
    
    # Get OG performance as benchmark (RMSE)
    og_performance_rmse = 0.0007695 
    
    # Initialization
    num_models_to_train = 2 if dry_run else NUM_MODELS
    
    state = None
    if not force_restart and os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                state = pickle.load(f)
            print(f"--- Resuming from saved state (Round {state['round_num']}) ---")
        except Exception as e:
            print(f"--- Could not load state file: {e}. Starting from scratch. ---")

    if state:
        current_models = state['current_models']
        round_num = state['round_num']
        # 'results' contains models already trained in the CURRENT round
        round_results = state.get('round_results', [])
    else:
        current_models = list(range(num_models_to_train))
        round_num = 1
        round_results = []

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    while len(current_models) > 1:
        print(f"--- Round {round_num}: {len(current_models)} models remaining ---")
        
        # Identify which models in this round haven't been trained yet
        trained_indices = [r[0] for r in round_results]
        remaining_in_round = [m for m in current_models if m not in trained_indices]
        
        for midx in remaining_in_round:
            score = train_one_model(midx, device, train_loader, val_loader, round_num, og_performance_rmse, dry_run=dry_run)
            round_results.append((midx, score))
            
            # Save state after EACH model to ensure we can restart precisely
            with open(STATE_FILE, 'wb') as f:
                pickle.dump({
                    'round_num': round_num,
                    'current_models': current_models,
                    'round_results': round_results
                }, f)
        
        # All models in current round finished, calculate survivors
        # Sort by score (best is lowest L2 Norm)
        round_results.sort(key=lambda x: x[1])
        # Keep top 50%
        survivors = [r[0] for r in round_results[:max(1, len(current_models)//2)]]
        print(f"Round {round_num} complete. Survivors: {survivors}")
        
        # Update for next round
        current_models = survivors
        round_num += 1
        round_results = []
        
        # Save state for the beginning of the next round
        with open(STATE_FILE, 'wb') as f:
            pickle.dump({
                'round_num': round_num,
                'current_models': current_models,
                'round_results': round_results
            }, f)

    if len(current_models) == 1:
        print(f"Final Winner: NeurIPS_Model_{current_models[0]:02d}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run a quick test with 2 models and short duration")
    parser.add_argument("--force-restart", action="store_true", help="Ignore saved state and start from Round 1")
    args = parser.parse_args()
    main(dry_run=args.dry_run, force_restart=args.force_restart)
