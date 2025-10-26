#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train script for WAE model 01.
This script trains the WAE model using either Apple Silicon or NVIDIA GPU if available.
Supports loading from a saved model checkpoint with the --resume_checkpoint argument.
"""

import os
import sys
import logging
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse  # <---- NEW
import threading
import queue

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import our custom modules
from encoder.model_WAE_01 import WAE
from EfficientDataLoader import EfficientDataLoader

# Figure out the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preferences_path = os.path.join(project_root, "experiment.preferences")
from Ordered_001_Initialize import HostPreferences

# When creating preferences, pass the resolved path:
preferences = HostPreferences(filename=preferences_path)

# Training configuration constants
BATCH_SIZE = 128
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-5
NUM_WORKERS = 10
SAVE_INTERVAL = 10
BATCHES_PER_EPOCH = 10
CACHE_SIZE = 50
# Prefetch queue depth: 0 disables prefetch; N>0 enables background prefetch with that many buffered batches
PREFETCH__QUEUE = 100
MODEL_NAME = "WAE_Prefetch_001"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set logging level from preferences
if hasattr(preferences, 'logging_level'):
    level = getattr(logging, preferences.logging_level.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        logger.info(f"Set logging level to {preferences.logging_level.upper()}")

# === Accelerator detection & colorful report ===
CSI = "\033["
RESET = f"{CSI}0m"
COLORS = [31, 33, 32, 36, 34, 35]  # R, Y, G, C, B, M

def rainbow(msg: str) -> str:
    out = []
    k = 0
    for ch in msg:
        if ch.strip():
            out.append(f"{CSI}{COLORS[k % len(COLORS)]}m{ch}{RESET}")
            k += 1
        else:
            out.append(ch)
    return ''.join(out)


def accelerator_report():
    """Detect CUDA/MPS/CPU and print a colorful diagnostic. Returns (device, info_dict)."""
    pyv = torch.__version__
    try:
        import platform
        py = platform.python_version()
    except Exception:
        py = "?"

    has_cuda = torch.cuda.is_available()
    cuda_version = getattr(torch.version, 'cuda', None)
    mps_built = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    mps_avail = mps_built and torch.backends.mps.is_available()

    # Select device preference: CUDA > MPS > CPU
    device = torch.device('cuda') if has_cuda else (torch.device('mps') if mps_avail else torch.device('cpu'))

    lines = [
        f"Python: {py} | PyTorch: {pyv}",
        f"CUDA available: {has_cuda} | CUDA toolkit: {cuda_version}",
        f"MPS built: {mps_built} | MPS available (runtime): {mps_avail}",
        f"Selected default device: {device}",
    ]

    if has_cuda:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            cap = torch.cuda.get_device_capability(idx)
            lines += [
                f"CUDA device count: {torch.cuda.device_count()}",
                f"Current device: {idx} | Name: {props.name}",
                f"Compute capability: {cap[0]}.{cap[1]} | Total memory: {props.total_memory/1024**3:.2f} GiB",
            ]
        except Exception as e:
            lines.append(f"CUDA detail query failed: {e}")

    # Print colorful lines and also log them
    for ln in lines:
        print(rainbow(ln))
        logger.info(ln)

    return device, {
        'python': py,
        'torch': pyv,
        'cuda_available': has_cuda,
        'cuda_version': cuda_version,
        'mps_built': mps_built,
        'mps_available': mps_avail,
        'selected_device': str(device),
    }


def setup_device():
    """Set up the computing device (GPU/CPU) for training with a colorful report."""
    device, _info = accelerator_report()
    return device

import wandb

def main():
    # Parse command-line arguments for model checkpoint resuming (<---- NEW)
    parser = argparse.ArgumentParser(description="Train WAE model 01 (with checkpoint resume support)")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Path to checkpoint (.pt file) to resume training from (optional).")
    # Debug and performance flags
    parser.add_argument('--debug_profile', action='store_true', help='Enable detailed timing logs')
    parser.add_argument('--fast_debug', action='store_true', help='Use very small epoch/batch counts for quick runs')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--num_workers', type=int, default=None, help='Override NUM_WORKERS for data loading')
    args = parser.parse_args()

    # Derive effective settings (do not mutate module constants)
    effective_batches_per_epoch = BATCHES_PER_EPOCH
    effective_num_epochs = NUM_EPOCHS
    effective_num_workers = NUM_WORKERS if args.num_workers is None else args.num_workers
    effective_batch_size = BATCH_SIZE

    if args.fast_debug:
        effective_batches_per_epoch = 2
        effective_num_epochs = 1
        effective_batch_size = min(128, BATCH_SIZE)
        logger.info("FAST DEBUG active: epochs=%s, batches/epoch=%s, batch_size=%s", effective_num_epochs, effective_batches_per_epoch, effective_batch_size)

    if args.num_workers is not None:
        logger.info("Overriding num_workers: %s -> %s", NUM_WORKERS, effective_num_workers)

    # Read model file for notes (adjust path as needed)
    model_filepath = os.path.join(parent_dir, 'encoder', 'model_WAE_01.py')
    notes_content = ""
    try:
        with open(model_filepath, 'r') as file:
            notes_content = file.read()
    except Exception as e:
        notes_content = f"Failed to load model file: {e}"

    # Start wandb run, use MODEL_NAME as project, add notes
    wandb_run = None
    if not args.no_wandb:
        # On Windows, disable symlinks to avoid WinError 1314 during wandb.save
        try:
            import platform
            is_windows = platform.system().lower().startswith('win')
        except Exception:
            is_windows = False
        wandb_run = wandb.init(
            project=MODEL_NAME,
            name=f"{MODEL_NAME}_run",
            config={
                "batch_size": effective_batch_size,
                "epochs": effective_num_epochs,
                "learning_rate": LEARNING_RATE,
                "model": MODEL_NAME,
            },
            notes=notes_content,
            settings=wandb.Settings(symlink=not is_windows)
        )

    # Setup device (GPU/CPU) with colorful report and log into W&B
    device, acc_info = accelerator_report()
    try:
        if not args.no_wandb:
            wandb.config.update({'accelerator': acc_info}, allow_val_change=True)
    except Exception:
        pass
    
    # Create save directories in current directory
    save_dir = './saved_models'
    log_dir = './logs'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize model and optimizer
    model = WAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_epoch = 0  # Default start at first epoch
    
    # === Load checkpoint if provided ===  (<---- NEW)
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint
    else:
        # Convenience: If wae_final.pt exists and not specified, can default to that (optional)
        default_final = os.path.join(save_dir, f"wae_final.pt")
        resume_path = default_final if os.path.isfile(default_final) else None

    if resume_path and os.path.isfile(resume_path):
        try:
            logger.info(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']  # Resume from next epoch
                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                logger.info("Checkpoint does not contain epoch info, starting from epoch 0")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error("Training will start from scratch.")
    else:
        if args.resume_checkpoint is not None:
            logger.warning(f"Checkpoint {args.resume_checkpoint} does not exist, will start from scratch.")

    logger.info(f"Initialized WAE model with latent dimension {model.fc4.out_features}")
    
    # Initialize data loader with timing
    logger.info(f"Starting EfficientDataLoader construction... root={preferences.training_data_path}")
    t0_loader = time.time()
    dataloader = EfficientDataLoader(
        root_directory=preferences.training_data_path,
        batch_size=effective_batch_size,
        num_workers=effective_num_workers,
        cache_size=CACHE_SIZE,
        shuffle=True
    )
    t1_loader = time.time()
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")
    logger.info(f"EfficientDataLoader ready in {t1_loader - t0_loader:.2f}s (workers={effective_num_workers}, cache_size={CACHE_SIZE})")
    
    # Training loop
    start_time = time.time()
    global_step = 0

    def _start_prefetcher(dl, batch_rows, depth):
        q = queue.Queue(maxsize=depth)
        stop_event = threading.Event()
        def producer():
            while not stop_event.is_set():
                try:
                    batch = dl.get_batch(NUMBER_OF_ROWS=batch_rows)
                    # Use timeout so we can react to stop_event
                    while not stop_event.is_set():
                        try:
                            q.put(batch, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                except Exception as e:
                    # propagate error by placing it in queue
                    try:
                        q.put({'__error__': str(e)}, timeout=0.1)
                    except Exception:
                        pass
                    break
        th = threading.Thread(target=producer, name="PrefetchProducer", daemon=True)
        th.start()
        return q, stop_event, th
    
    for epoch in range(start_epoch, effective_num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_mmd_loss = 0
        epoch_triplet_loss = 0
        num_batches = 0
        
        epoch_start_time = time.time()

        prefetch_enabled = PREFETCH__QUEUE and PREFETCH__QUEUE > 0
        if prefetch_enabled:
            if args.debug_profile:
                logger.info(f"Starting prefetcher with depth={PREFETCH__QUEUE}")
            pf_queue, pf_stop, pf_thread = _start_prefetcher(dataloader, effective_batch_size, PREFETCH__QUEUE)
        else:
            pf_queue = pf_stop = pf_thread = None
        
        for _ in range(effective_batches_per_epoch):
            # Get batch from dataloader (or prefetch queue)
            t_batch0 = time.time() if args.debug_profile else None
            if prefetch_enabled:
                while True:
                    try:
                        batch = pf_queue.get(timeout=0.1)
                        pf_queue.task_done()
                        break
                    except queue.Empty:
                        if args.debug_profile:
                            # waiting on prefetch
                            pass
                        continue
                if '__error__' in batch:
                    raise RuntimeError(f"Prefetch error: {batch['__error__']}")
            else:
                batch = dataloader.get_batch(NUMBER_OF_ROWS=effective_batch_size)
            t_batch1 = time.time() if args.debug_profile else None
            velocity_data = batch['velocity_data']
            
            # Convert numpy array to tensor and move to device (optimize copy path)
            t_conv0 = time.time() if args.debug_profile else None
            x_cpu = torch.from_numpy(velocity_data).float()
            if device.type == 'cuda':
                x_cpu = x_cpu.pin_memory()
                x = x_cpu.to(device, non_blocking=True)
            else:
                x = x_cpu.to(device)
            t_conv1 = time.time() if args.debug_profile else None
            
            # Forward pass
            if args.debug_profile and device.type == 'cuda':
                torch.cuda.synchronize()
            t_fwd0 = time.time() if args.debug_profile else None
            recon_x, z = model(x)
            if args.debug_profile and device.type == 'cuda':
                torch.cuda.synchronize()
            t_fwd1 = time.time() if args.debug_profile else None
            
            # Calculate loss
            t_loss0 = time.time() if args.debug_profile else None
            loss, recon_loss, mmd_loss, triplet_loss = model.loss_function(recon_x, x, z)
            if args.debug_profile and device.type == 'cuda':
                torch.cuda.synchronize()
            t_loss1 = time.time() if args.debug_profile else None
            
            # Backward pass and optimize
            t_bwd0 = time.time() if args.debug_profile else None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.debug_profile and device.type == 'cuda':
                torch.cuda.synchronize()
            t_bwd1 = time.time() if args.debug_profile else None
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_mmd_loss += mmd_loss.item()
            epoch_triplet_loss += triplet_loss.item()
            num_batches += 1
            global_step += 1
            
            # Optional per-batch debug timings
            if args.debug_profile and (num_batches <= 5 or num_batches % 20 == 0):
                logger.debug(
                    "Batch %d timings (s): get_batch=%.3f, to_tensor/H2D=%.3f, fwd=%.3f, loss=%.3f, bwd+step=%.3f%s",
                    num_batches,
                    (t_batch1 - t_batch0) if t_batch0 is not None else -1,
                    (t_conv1 - t_conv0) if t_conv0 is not None else -1,
                    (t_fwd1 - t_fwd0) if t_fwd0 is not None else -1,
                    (t_loss1 - t_loss0) if t_loss0 is not None else -1,
                    (t_bwd1 - t_bwd0) if t_bwd0 is not None else -1,
                    " (prefetched)" if prefetch_enabled else "",
                )
            
            # Log to tensorboard
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Loss/recon_step', recon_loss.item(), global_step)
            writer.add_scalar('Loss/mmd_step', mmd_loss.item(), global_step)
            writer.add_scalar('Loss/triplet_step', triplet_loss.item(), global_step)
        
        # Stop prefetcher for this epoch
        if prefetch_enabled:
            pf_stop.set()
            # Drain queue quickly to unblock producer if needed
            try:
                while True:
                    pf_queue.get_nowait()
                    pf_queue.task_done()
            except queue.Empty:
                pass
            pf_thread.join(timeout=5.0)

            # Log metrics to wandb (if enabled)
            if not args.no_wandb:
                wandb.log({
                    "Loss/train_step": loss.item(),
                    "Loss/recon_step": recon_loss.item(),
                    "Loss/mmd_step": mmd_loss.item(),
                    "Loss/triplet_step": triplet_loss.item(),
                    "global_step": global_step
                })
        
        # Compute epoch average losses
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_mmd_loss = epoch_mmd_loss / num_batches
        avg_triplet_loss = epoch_triplet_loss / num_batches
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{effective_num_epochs}, " 
                   f"Loss: {avg_loss:.6f}, "
                   f"Recon: {avg_recon_loss:.6f}, "
                   f"MMD: {avg_mmd_loss:.6f}, "
                   f"Triplet: {avg_triplet_loss:.6f}, "
                   f"Time: {epoch_time:.2f}s")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
        writer.add_scalar('Loss/recon_epoch', avg_recon_loss, epoch)
        writer.add_scalar('Loss/mmd_epoch', avg_mmd_loss, epoch)
        writer.add_scalar('Loss/triplet_epoch', avg_triplet_loss, epoch)

        # Epoch summary to wandb
        if not args.no_wandb:
            wandb.log({
                "Loss/train_epoch": avg_loss,
                "Loss/recon_epoch": avg_recon_loss,
                "Loss/mmd_epoch": avg_mmd_loss,
                "Loss/triplet_epoch": avg_triplet_loss,
                "epoch": epoch
            })
        
        # Save model checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{epoch+1}.pt")
            #checkpoint = torch.load(checkpoint_path, map_location='cpu')
            #This code is broken I believe...
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
            if not args.no_wandb:
                try:
                    wandb.save(checkpoint_path)
                except OSError as e:
                    logger.warning(f"wandb.save failed (likely symlink issue on Windows): {e}")
                    try:
                        import shutil
                        run_dir = wandb.run.dir if wandb.run else None
                        if run_dir:
                            dst_dir = os.path.join(run_dir, 'files', 'saved_models')
                            os.makedirs(dst_dir, exist_ok=True)
                            dst_path = os.path.join(dst_dir, os.path.basename(checkpoint_path))
                            shutil.copy2(checkpoint_path, dst_path)
                            logger.info(f"Copied checkpoint to W&B files dir: {dst_path}")
                    except Exception as e2:
                        logger.error(f"Failed to copy checkpoint into W&B dir: {e2}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{MODEL_NAME}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    if not args.no_wandb:
        try:
            wandb.save(final_model_path)
        except OSError as e:
            logger.warning(f"wandb.save (final) failed (likely symlink issue on Windows): {e}")
            try:
                import shutil
                run_dir = wandb.run.dir if wandb.run else None
                if run_dir:
                    dst_dir = os.path.join(run_dir, 'files', 'saved_models')
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, os.path.basename(final_model_path))
                    shutil.copy2(final_model_path, dst_path)
                    logger.info(f"Copied final model to W&B files dir: {dst_path}")
            except Exception as e2:
                logger.error(f"Failed to copy final model into W&B dir: {e2}")
    
    # Training summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Final loss: {avg_loss:.6f}")
    
    writer.close()
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()