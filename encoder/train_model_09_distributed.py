#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed training script for Model_09_Residual_AE using PyTorch DDP

Usage on Slurm:
    srun python encoder/train_model_09_distributed.py --epochs 500
"""

import os
import sys
import time
import logging
import pickle
import argparse
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from encoder.permutations.model_09_residual_ae import ResidualAE
from Ordered_001_Initialize import HostPreferences

# Import wandb (only on rank 0)
import wandb

# Configuration
MODEL_NAME = "Model_09_Residual_AE_DDP"
DEFAULT_BATCH_SIZE = 128  # Per GPU
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 500

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Rank %(rank)s] - %(levelname)s - %(message)s'
)


def setup_distributed():
    """Initialize distributed training from Slurm environment"""
    # Get Slurm environment variables
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))

    # Set up process group
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',  # Use NCCL for GPU
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_logger(rank):
    """Get logger with rank info"""
    logger = logging.getLogger(__name__)
    # Add rank to all log records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    return logger


def load_cached_array(file_path, sample_percentage=100):
    """Load pickled NumPy array with optional sampling"""
    t0 = time.time()
    with open(file_path, 'rb') as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"File {file_path} does not contain a NumPy array")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    if sample_percentage < 100:
        target_size = int(len(arr) * sample_percentage / 100.0)
        arr = arr[:target_size]

    t1 = time.time()
    return arr


def make_distributed_loader(arr, batch_size, shuffle, rank, world_size):
    """Create distributed DataLoader"""
    x_tensor = torch.from_numpy(arr)
    dataset = TensorDataset(x_tensor)

    # Distributed sampler ensures each GPU gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,  # Parallel data loading
        drop_last=False
    )

    return loader, sampler


def prune_old_checkpoints(save_dir, model_name, keep_last=5):
    """Keep only the last N checkpoints"""
    try:
        pattern = os.path.join(save_dir, f"{model_name}_epoch_*.pt")
        checkpoints = glob.glob(pattern)

        checkpoint_epochs = []
        for ckpt in checkpoints:
            match = re.search(r'epoch_(\d+)\.pt$', ckpt)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_epochs.append((epoch_num, ckpt))

        checkpoint_epochs.sort(key=lambda x: x[0])

        if len(checkpoint_epochs) > keep_last:
            to_delete = checkpoint_epochs[:-keep_last]
            for epoch_num, ckpt_path in to_delete:
                try:
                    os.remove(ckpt_path)
                except Exception:
                    pass
    except Exception:
        pass


def train_model(model, train_loader, val_loader, device, optimizer, train_sampler,
                start_epoch=0, num_epochs=500, save_dir=None, use_wandb=True,
                keep_checkpoints=5, rank=0, world_size=1):
    """Distributed training loop"""
    logger = get_logger(rank)

    best_val_rmse = float('inf')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        train_sampler.set_epoch(epoch)

        # Training
        model.train()
        train_loss = 0.0
        train_sse = 0.0
        train_elements = 0

        for batch in train_loader:
            x_cpu = batch[0]
            x = x_cpu.to(device, non_blocking=True)

            optimizer.zero_grad()

            recon_x, z = model(x)
            loss, recon_loss, l2_reg, _ = model.module.loss_function(recon_x, x, z)

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

            with torch.no_grad():
                sse = torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                train_sse += sse
                train_elements += x.numel()

        # Gather metrics across all GPUs
        train_sse_tensor = torch.tensor(train_sse, device=device)
        train_elements_tensor = torch.tensor(train_elements, device=device)

        if world_size > 1:
            dist.all_reduce(train_sse_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_elements_tensor, op=dist.ReduceOp.SUM)

        train_rmse = np.sqrt(train_sse_tensor.item() / train_elements_tensor.item())

        # Validation (only on rank 0 to save time)
        if rank == 0:
            model.eval()
            val_sse = 0.0
            val_elements = 0

            with torch.no_grad():
                for batch in val_loader:
                    x_cpu = batch[0]
                    x = x_cpu.to(device, non_blocking=True)

                    recon_x, z = model(x)

                    sse = torch.sum((recon_x - x.view_as(recon_x)) ** 2).item()
                    val_sse += sse
                    val_elements += x.numel()

            val_rmse = np.sqrt(val_sse / val_elements)
        else:
            val_rmse = 0.0

        # Broadcast validation RMSE to all ranks
        if world_size > 1:
            val_rmse_tensor = torch.tensor(val_rmse, device=device)
            dist.broadcast(val_rmse_tensor, src=0)
            val_rmse = val_rmse_tensor.item()

        epoch_time = time.time() - epoch_start_time
        epochs_remaining = num_epochs - (epoch + 1)
        eta_minutes = (epoch_time * epochs_remaining) / 60.0

        # Logging and saving only on rank 0
        if rank == 0:
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'epoch_time_seconds': epoch_time,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'gpus_used': world_size
                })

            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"train_RMSE={train_rmse:.6f} val_RMSE={val_rmse:.6f} "
                           f"epoch_time={epoch_time:.1f}s [ETA: {eta_minutes:.1f}min] "
                           f"[{world_size} GPUs]")

            # Save checkpoint
            if save_dir:
                checkpoint_path = os.path.join(save_dir, f"{MODEL_NAME}_epoch_{epoch+1}.pt")
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.module.state_dict(),  # .module for DDP
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_rmse': train_rmse,
                        'val_rmse': val_rmse,
                        'best_val_rmse': best_val_rmse,
                    }, checkpoint_path)

                    if (epoch + 1) % 5 == 0:
                        logger.info(f"  → Saved checkpoint: {os.path.basename(checkpoint_path)}")

                    if use_wandb and (epoch + 1) % 10 == 0:
                        wandb.save(checkpoint_path)

                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")

            # Save best model
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                if save_dir:
                    best_path = os.path.join(save_dir, f"{MODEL_NAME}_best.pt")
                    try:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_rmse': train_rmse,
                            'val_rmse': val_rmse,
                            'best_val_rmse': best_val_rmse,
                        }, best_path)
                        logger.info(f"  → New best model! Val RMSE: {val_rmse:.6f}")
                        if use_wandb:
                            wandb.save(best_path)
                    except Exception:
                        pass

            # Prune old checkpoints
            if save_dir and (epoch + 1) % 5 == 0:
                prune_old_checkpoints(save_dir, MODEL_NAME, keep_last=keep_checkpoints)

    return {'best_val_rmse': best_val_rmse}


def main():
    parser = argparse.ArgumentParser(description="Distributed training for Model_09")
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size PER GPU')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--data_percentage', type=int, default=100)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='fluid-dynamics-ae')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--keep_checkpoints', type=int, default=5)
    parser.add_argument('--no_resume', action='store_true')

    args = parser.parse_args()

    # Set up distributed training
    rank, world_size, device = setup_distributed()
    logger = get_logger(rank)

    if rank == 0:
        logger.info(f"Distributed training on {world_size} GPUs")
        logger.info(f"Effective batch size: {args.batch_size} × {world_size} = {args.batch_size * world_size}")

    # Set default checkpoint
    if args.resume_checkpoint is None and not args.no_resume:
        args.resume_checkpoint = os.path.join(
            PARENT_DIR,
            'encoder/permutations/checkpoints/Model_09_Residual_AE/Model_09_Residual_AE_epoch_100.pt'
        )

    # Load preferences
    preferences_path = os.path.join(PARENT_DIR, "experiment.preferences")
    prefs = HostPreferences(filename=preferences_path)
    root_dir = getattr(prefs, 'root_path', os.getcwd())

    train_path = os.path.join(root_dir, 'training_auto_encoder.pkl')
    val_path = os.path.join(root_dir, 'validation_auto_encoder.pkl')

    # Initialize wandb (only rank 0)
    use_wandb = not args.no_wandb and rank == 0
    if use_wandb:
        model_source_path = os.path.join(PARENT_DIR, 'encoder', 'permutations', 'model_09_residual_ae.py')
        try:
            with open(model_source_path, 'r') as f:
                model_source_code = f.read()
        except:
            model_source_code = "Could not load"

        wandb_name = args.wandb_name or f"{MODEL_NAME}_{world_size}gpus"

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            notes=model_source_code,
            config={
                'model': MODEL_NAME,
                'num_gpus': world_size,
                'batch_size_per_gpu': args.batch_size,
                'effective_batch_size': args.batch_size * world_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
            }
        )

    # Load data (all ranks load, but sampler divides it)
    if rank == 0:
        logger.info(f"Loading data: {args.data_percentage}% of dataset")

    train_np = load_cached_array(train_path, args.data_percentage)
    val_np = load_cached_array(val_path, args.data_percentage)

    train_loader, train_sampler = make_distributed_loader(
        train_np, args.batch_size, shuffle=True, rank=rank, world_size=world_size
    )
    val_loader, _ = make_distributed_loader(
        val_np, args.batch_size, shuffle=False, rank=rank, world_size=world_size
    )

    del train_np, val_np

    # Initialize model
    model = ResidualAE().to(device)

    # Load checkpoint (all ranks load)
    start_epoch = 0
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        if rank == 0:
            logger.info(f"Loading checkpoint: {args.resume_checkpoint}")

        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

        if rank == 0:
            logger.info(f"✓ Resumed from epoch {start_epoch}")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    # Scale learning rate with number of GPUs (linear scaling rule)
    scaled_lr = args.lr * world_size
    optimizer = optim.Adam(model.parameters() if world_size == 1 else model.module.parameters(),
                          lr=scaled_lr)

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Create save directory (only rank 0)
    save_dir = None
    if rank == 0:
        save_dir = os.path.join(PARENT_DIR, 'encoder', 'saved_models')
        os.makedirs(save_dir, exist_ok=True)

    # Train
    if rank == 0:
        logger.info("="*80)
        logger.info(f"Starting distributed training: epochs {start_epoch + 1} to {args.epochs}")
        logger.info("="*80)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        train_sampler=train_sampler,
        start_epoch=start_epoch,
        num_epochs=args.epochs,
        save_dir=save_dir,
        use_wandb=use_wandb,
        keep_checkpoints=args.keep_checkpoints,
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        logger.info("="*80)
        logger.info("Training complete!")
        logger.info(f"Best validation RMSE: {results['best_val_rmse']:.6f}")
        logger.info("="*80)

        if use_wandb:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
