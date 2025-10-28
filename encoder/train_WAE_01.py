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
import multiprocessing as mp
import numpy as np

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
    device, _info = accelerator_report()
    return device

# --- Multiprocessing prefetch worker (top-level for Windows spawn) ---
# Note: Keep it import-safe; only simple args; no closures.

def _mp_prefetch_worker(root_directory: str,
                        batch_rows: int,
                        num_workers: int,
                        cache_size: int,
                        shuffle: bool,
                        out_q: "mp.Queue",
                        stop_evt: "mp.Event",
                        worker_id: int,
                        worker_count: int,
                        enable_dl_profile: bool):
    try:
        # Local imports are fine inside subprocess
        from EfficientDataLoader import EfficientDataLoader as _EDL
        import numpy as _np  # noqa: F401
        import torch as _torch  # noqa: F401
        dl = _EDL(
            root_directory=root_directory,
            batch_size=batch_rows,
            num_workers=num_workers,
            cache_size=cache_size,
            shuffle=shuffle,
            enable_profiling=enable_dl_profile
        )
        # Shard file list across workers to reduce contention
        try:
            fm = getattr(dl, 'file_metadata', None)
            if fm and worker_count > 1:
                dl.file_metadata = fm[worker_id::worker_count]
        except Exception:
            pass
        batches_made = 0
        last_metrics_emit = time.time()
        while not stop_evt.is_set():
            try:
                batch = dl.get_batch(NUMBER_OF_ROWS=batch_rows)
                vel_np = batch['velocity_data']
                # Put numpy array; conversion to tensor happens in main proc
                put_ok = False
                while not stop_evt.is_set() and not put_ok:
                    try:
                        out_q.put(vel_np, timeout=0.1)
                        put_ok = True
                    except Exception:
                        # queue.Full or other transient issues
                        continue
                batches_made += 1
                # Periodically emit loader metrics if profiling enabled
                if enable_dl_profile and batches_made % 20 == 0:
                    try:
                        out_q.put({'__metrics__': {'worker_id': worker_id,
                                                   'batches_made': batches_made,
                                                   'timings': getattr(dl, 'profiling', {}).get('timings', {})}}, timeout=0.1)
                    except Exception:
                        pass
            except Exception as e:
                try:
                    out_q.put({'__error__': str(e), 'worker_id': worker_id}, timeout=0.1)
                except Exception:
                    pass
                break
    except Exception as e:
        try:
            out_q.put({'__error__': f'worker_init: {e}', 'worker_id': worker_id}, timeout=0.1)
        except Exception:
            pass

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
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--batches_per_epoch', type=int, default=None, help='Override number of batches per epoch')
    parser.add_argument('--cache_size', type=int, default=None, help='Override dataloader cache size')
    parser.add_argument('--prefetch_depth', type=int, default=None, help='Override host prefetch queue depth (0 disables)')
    parser.add_argument('--producers', type=int, default=None, help='Number of prefetch producer threads')
    parser.add_argument('--device_prefetch', action='store_true', help='Enable device-side prefetch (double-buffer H2D with CUDA stream)')
    parser.add_argument('--no_device_prefetch', action='store_true', help='Disable auto device-side prefetch (when CUDA)')
    parser.add_argument('--mp_prefetch', action='store_true', help='Use multiprocessing prefetch producers (bypass GIL)')
    parser.add_argument('--mp_producers', type=int, default=None, help='Number of multiprocessing prefetch producers')
    parser.add_argument('--dl_profile', action='store_true', help='Enable EfficientDataLoader stage-level profiling logs')
    args = parser.parse_args()

    # Derive effective settings (do not mutate module constants)
    effective_batches_per_epoch = args.batches_per_epoch if args.batches_per_epoch is not None else BATCHES_PER_EPOCH
    effective_num_epochs = NUM_EPOCHS
    effective_num_workers = NUM_WORKERS if args.num_workers is None else args.num_workers
    effective_batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    effective_cache_size = args.cache_size if args.cache_size is not None else CACHE_SIZE
    effective_prefetch_depth = args.prefetch_depth if args.prefetch_depth is not None else PREFETCH__QUEUE
    effective_producers = max(1, args.producers) if args.producers is not None else 1
    effective_mp_producers = max(1, args.mp_producers) if args.mp_producers is not None else 0

    if args.fast_debug:
        effective_batches_per_epoch = 2
        effective_num_epochs = 1
        effective_batch_size = min(128, effective_batch_size)
        logger.info("FAST DEBUG active: epochs=%s, batches/epoch=%s, batch_size=%s", effective_num_epochs, effective_batches_per_epoch, effective_batch_size)

    if args.num_workers is not None:
        logger.info("Overriding num_workers: %s -> %s", NUM_WORKERS, effective_num_workers)
    if args.batch_size is not None:
        logger.info("Overriding batch_size: %s -> %s", BATCH_SIZE, effective_batch_size)
    if args.batches_per_epoch is not None:
        logger.info("Overriding batches_per_epoch: %s -> %s", BATCHES_PER_EPOCH, effective_batches_per_epoch)
    if args.cache_size is not None:
        logger.info("Overriding cache_size: %s -> %s", CACHE_SIZE, effective_cache_size)
    if args.prefetch_depth is not None:
        logger.info("Overriding prefetch_depth: %s -> %s", PREFETCH__QUEUE, effective_prefetch_depth)
    if args.producers is not None:
        logger.info("Using %s prefetch producer thread(s)", effective_producers)

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
    # Enable cuDNN autotuner for fixed-size inputs to speed up convs (safe if shapes are stable)
    try:
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
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
        cache_size=effective_cache_size,
        shuffle=True,
        enable_profiling=args.dl_profile
    )
    t1_loader = time.time()
    logger.info(f"Found {len(dataloader.file_metadata)} valid files with velocity data")
    logger.info(f"EfficientDataLoader ready in {t1_loader - t0_loader:.2f}s (workers={effective_num_workers}, cache_size={effective_cache_size})")
    
    # Training loop
    start_time = time.time()
    global_step = 0
    latest_metrics = {}
    consecutive_timeouts = 0

    def _start_prefetch_farm(dl, batch_rows, depth, producers):
        q = queue.Queue(maxsize=max(0, depth)) if depth and depth > 0 else None
        stop_event = threading.Event()

        def producer_loop(pid: int):
            name = f"PrefetchProducer-{pid}"
            while not stop_event.is_set():
                try:
                    batch = dl.get_batch(NUMBER_OF_ROWS=batch_rows)
                    # Expecting numpy array at batch['velocity_data']
                    vel_np = batch['velocity_data']
                    x_cpu = torch.from_numpy(vel_np).float()
                    # Pin to speed up H2D on CUDA (safe no-op on CPU)
                    try:
                        x_cpu = x_cpu.pin_memory()
                    except Exception:
                        pass
                    if q is not None:
                        put_ok = False
                        while not stop_event.is_set() and not put_ok:
                            try:
                                q.put(x_cpu, timeout=0.1)
                                put_ok = True
                            except queue.Full:
                                continue
                    else:
                        # depth==0: no prefetching, nothing to do
                        time.sleep(0.001)
                except Exception as e:
                    # propagate error by placing it in queue (once)
                    if q is not None:
                        try:
                            q.put({'__error__': str(e)}, timeout=0.1)
                        except Exception:
                            pass
                    break

        threads = []
        for i in range(max(1, producers)):
            th = threading.Thread(target=producer_loop, args=(i,), name=f"PrefetchProducer-{i}", daemon=True)
            th.start()
            threads.append(th)
        return q, stop_event, threads

    # Initialize persistent prefetch (host side)
    prefetch_enabled = bool(effective_prefetch_depth and effective_prefetch_depth > 0)
    pf_mode = 'none'
    pf_queue = None
    pf_stop = None
    pf_threads = []
    mp_procs = []
    pf_producer_count = 0
    if prefetch_enabled and args.mp_prefetch and effective_mp_producers > 0:
        # Multiprocessing prefetch: bypass GIL for CPU-bound get_batch
        pf_mode = 'mp'
        pf_producer_count = effective_mp_producers
        if args.debug_profile:
            logger.info(f"Starting MP prefetcher with depth={effective_prefetch_depth} using {effective_mp_producers} process(es)")
        try:
            try:
                mp.set_start_method('spawn', force=False)
            except RuntimeError:
                # Start method already set
                pass
            mp_queue = mp.Queue(maxsize=max(1, effective_prefetch_depth))
            mp_stop_evt = mp.Event()
            for i in range(effective_mp_producers):
                p = mp.Process(
                    target=_mp_prefetch_worker,
                    args=(preferences.training_data_path,
                          effective_batch_size,
                          effective_num_workers,
                          effective_cache_size,
                          True,
                          mp_queue,
                          mp_stop_evt,
                          i,
                          effective_mp_producers,
                          args.dl_profile),
                    name=f"MPPrefetch-{i}",
                    daemon=True,
                )
                p.start()
                try:
                    logger.info(f"Started MP producer {i} (pid={p.pid})")
                except Exception:
                    pass
                mp_procs.append(p)
            pf_queue = mp_queue
            pf_stop = mp_stop_evt
        except Exception as e:
            logger.error(f"Failed to start multiprocessing prefetchers, falling back to threads: {e}")
            pf_mode = 'thread'
            pf_queue, pf_stop, pf_threads = _start_prefetch_farm(dataloader, effective_batch_size, effective_prefetch_depth, effective_producers)
            pf_producer_count = effective_producers
    elif prefetch_enabled:
        pf_mode = 'thread'
        pf_producer_count = effective_producers
        if args.debug_profile:
            logger.info(f"Starting prefetcher with depth={effective_prefetch_depth} using {effective_producers} producer thread(s)")
        pf_queue, pf_stop, pf_threads = _start_prefetch_farm(dataloader, effective_batch_size, effective_prefetch_depth, effective_producers)
    else:
        pf_mode = 'none'
        pf_queue = pf_stop = None
        pf_threads = []

    # Decide device-side prefetch default
    device_prefetch_enabled = ((torch.cuda.is_available() and not args.no_device_prefetch) or args.device_prefetch) and (device.type == 'cuda')
    logger.info(
        f"Device-side prefetch: {'ENABLED' if device_prefetch_enabled else 'DISABLED'} "
        f"(cuda={device.type == 'cuda'}, flag=--device_prefetch, no_device_prefetch={args.no_device_prefetch}); "
        f"host prefetch mode={pf_mode}"
    )

    for epoch in range(start_epoch, effective_num_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_mmd_loss = 0
        epoch_triplet_loss = 0
        num_batches = 0
        
        epoch_start_time = time.time()

        # Prefetcher is persistent across epochs; nothing to start/stop here.
        if args.debug_profile and prefetch_enabled and (epoch == start_epoch):
            logger.info(f"Prefetcher active: depth={effective_prefetch_depth}, producers={pf_producer_count}")
        # device_prefetch_enabled decided before epoch loop

        def _dequeue_batch():
            nonlocal consecutive_timeouts, latest_metrics
            t0 = time.time() if args.debug_profile else None
            if prefetch_enabled:
                while True:
                    try:
                        item = pf_queue.get(timeout=0.1)
                        try:
                            # Only present on threading.Queue
                            pf_queue.task_done()
                        except Exception:
                            pass
                        # Handle control messages from MP workers
                        if isinstance(item, dict):
                            if item.get('__metrics__'):
                                latest_metrics = item['__metrics__']
                                continue  # keep pulling for real data
                            if item.get('__error__'):
                                raise RuntimeError(f"Prefetch error: {item['__error__']}")
                        # Got a data item, reset timeout counter and proceed
                        consecutive_timeouts = 0
                        break
                    except Exception:
                        # Includes queue.Empty and multiprocessing queue timeouts
                        consecutive_timeouts += 1
                        if args.debug_profile and (consecutive_timeouts % 50 == 0):
                            try:
                                occ = pf_queue.qsize()
                            except Exception:
                                occ = -1
                            logger.debug(
                                "Waiting for prefetch... timeouts=%d, qsize=%s/%s, mode=%s",
                                consecutive_timeouts, occ, (effective_prefetch_depth if prefetch_enabled else 0), pf_mode
                            )
                        continue
                # Item may be a torch.Tensor (thread mode) or numpy.ndarray (mp mode)
                if isinstance(item, torch.Tensor):
                    x_cpu = item
                else:
                    # Treat as numpy array
                    x_cpu = torch.from_numpy(item).float()
                if device.type == 'cuda':
                    try:
                        x_cpu = x_cpu.pin_memory()
                    except Exception:
                        pass
            else:
                batch = dataloader.get_batch(NUMBER_OF_ROWS=effective_batch_size)
                vel_np = batch['velocity_data']
                x_cpu = torch.from_numpy(vel_np).float()
                if device.type == 'cuda':
                    try:
                        x_cpu = x_cpu.pin_memory()
                    except Exception:
                        pass
            t1 = time.time() if args.debug_profile else None
            return x_cpu, ((t1 - t0) if args.debug_profile else None)

        if device_prefetch_enabled:
            # Create copy stream and CUDA events
            h2d_stream = torch.cuda.Stream()
            ev_ready = torch.cuda.Event()
            # Prime the first device batch (x_cpu already a tensor, pinned when possible)
            primed_x_cpu, t_get0 = _dequeue_batch()
            t_h2d0p = time.time() if args.debug_profile else None
            with torch.cuda.stream(h2d_stream):
                next_dev = primed_x_cpu.to(device, non_blocking=True)
                ev_ready.record(h2d_stream)
            t_h2d1p = time.time() if args.debug_profile else None

            for _ in range(effective_batches_per_epoch):
                # Wait for the next device batch to be ready on default stream
                t_wait0 = time.time() if args.debug_profile else None
                torch.cuda.current_stream().wait_event(ev_ready)
                t_wait1 = time.time() if args.debug_profile else None
                x = next_dev

                # Kick off H2D for the following batch
                next_x_cpu, t_get = _dequeue_batch()
                t_h2d0 = time.time() if args.debug_profile else None
                with torch.cuda.stream(h2d_stream):
                    next_dev = next_x_cpu.to(device, non_blocking=True)
                    ev_ready.record(h2d_stream)
                t_h2d1 = time.time() if args.debug_profile else None

                # Forward pass
                t_fwd0 = time.time() if args.debug_profile else None
                recon_x, z = model(x)
                t_fwd1 = time.time() if args.debug_profile else None
                # Loss
                t_loss0 = time.time() if args.debug_profile else None
                loss, recon_loss, mmd_loss, triplet_loss = model.loss_function(recon_x, x, z)
                t_loss1 = time.time() if args.debug_profile else None
                # Backward + step
                t_bwd0 = time.time() if args.debug_profile else None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t_bwd1 = time.time() if args.debug_profile else None

                # Metrics
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_mmd_loss += mmd_loss.item()
                epoch_triplet_loss += triplet_loss.item()
                num_batches += 1
                global_step += 1

                # Debug timings
                if args.debug_profile and (num_batches <= 5 or num_batches % 20 == 0):
                    # Attempt queue occupancy estimation (may fail on Windows)
                    occ = -1
                    if prefetch_enabled and pf_queue is not None:
                        try:
                            occ = pf_queue.qsize()
                        except Exception:
                            occ = -1
                    logger.debug(
                        "Batch %d timings (s): get_batch=%.3f, H2D_async=%.3f, wait_event=%.3f, fwd=%.3f, loss=%.3f, bwd+step=%.3f (device_prefetch) | qsize=%s/%s",
                        num_batches,
                        (t_get if 't_get' in locals() else t_get0) or -1.0,
                        (t_h2d1 - t_h2d0) if 't_h2d0' in locals() else (t_h2d1p - t_h2d0p),
                        (t_wait1 - t_wait0),
                        (t_fwd1 - t_fwd0),
                        (t_loss1 - t_loss0),
                        (t_bwd1 - t_bwd0),
                        occ,
                        (effective_prefetch_depth if prefetch_enabled else 0),
                    )

                # TensorBoard per-step
                writer.add_scalar('Loss/train_step', loss.item(), global_step)
                writer.add_scalar('Loss/recon_step', recon_loss.item(), global_step)
                writer.add_scalar('Loss/mmd_step', mmd_loss.item(), global_step)
                writer.add_scalar('Loss/triplet_step', triplet_loss.item(), global_step)
        else:
            for _ in range(effective_batches_per_epoch):
                # Get batch tensor from dataloader (or prefetch queue). Already tensorized; pinned on CUDA.
                t_batch0 = time.time() if args.debug_profile else None
                x_cpu, _tget = _dequeue_batch()
                t_batch1 = time.time() if args.debug_profile else None

                # Move to device
                t_conv0 = time.time() if args.debug_profile else None
                x = x_cpu.to(device, non_blocking=(device.type == 'cuda'))
                if args.debug_profile and device.type == 'cuda':
                    torch.cuda.synchronize()
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

        
        # Per-epoch: no prefetcher shutdown; it's persistent. Optionally log last-step metrics to W&B.
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
    
    # Shutdown persistent prefetcher before saving
    if prefetch_enabled:
        try:
            if pf_stop is not None:
                try:
                    pf_stop.set()
                except Exception:
                    pass
            # Drain queue if possible (thread queue supports task_done; mp queue does not on Windows)
            if pf_queue is not None:
                try:
                    while True:
                        item = pf_queue.get_nowait()
                        try:
                            pf_queue.task_done()
                        except Exception:
                            pass
                except Exception:
                    pass
            # Join thread producers
            for th in pf_threads:
                try:
                    th.join(timeout=5.0)
                except Exception:
                    pass
            # Join/terminate mp producers
            try:
                for p in mp_procs:
                    if p.is_alive():
                        p.join(timeout=5.0)
                for p in mp_procs:
                    if p.is_alive():
                        p.terminate()
                # Close and join mp queue if available
                try:
                    pf_queue.close()
                    pf_queue.join_thread()
                except Exception:
                    pass
            except Exception:
                pass
            logger.info("Prefetcher stopped.")
        except Exception as _e:
            logger.warning(f"Error during prefetcher shutdown: {_e}")

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