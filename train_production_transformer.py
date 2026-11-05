"""
Transformer Training Production Script for NeurIPS
=================================================
This script handles the training and architecture search for fluid dynamics latent sequences.

KEY METRICS TO MONITOR:
1. Rollout L2: Multi-step autoregressive prediction error (28 full time steps).
2. Persistence L2: The baseline "do-nothing" error (assuming t=12 is the result for all future t).
3. Persistence Improvement %: How much BETTER the model is than doing nothing. 
   Goal: MUST be > 0% and ideally > 50%.

WHERE TO LOOK:
- Search results: Console table at the end of the run.
- Telemetry: W&B project "transformer_neurIPS_production".
- Best Models: Saved in 'saved_models/' with '_rollout_best.pt' suffix.
"""

import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import sys
import time
try:
    from model_variants import get_model
except ImportError:
    from transformer_neurIPS.model_variants import get_model

# --- Configuration ---
class Config:
    """
    Global configuration for training and model parameters.
    Modify SEARCH_SPACE to test new architectures.
    """
    # Data paths - Absolute paths derived from file location
    TRAIN_H5 = os.path.join(os.path.dirname(__file__), "data/train_40.h5")
    VAL_H5 = os.path.join(os.path.dirname(__file__), "data/val_40.h5")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
    
    # Model architecture constants
    LATENT_DIM = 47 # Latent features from encoder
    NUM_X = 26      # Spatial locations per time step
    NUM_TIME = 40   # Total time steps in sequence
    SEQ_LEN = NUM_X * NUM_TIME # 1040 tokens total
    
    INPUT_DIM = 52  # 47 (latents) + 1 (time) + 1 (x) + 3 (params)
    
    # Defaults for single run (overwritten by SEARCH_SPACE if searching)
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.01
    BIAS = True
    VARIANT = 'base' 
    
    # Training Hyperparameters
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # Micro-batching: BATCH_SIZE is the PHYSICAL per-step batch fed through
    # forward/backward. A single 512-batch x 1039-token step was the actual
    # bottleneck on the B300 (huge activations for one step means the SM
    # occupancy/launch pattern is poor and there's nothing to overlap with --
    # HBM capacity was never the constraint). Splitting it into smaller
    # micro-batches and accumulating gradients keeps the effective batch
    # (used by AdamW/OneCycleLR) identical while giving the GPU many smaller,
    # cheaper steps instead of one giant one.
    BATCH_SIZE = 64
    ACCUMULATION_STEPS = 8 # Effective batch = BATCH_SIZE * ACCUMULATION_STEPS = 512
    # Validation only does a forward pass (no backward/optimizer state), so it
    # doesn't need micro-batching -- use one big batch to minimize the number
    # of Python-level loop iterations.
    EVAL_BATCH_SIZE = 512
    LEARNING_RATE = 2e-3
    EPOCHS = 20
    # Fraction of the training set actually used. Lets you check how much of
    # the slowness/quality is a function of dataset size before committing to
    # full-dataset runs -- e.g. 0.1 uses ~746 samples (~2 batches/epoch).
    # Validation stays at full size (below) so quality comparisons across
    # different TRAIN_SUBSET_RATIO runs remain meaningful.
    TRAIN_SUBSET_RATIO = 0.1
    # The whole training set is ~7.5k samples x 1040 tokens x 52 dims ~= 1.6GB
    # as float32 -- small enough to load entirely into RAM once (see
    # TransformerDataset below) instead of going through a multiprocess
    # DataLoader that re-opens the HDF5 file per worker per process. That
    # per-worker cold-open was costing minutes at the START of every run
    # (and every architecture-search candidate, since each gets a fresh
    # DataLoader). With the dataset already resident in memory, num_workers=0
    # is the FAST path, not a fallback -- indexing an in-memory tensor is
    # microseconds, so there is nothing for background workers to overlap.
    NUM_WORKERS = 0
    PIN_MEMORY = True
    # Irrelevant when NUM_WORKERS=0 (no workers to keep alive), kept only so
    # bumping NUM_WORKERS back up for a much larger future dataset doesn't
    # silently regress to the old cold-open-every-epoch behavior.
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 4

    # Stability & Robustness Techniques
    # Micro-batching setup: BATCH_SIZE is the per-step batch, 
    # effective batch size = BATCH_SIZE * ACCUMULATION_STEPS.
    NOISE_STD = 5e-4       
    AR_ROLLOUT_STEPS = 5 # Reduced from 10 to speed up training; sequential loops are slow
    AR_LOSS_WEIGHT = 0.05
    # Only compute the (very expensive, sequential) AR rollout aux loss every N training
    # steps instead of on every batch. With N=8 the AR-loss compute overhead drops ~8x
    # while still providing regularization signal.
    AR_EVERY_N_STEPS = 8

    # In-training "robust" TorchScript tracing does model.to('cpu') -> jit.trace ->
    # model.to(DEVICE) which is a catastrophic pipeline stall on a B300 and can also
    # invalidate torch.compile artifacts. Default OFF; enable only when you want a
    # portable artifact.
    SAVE_SCRIPTED_MODELS = False
    
    # Persistence Baseline Evaluation Parameters
    VAL_CONTEXT_STEPS = 12 # Feed first 12 steps as context
    VAL_ROLLOUT_STEPS = 26 * (40 - VAL_CONTEXT_STEPS) # Predict remaining 28 steps (728 tokens)
    
    # Validation settings for speed on MPS
    VAL_INTERVAL = 5 # Validate every 5 epochs to speed up training on powerful cards
    
    # METRIC DEFINITION:
    # Persistence MSE = MSE(Target_Steps_13_to_40, Step_12_Repeated)
    # Model MSE = MSE(Target_Steps_13_to_40, Model_Predictions_13_to_40)
    # Goal: Model MSE < Persistence MSE

    USE_TF32 = True # Enable TensorFloat32 for Hopper/Blackwell speedup
    USE_CUDNN_BENCHMARK = True # Autotune convolution kernels for the ConvBlock variants

    # Architecture search space - Exploring diverse architectural inductive biases
    SEARCH_SPACE = [
        # 0: Baseline - Standard capacity (Strongest early performer)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "base"},
        
        # 1: Expressive Gating - SwiGLU activation (Llama-style, High throughput)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "swiglu"},
        
        # 2: Efficiency - Multi-Query Attention (MQA)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "mqa"},
        
        # 3: Hybrid - Convolutional-Transformer (Local spatial inductive bias)
        {"EMBED_SIZE": 256, "N_HEADS": 8, "N_LAYERS": 6, "VARIANT": "conv"},
        
        # 4: Deep Capacity Baseline
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 8, "VARIANT": "base"},
        
        # 5: Hybrid Wide - Conv + SwiGLU (Combining local bias with advanced gating)
        {"EMBED_SIZE": 512, "N_HEADS": 8, "N_LAYERS": 4, "VARIANT": "conv", "USE_SWIGLU": True},
    ]

    # --- Runtime logic ---
    MAX_RUNTIME_PER_CANDIDATE = 86400 # 24 hours for production runs on B300
    CHECKPOINT_INTERVAL = 60 # Save every 60 seconds
    
    # H200 Optimization Recommendations:
    # 1. Use torch.compile(model) for Blackwell/Hopper speedups.
    # 2. Use torch.cuda.amp.autocast() or FP8 TransformerEngine if available.
    # 3. Increase ACCUMULATION_STEPS (not BATCH_SIZE) to raise the effective
    #    batch size -- keep the physical BATCH_SIZE small so each step stays
    #    cheap; see the micro-batching comment above.
    # 4. Use FlashAttention-3 kernels.

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def l2_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

class TransformerDataset(Dataset):
    def __init__(self, h5_path, subset_ratio=1.0):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_length = f['data'].shape[0]
            self.length = int(total_length * subset_ratio)
            if self.length == 0 and total_length > 0:
                self.length = 1
            # Read the (small, ~1.6GB at full size) slice we actually need
            # ONCE here, in the main process, and keep it resident as a
            # tensor. This is the one HDF5 read for this dataset's lifetime --
            # __getitem__ below never touches h5py again, so there's no
            # per-worker file handle to open and no repeated disk/network I/O.
            raw = f['data'][:self.length]
        self.data = torch.from_numpy(raw).float().reshape(self.length, Config.SEQ_LEN, Config.INPUT_DIM)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


class InMemoryBatcher:
    """Direct in-memory batch iterator.

    Replaces torch.utils.data.DataLoader for the case where the entire
    dataset is already a single resident tensor (see TransformerDataset).
    We hit 90% dataloader-wait epochs even with num_workers=0 because the
    DataLoader still went through collate + pin_memory + per-batch H2D
    copy for every micro-batch. This class skips all of that: it holds
    ONE tensor (already on the target device, if it fits), reshuffles a
    permutation of row indices at the start of each epoch, and yields
    contiguous slices of that tensor. No workers, no copies per batch,
    no HDF5 handle -- indexing an already-resident tensor is microseconds.
    """

    def __init__(self, data, batch_size, shuffle):
        self.data = data
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n = int(data.shape[0])

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            # randperm on the same device as the data avoids an
            # index-tensor H2D copy every epoch.
            idx = torch.randperm(self.n, device=self.data.device)
        else:
            idx = torch.arange(self.n, device=self.data.device)
        for start in range(0, self.n, self.batch_size):
            yield self.data.index_select(0, idx[start:start + self.batch_size])


def _preload_to_device(dataset, name, device):
    """Move an in-memory TransformerDataset's tensor to `device` if it fits.

    Prints progress and falls back to keeping it on CPU if the GPU copy
    raises (e.g. OOM). This is done ONCE per run, so any transfer cost
    here amortizes across every epoch, every AR-aux rollout, and every
    validation pass.
    """
    n_bytes = dataset.data.numel() * dataset.data.element_size()
    print(f"  [diag] preloading '{name}' dataset to {device}: "
          f"{dataset.data.shape} ({n_bytes / 1e9:.2f} GB float32) ...", flush=True)
    if device == "cpu":
        print(f"  [diag]   -> device is cpu; leaving tensor in host memory.")
        return dataset.data
    _t0 = time.time()
    try:
        dataset.data = dataset.data.to(device, non_blocking=False)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"  [diag]   -> resident on {device} after {time.time()-_t0:.1f}s.")
    except (RuntimeError, MemoryError) as e:
        print(f"  [diag]   -> {device} preload FAILED ({e.__class__.__name__}: {e}). "
              f"Keeping '{name}' on CPU; batches will still bypass DataLoader.")
        if device == "cuda":
            torch.cuda.empty_cache()
    return dataset.data

def train(variant_idx=None):
    start_time = time.time()
    if variant_idx is not None:
        variant_cfg = Config.SEARCH_SPACE[variant_idx]
        Config.EMBED_SIZE = variant_cfg["EMBED_SIZE"]
        Config.N_HEADS = variant_cfg["N_HEADS"]
        Config.N_LAYERS = variant_cfg["N_LAYERS"]
        Config.VARIANT = variant_cfg["VARIANT"]
    
    variant = Config.VARIANT
    # Stable run name for resumption (excludes timestamp if searching or top3)
    if variant_idx is not None:
        run_name = f"production_{variant}_E{Config.EMBED_SIZE}_L{Config.N_LAYERS}"
    else:
        run_name = f"production_{variant}_E{Config.EMBED_SIZE}_L{Config.N_LAYERS}_{int(time.time())}"
    
    wandb.init(project="runpod_b300_v1", name=run_name, id=run_name, resume="allow", config={
        "variant": variant,
        "embed_size": Config.EMBED_SIZE,
        "n_layers": Config.N_LAYERS,
        "n_heads": Config.N_HEADS,
        "noise_std": Config.NOISE_STD,
        "ar_steps": Config.AR_ROLLOUT_STEPS,
        "ar_weight": Config.AR_LOSS_WEIGHT,
        "val_context_steps": Config.VAL_CONTEXT_STEPS,
        "val_rollout_steps": Config.VAL_ROLLOUT_STEPS
    })
    
    # --- Data Loading ---
    _data_load_t0 = time.time()
    train_dataset = TransformerDataset(Config.TRAIN_H5, subset_ratio=1.0)
    val_dataset = TransformerDataset(Config.VAL_H5, subset_ratio=1.0)
    _data_load_time = time.time() - _data_load_t0
    print(f"  [diag] one-time HDF5 read into memory = {_data_load_time:.1f}s "
          f"(train={len(train_dataset):,} + val={len(val_dataset):,} samples, "
          f"~{(train_dataset.data.numel() + val_dataset.data.numel()) * 4 / 1e9:.2f} GB as float32). "
          f"If this is still slow, the bottleneck is the underlying storage (e.g. a "
          f"network-mounted volume), not the DataLoader.")
    # --- Bypass DataLoader entirely: hold the full dataset on-device. ---
    # Diagnostics on the previous DataLoader-based pipeline showed epochs
    # spending ~90% of wall time blocked on next(loader). With the whole
    # (small, ~1.6GB) dataset already read into a tensor by
    # TransformerDataset.__init__, going back through DataLoader just to
    # re-collate + pin_memory + copy each micro-batch is pure overhead.
    # We move both tensors to the training device once here and then
    # iterate them directly via InMemoryBatcher.
    _preload_to_device(train_dataset, "train", Config.DEVICE)
    _preload_to_device(val_dataset, "val", Config.DEVICE)
    _persistent = False
    _prefetch = None
    train_loader = InMemoryBatcher(
        train_dataset.data, batch_size=Config.BATCH_SIZE, shuffle=True,
    )
    val_loader = InMemoryBatcher(
        val_dataset.data, batch_size=Config.EVAL_BATCH_SIZE, shuffle=False,
    )

    # --- Pipeline diagnostics (printed on every run, not just run_diagnostics.py) ---
    # These are the numbers that actually explain "why is this epoch slow":
    # a misconfigured DataLoader (workers respawning every epoch, re-opening
    # the HDF5 file) shows up here immediately instead of needing a separate
    # profiling pass.
    print("\n[Pipeline diagnostics]")
    print(f"  train samples = {len(train_dataset):,}  ->  ~{len(train_loader)} micro-batches/epoch")
    print(f"  micro_batch_size={Config.BATCH_SIZE}  accumulation_steps={Config.ACCUMULATION_STEPS}  "
          f"effective_batch_size={Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}  "
          f"eval_batch_size={Config.EVAL_BATCH_SIZE}")
    print(f"  DataLoader DISABLED -- iterating directly over in-memory tensors "
          f"(resident on '{train_dataset.data.device}' / '{val_dataset.data.device}').")
    print(f"  No workers, no pin_memory staging, no per-batch H2D copies.")

    if getattr(Config, 'USE_TF32', False) and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        # Also allow TF32 explicitly for cuBLAS/cuDNN paths that don't consult the
        # matmul-precision hint.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TensorFloat32 (TF32) enabled.")
    if getattr(Config, 'USE_CUDNN_BENCHMARK', False) and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled.")

    model = get_model(Config).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    best_val_loss = float('inf')
    best_train_l2 = float('inf')
    best_rollout_loss = float('inf')
    best_improvement = -float('inf')
    last_checkpoint_time = time.time()
    
    # Throughput and logging state
    last_log_time = time.time()
    tokens_since_last_log = 0
    
    start_epoch = 0
    resume_batch_idx = -1

    # --- Checkpoint Resumption (Before compilation to avoid prefix issues) ---
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    latest_cp_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")
    if os.path.exists(latest_cp_path):
        print(f"Resuming from checkpoint: {latest_cp_path}")
        try:
            checkpoint = torch.load(latest_cp_path, map_location=Config.DEVICE)
            
            # Handle state_dict key mismatch (torch.compile prefix '_orig_mod.')
            state_dict = checkpoint['model_state_dict']
            first_key = next(iter(state_dict))
            if first_key.startswith('_orig_mod.') and not (hasattr(model, '_orig_mod')):
                # Checkpoint was compiled, current model is not
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            elif not first_key.startswith('_orig_mod.') and (hasattr(model, '_orig_mod')):
                # Checkpoint was not compiled, current model is
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            
            # strict=False so the removed (SEQ_LEN, SEQ_LEN) `causal_mask` buffer
            # from older checkpoints doesn't block resumption.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                # Filter out the known-safe legacy key so we don't spam warnings.
                unexpected = [k for k in unexpected if not k.endswith('causal_mask')]
            if missing or unexpected:
                print(f"  -> load_state_dict: missing={missing}, unexpected={unexpected}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            if 'batch_idx' in checkpoint:
                resume_batch_idx = checkpoint['batch_idx']
                print(f"  -> Resuming from Epoch {start_epoch + 1}, Batch {resume_batch_idx + 1}")
            else:
                start_epoch += 1
                print(f"  -> Resuming from Epoch {start_epoch + 1}")
            
            if 'val_l2' in checkpoint: best_val_loss = checkpoint['val_l2']
            if 'rollout_mse' in checkpoint: best_rollout_loss = checkpoint['rollout_mse']
            if 'improvement' in checkpoint: best_improvement = checkpoint['improvement']
        except Exception as e:
            print(f"  -> Warning: Could not load checkpoint: {e}")

    # Bail out BEFORE compiling if this candidate already hit its epoch target --
    # compiling just to immediately skip wastes ~11s per candidate for nothing.
    if start_epoch >= Config.EPOCHS:
        print(f"Training already completed up to {start_epoch} epochs. Target is {Config.EPOCHS}. Skipping training.")
        wandb.finish()
        return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

    # Use torch.compile for faster execution on supported devices
    if hasattr(torch, "compile") and Config.DEVICE == "cuda":
        try:
            # Safety margin: the model is also invoked (below, via raw_model)
            # with a handful of other fixed batch sizes (full val batch,
            # AR-aux subset, rollout subset). Bumping this costs nothing
            # unless it's actually needed.
            torch._dynamo.config.cache_size_limit = 32
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    # `model` (possibly compiled) is used ONLY for the fixed-shape
    # teacher-forced forward/backward pass -- that's the actual hot path
    # and where torch.compile earns its keep.
    #
    # `raw_model` (always the eager module) is used for every autoregressive
    # rollout loop (AR aux loss below, and the validation multi-step
    # rollout). Those loops feed the model a NEW sequence length on every
    # single step (e.g. validation grows T from 312 -> 1039, one token at a
    # time -- 728 distinct shapes). Running that through the compiled graph
    # means Dynamo either recompiles per shape (each recompile costs
    # roughly as much as the initial ~11s cold compile) or, once it hits
    # torch._dynamo.config.cache_size_limit, falls back to eager anyway
    # after burning time on the recompiles it did attempt. This was almost
    # certainly the dominant cause of epochs getting slower over time: it
    # re-triggers every VAL_INTERVAL epochs and gets worse as more distinct
    # shapes pile up against the compile cache. Using the eager module here
    # sidesteps recompilation entirely -- there's nothing to gain from
    # compiling a single-token decode step on a 4.78M param model anyway.
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # OneCycleLR for faster convergence
    # Adjust steps_per_epoch for gradient accumulation
    effective_steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    if len(train_loader) % Config.ACCUMULATION_STEPS != 0:
        effective_steps_per_epoch += 1
    
    # Ensure total_steps is at least 10 to avoid ZeroDivisionError with small pct_start
    total_steps = effective_steps_per_epoch * Config.EPOCHS
    if total_steps < 10:
        pct_start = 0.3 # Higher pct_start for very short runs
    else:
        pct_start = 0.1

    try:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=Config.LEARNING_RATE, 
            steps_per_epoch=effective_steps_per_epoch, 
            epochs=Config.EPOCHS,
            pct_start=pct_start,
            last_epoch=start_epoch * effective_steps_per_epoch - 1 if start_epoch > 0 else -1
        )
    except Exception as e:
        print(f"  -> Warning: Could not initialize OneCycleLR: {e}. Falling back to constant LR.")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)

    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(device='cuda', enabled=(Config.DEVICE == "cuda"))

    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0
        val_loss = float('inf') # Initialize to avoid UnboundLocalError when validation is skipped

        # Determine if we need to skip batches (only for the first epoch of resumption)
        skip_count = 0
        if epoch == start_epoch and resume_batch_idx >= 0:
            skip_count = resume_batch_idx + 1

        # Step-level baseline metrics are only computed on some batches (see
        # the `batch_idx % 50 == 0` block below). Initialize them here so that
        # if the very first *non-skipped* batch this epoch isn't a
        # baseline-compute batch (e.g. resuming mid-epoch and landing on
        # batch_idx=50 before batch_idx=0 has ever been seen this run), we
        # don't hit UnboundLocalError when the logging block references them.
        step_persistence_mse = float('nan')
        step_model_mse = float('nan')

        # --- Per-epoch pipeline diagnostics ---
        # dataloader_wait_total accumulates time spent blocked on next(pbar)
        # (worker startup, HDF5 reads, collation, H2D copy of the *fetch*
        # itself). If this stays large after epoch 1 with persistent_workers
        # enabled, the workers aren't actually persisting.
        epoch_t0 = time.time()
        dataloader_wait_total = 0.0
        _batch_fetch_t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            # Efficiently skip batches that were already processed
            if skip_count > 0:
                skip_count -= 1
                if skip_count % 100 == 0 or skip_count == 0:
                    pbar.set_postfix({'status': f'fast-forwarding ({skip_count} left)'})
                continue

            if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
                print(f"\nReached {Config.MAX_RUNTIME_PER_CANDIDATE}s limit. Moving to next candidate.")
                wandb.finish()
                return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

            _fetch_wait = time.time() - _batch_fetch_t0
            dataloader_wait_total += _fetch_wait
            if epoch == start_epoch and batch_idx == 0:
                print(f"  [diag] first-batch cold latency this run = {_fetch_wait*1000:.0f} ms "
                      f"(DataLoader bypassed -- pure in-memory index_select)")

            # In-memory batcher already yields tensors on the target device
            # when preload succeeded; .to() is a no-op in that case and a
            # safe fallback if preload had to keep the tensor on CPU.
            if batch.device.type != Config.DEVICE.split(':')[0]:
                batch = batch.to(Config.DEVICE, non_blocking=True)
            inputs = batch[:, :-1, :]
            targets = batch[:, 1:, :Config.LATENT_DIM]
            
            # Noise injection
            if Config.NOISE_STD > 0:
                noise = torch.zeros_like(inputs)
                noise[:, :, :Config.LATENT_DIM] = torch.randn_like(inputs[:, :, :Config.LATENT_DIM]) * Config.NOISE_STD
                inputs = inputs + noise
            
            # Primary Teacher-Forced Pass
            with torch.amp.autocast(device_type=('cuda' if 'cuda' in Config.DEVICE else 'cpu'), enabled=('cuda' in Config.DEVICE or 'mps' in Config.DEVICE)):
                outputs = model(inputs)
                loss = l2_loss(outputs, targets)
                
                # --- Step-Level Baseline Check ---
                # Cadence matches the wandb logging block below (every 50
                # batches) so the logged step_persistence_mse / step_model_mse
                # values are always freshly computed from the current batch,
                # not stale from ~50 batches earlier, and are guaranteed to be
                # set before that log fires (previously only set every 100 --
                # first log at batch_idx=50 hit UnboundLocalError).
                if batch_idx % 50 == 0:
                    with torch.no_grad():
                        prev_frame = inputs[:, -1:, :Config.LATENT_DIM]
                        step_target = targets[:, -1:, :]
                        step_persistence_mse = mse_loss(prev_frame, step_target).item()
                        step_model_mse = mse_loss(outputs[:, -1:, :], step_target).item()

                # --- Optional Short AR Rollout Loss ---
                # Only run the sequential AR loop every N steps. The sequential
                # forward passes inside this block cannot use CUDA graphs and
                # dominate step time when done every iteration.
                _ar_every = max(1, getattr(Config, 'AR_EVERY_N_STEPS', 1))
                if (Config.AR_ROLLOUT_STEPS > 0
                        and Config.AR_LOSS_WEIGHT > 0
                        and batch_idx % _ar_every == 0):
                    # Optimized AR rollout: Use a very small subset and limited context for training loss
                    ar_batch_size = min(inputs.shape[0], 16) # Reduced from 32
                    ar_context_len = min(inputs.shape[1], 128) # Reduced from 256
                    
                    ar_inputs = inputs[:ar_batch_size, :ar_context_len, :].clone()
                    ar_targets = targets[:ar_batch_size, ar_context_len : ar_context_len + Config.AR_ROLLOUT_STEPS, :]
                    
                    curr = ar_inputs
                    ar_preds = []
                    # AR loop is still sequential, but on smaller data it's faster.
                    # Uses raw_model (eager) -- see comment at torch.compile call site.
                    for _ in range(Config.AR_ROLLOUT_STEPS):
                        out = raw_model(curr)
                        next_lat = out[:, -1:, :]
                        ar_preds.append(next_lat)
                        
                        next_idx = curr.shape[1]
                        if next_idx >= inputs.shape[1]: break
                        
                        # Optimization: Avoid clone if possible, but cat creates new tensor anyway
                        next_tok = inputs[:ar_batch_size, next_idx : next_idx + 1, :].clone()
                        next_tok[:, :, :Config.LATENT_DIM] = next_lat
                        curr = torch.cat([curr, next_tok], dim=1)
                    
                    if len(ar_preds) == Config.AR_ROLLOUT_STEPS:
                        ar_preds = torch.cat(ar_preds, dim=1)
                        ar_loss = l2_loss(ar_preds, ar_targets)
                        loss = loss + Config.AR_LOSS_WEIGHT * ar_loss
                
                # Normalize loss for accumulation
                loss = loss / Config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * Config.ACCUMULATION_STEPS

            # Logging step-level metrics
            pbar.set_postfix({'l2': loss.item()})
            if batch_idx % 50 == 0:
                # Calculate throughput: tokens = batch_size * seq_len
                # For this specific task, seq_len = Config.SEQ_LEN - 1 (target length)
                tokens_in_batch = batch.shape[0] * (batch.shape[1] - 1)
                
                # Simple timing for throughput (approximate)
                tokens_since_last_log += tokens_in_batch
                current_time = time.time()
                dt = current_time - last_log_time
                
                throughput = 0
                if dt > 1.0: # Only update throughput every second to avoid noise
                    throughput = tokens_since_last_log / dt
                    last_log_time = current_time
                    tokens_since_last_log = 0

                log_dict = {
                    "step_train_l2": loss.item(),
                    "step_persistence_mse": step_persistence_mse,
                    "step_model_mse": step_model_mse,
                    "step_improvement_pct": (step_persistence_mse - step_model_mse) / (step_persistence_mse + 1e-8) * 100,
                    "lr": scheduler.get_last_lr()[0],
                    "throughput_tokens_sec": throughput
                }
                
                # Hardware metrics (CUDA specific)
                if torch.cuda.is_available():
                    log_dict.update({
                        "cuda_vram_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                        "cuda_vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                        "cuda_max_vram_gb": torch.cuda.max_memory_allocated() / 1e9
                    })
                elif torch.backends.mps.is_available():
                    log_dict.update({
                        "mps_vram_allocated_gb": torch.mps.current_allocated_memory() / 1e9
                    })
                
                wandb.log(log_dict)
            
            # --- Periodic Checkpointing (Every Minute) ---
            if time.time() - last_checkpoint_time > Config.CHECKPOINT_INTERVAL:
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                cp_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_l2': best_val_loss,
                    'rollout_mse': best_rollout_loss,
                    'improvement': best_improvement,
                    'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
                }, cp_path)
                
                # Robust checkpointing: save scripted/traced model for portability.
                # GATED: Disabled by default because moving the LIVE training model
                # to CPU for tracing and back stalls the GPU pipeline for many
                # seconds and can invalidate torch.compile caches. Enable via
                # Config.SAVE_SCRIPTED_MODELS when you actually need the artifact.
                try:
                    if not getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                        raise RuntimeError('scripted-save disabled by config')
                    # We attempt to script a clean version of the model.
                    if hasattr(model, "_orig_mod"):
                        model_to_save = model._orig_mod
                    else:
                        model_to_save = model
                    
                    # Use trace as primary for robust architecture embedding since it handles
                    # most standard Transformer patterns better than scripting.
                    try:
                        # Use CPU for tracing to avoid device-specific issues in the trace
                        model_cpu = model_to_save.to('cpu')
                        model_cpu.eval() # Ensure eval mode for tracing
                        dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                        with torch.no_grad():
                            traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                        torch.jit.save(traced_model, cp_path.replace(".pt", "_scripted.pt"))
                        # Move back to original device and train mode
                        model_to_save.to(Config.DEVICE)
                        model_to_save.train()
                    except Exception as trace_err:
                        # Fallback to scripting if tracing fails
                        scripted_model = torch.jit.script(model_to_save)
                        torch.jit.save(scripted_model, cp_path.replace(".pt", "_scripted.pt"))
                except Exception as e:
                    # Don't fail the whole run if scripting/tracing fails, but log it
                    print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                
                last_checkpoint_time = time.time()
                # Also log to wandb that we saved a checkpoint
                wandb.log({"checkpoint_saved": 1}, commit=False)

            _batch_fetch_t0 = time.time()

        train_loss /= len(train_loader)

        # --- Per-epoch pipeline diagnostics summary ---
        epoch_wall_time = time.time() - epoch_t0
        _wait_pct = 100.0 * dataloader_wait_total / epoch_wall_time if epoch_wall_time > 0 else 0.0
        print(f"  [diag] epoch {epoch+1} wall time = {epoch_wall_time:.1f}s   "
              f"dataloader wait = {dataloader_wait_total:.1f}s ({_wait_pct:.0f}%)")
        if _wait_pct > 25.0:
            print(f"  [diag] WARNING: dataloader wait is {_wait_pct:.0f}% of epoch time -- "
                  f"pipeline is I/O-bound, not compute-bound. Check persistent_workers is "
                  f"actually taking effect (see cold-latency line above) before tuning the model.")
        wandb.log({"epoch_wall_time_s": epoch_wall_time,
                   "epoch_dataloader_wait_s": dataloader_wait_total,
                   "epoch_dataloader_wait_pct": _wait_pct,
                   "epoch": epoch}, commit=False)

        # Save best model based on training L2 performance
        if train_loss < best_train_l2:
            best_train_l2 = train_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            # Existing convention: saved_models/[run_name]_train_best.pt
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_train_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_l2': train_loss,
                'val_l2': best_val_loss,
                'rollout_mse': best_rollout_loss,
                'improvement': best_improvement,
                'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
            }, save_path)

            # Robust checkpointing: save scripted/traced model.
            # GATED off by default: this block moves the LIVE model to CPU and
            # back, which stalls the GPU. Enable via Config.SAVE_SCRIPTED_MODELS.
            try:
                if not getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                    raise RuntimeError('scripted-save disabled by config')
                if hasattr(model, "_orig_mod"):
                    model_to_save = model._orig_mod
                else:
                    model_to_save = model

                try:
                    model_cpu = model_to_save.to('cpu')
                    model_cpu.eval()
                    dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                    torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                    model_to_save.to(Config.DEVICE)
                    model_to_save.train()
                except:
                    scripted_model = torch.jit.script(model_to_save)
                    torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
            except Exception as e:
                if getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                    print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                
            print(f"  --> Saved new best training model (L2={train_loss:.6f})")
            print(f"\033[94m{save_path}\033[0m")

        if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
            print(f"\nReached {Config.MAX_RUNTIME_PER_CANDIDATE}s limit. Moving to next candidate.")
            wandb.finish()
            return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

        # Validation
        if (epoch + 1) % Config.VAL_INTERVAL == 0 or (epoch == Config.EPOCHS - 1 and Config.DEVICE == "cuda"):
            _val_t0 = time.time()
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch.device.type != Config.DEVICE.split(':')[0]:
                        batch = batch.to(Config.DEVICE, non_blocking=True)
                    inputs = batch[:, :-1, :]
                    targets = batch[:, 1:, :Config.LATENT_DIM]
                    outputs = model(inputs)
                    val_loss += l2_loss(outputs, targets).item()
            val_loss /= len(val_loader)
            _val_fwd_time = time.time() - _val_t0

            # --- Multi-step Rollout Evaluation (The "Most Concerning Metric") ---
            # NOTE: this loop runs up to 10 batches * VAL_ROLLOUT_STEPS sequential
            # single-token decode steps (up to 7,280 forward calls). Timed
            # separately below because it is the single most expensive block
            # in the whole training loop and easy to mistake for "training is
            # slow" when it's really "validation is slow".
            _rollout_t0 = time.time()
            model.eval()
            rollout_mse = 0
            persistence_mse = 0
            rollout_count = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= 10: break 
                    if batch.device.type != Config.DEVICE.split(':')[0]:
                        batch = batch.to(Config.DEVICE, non_blocking=True)
                    
                    context_len = 26 * Config.VAL_CONTEXT_STEPS
                    inputs = batch[:, :context_len, :]
                    targets = batch[:, context_len : context_len + Config.VAL_ROLLOUT_STEPS, :Config.LATENT_DIM]
                    
                    # --- Persistence Baseline (Static Step 12) ---
                    # Take the last frame of the context (the 12th time step)
                    last_frame = inputs[:, -26:, :Config.LATENT_DIM] # shape (B, 26, 47)
                    num_repeats = Config.VAL_ROLLOUT_STEPS // 26
                    persistence_preds = last_frame.repeat(1, num_repeats, 1)
                    persistence_mse += mse_loss(persistence_preds, targets).item()
    
                    # --- Model Rollout ---
                    curr = inputs
                    preds = []
                    # Optimization: Use even fewer batches for rollout during validation to save time
                    curr = curr[:8] # Only rollout 8 sequences
                    rollout_batch = batch[:8]
                    targets_subset = targets[:8]

                    # Uses raw_model (eager) -- see comment at torch.compile call
                    # site. This loop alone sweeps ~728 distinct sequence lengths;
                    # running it through the compiled graph triggers a recompile
                    # storm every VAL_INTERVAL epochs.
                    for _ in range(Config.VAL_ROLLOUT_STEPS):
                        out = raw_model(curr)
                        next_lat = out[:, -1:, :]
                        preds.append(next_lat)
                        
                        next_idx = curr.shape[1]
                        if next_idx >= rollout_batch.shape[1]: break
                        
                        next_tok = rollout_batch[:, next_idx : next_idx + 1, :].clone()
                        next_tok[:, :, :Config.LATENT_DIM] = next_lat
                        curr = torch.cat([curr, next_tok], dim=1)
                    
                    if len(preds) == Config.VAL_ROLLOUT_STEPS:
                        preds = torch.cat(preds, dim=1)
                        rollout_mse += mse_loss(preds, targets_subset).item()
                        rollout_count += 1
            
            if rollout_count > 0:
                rollout_mse /= rollout_count
                persistence_mse /= rollout_count

            _rollout_time = time.time() - _rollout_t0
            print(f"  [diag] validation: fixed-shape fwd = {_val_fwd_time:.1f}s   "
                  f"multi-step rollout = {_rollout_time:.1f}s "
                  f"({Config.VAL_ROLLOUT_STEPS} steps x up to 10 batches)")

            persistence_improvement = (persistence_mse - rollout_mse) / (persistence_mse + 1e-8) * 100
            best_improvement = max(best_improvement, persistence_improvement)

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "rollout_mse": rollout_mse,
                "persistence_mse": persistence_mse,
                "persistence_improvement_pct": persistence_improvement,
                "val_fwd_time_s": _val_fwd_time,
                "val_rollout_time_s": _rollout_time,
                "baseline_red_line": persistence_mse, # Red line on rollout_mse plot
                "epoch": epoch
            })
            print(f"Epoch {epoch+1}: Train L2={train_loss:.6f}, Val L2={val_loss:.6f}")
            print(f"         Rollout MSE={rollout_mse:.6f}, Persistence MSE={persistence_mse:.6f} ({persistence_improvement:.1f}% better)")
            
            # Save best model based on rollout performance
            if rollout_mse < best_rollout_loss and rollout_mse > 0:
                best_rollout_loss = rollout_mse
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_rollout_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'rollout_mse': rollout_mse,
                    'persistence_improvement': persistence_improvement,
                    'val_l2': best_val_loss,
                    'improvement': best_improvement,
                    'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
                }, save_path)
                
                # Robust checkpointing: save scripted/traced model.
                # GATED off by default: this stalls the GPU. Enable via Config.SAVE_SCRIPTED_MODELS.
                try:
                    if not getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                        raise RuntimeError('scripted-save disabled by config')
                    if hasattr(model, "_orig_mod"):
                        model_to_save = model._orig_mod
                    else:
                        model_to_save = model

                    try:
                        model_cpu = model_to_save.to('cpu')
                        model_cpu.eval()
                        dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                        with torch.no_grad():
                            traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                        torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                        model_to_save.to(Config.DEVICE)
                        model_to_save.train()
                    except:
                        scripted_model = torch.jit.script(model_to_save)
                        torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
                except Exception as e:
                    if getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                        print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
                    
                print(f"  --> Saved new best rollout model!")
                print(f"\033[94m{save_path}\033[0m")
            
            wandb.run.summary["best_rollout_mse"] = best_rollout_loss
            wandb.run.summary["best_improvement"] = best_improvement
        else:
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}: Train L2={train_loss:.6f} (Validation skipped)")

        # --- Early exit for search ---
        if time.time() - start_time > Config.MAX_RUNTIME_PER_CANDIDATE:
            print(f"60s limit reached for {run_name}. Moving to next candidate.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            save_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_l2': val_loss,
                'rollout_mse': best_rollout_loss,
                'improvement': best_improvement,
                'config': {k: getattr(Config, k) for k in dir(Config) if not k.startswith('_') and not callable(getattr(Config, k))}
            }, save_path)
            print(f"  --> Saved new best validation model (L2={val_loss:.6f})")
            print(f"\033[94m{save_path}\033[0m")
            
            # Robust checkpointing: save scripted/traced model.
            # GATED off by default: this stalls the GPU. Enable via Config.SAVE_SCRIPTED_MODELS.
            try:
                if not getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                    raise RuntimeError('scripted-save disabled by config')
                if hasattr(model, "_orig_mod"):
                    model_to_save = model._orig_mod
                else:
                    model_to_save = model

                try:
                    model_cpu = model_to_save.to('cpu')
                    model_cpu.eval()
                    dummy_input = torch.zeros((1, Config.SEQ_LEN - 1, Config.INPUT_DIM))
                    with torch.no_grad():
                        traced_model = torch.jit.trace(model_cpu, dummy_input, check_trace=False)
                    torch.jit.save(traced_model, save_path.replace(".pt", "_scripted.pt"))
                    model_to_save.to(Config.DEVICE)
                    model_to_save.train()
                except:
                    scripted_model = torch.jit.script(model_to_save)
                    torch.jit.save(scripted_model, save_path.replace(".pt", "_scripted.pt"))
            except Exception as e:
                if getattr(Config, 'SAVE_SCRIPTED_MODELS', False):
                    print(f"  Warning: Could not save robust (scripted/traced) model: {e}")
            
    wandb.finish()
    return {"val_l2": best_val_loss, "rollout_l2": best_rollout_loss, "improvement": best_improvement}

if __name__ == "__main__":
    results = []
    # TOP 3 CANDIDATES: 0, 1, 5
    TOP_CANDIDATES = [0, 1, 5]
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "search":
            print(f"Starting architecture search with {Config.MAX_RUNTIME_PER_CANDIDATE}s per candidate...")
            for i in range(len(Config.SEARCH_SPACE)):
                print(f"\n--- Testing Candidate {i} ---")
                res = train(i)
                results.append((i, res))
        elif sys.argv[1] == "top3":
            print(f"Running TOP 3 candidates for {Config.MAX_RUNTIME_PER_CANDIDATE}s each...")
            for i in TOP_CANDIDATES:
                print(f"\n--- Testing Top Candidate {i} ---")
                res = train(i)
                results.append((i, res))
        elif sys.argv[1].isdigit():
            res = train(int(sys.argv[1]))
            results.append((int(sys.argv[1]), res))
        else:
            Config.VARIANT = sys.argv[1]
            res = train()
            results.append((Config.VARIANT, res))
    else:
        # Default to searching through all candidates
        print("Starting architecture search with 60s per candidate...")
        for i in range(len(Config.SEARCH_SPACE)):
            print(f"\n--- Testing Candidate {i} ---")
            res = train(i)
            results.append((i, res))

    if results:
        print("\n" + "="*50)
        print("ARCHITECTURE SEARCH LEADERBOARD")
        print("="*50)
        print(f"{'ID':<5} | {'Val L2':<10} | {'Rollout MSE':<12} | {'Improv %':<10}")
        print("-" * 50)
        for cand_id, res in results:
            if res:
                print(f"{str(cand_id):<5} | {res['val_l2']:<10.6f} | {res['rollout_l2']:<12.6f} | {res['improvement']:<10.2f}%")
        print("="*50)
