import os
import sys
import torch
import h5py
import numpy as np
import pandas as pd
from contextlib import nullcontext
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import json

# Add project root to sys.path to allow imports from sibling modules when run as a script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import model definitions
try:
    from transformer_model_v1 import OrderedTransformerV1
except ImportError:
    from transformer.transformer_model_v1 import OrderedTransformerV1

try:
    from helpers.TransformLatent import FloatConverter
except ImportError:
    from helpers.TransformLatent import FloatConverter

from transformer_config import add_config_arg, load_config, optional_int, resolve_path, str_to_bool

# ANSI Colors for Rainbow effect and highlighting
class Colors:
    CSI = "\033["
    RED = f"{CSI}91m"
    GREEN = f"{CSI}92m"
    YELLOW = f"{CSI}93m"
    BLUE = f"{CSI}94m"
    MAGENTA = f"{CSI}95m"
    CYAN = f"{CSI}96m"
    BOLD = f"{CSI}1m"
    RESET = f"{CSI}0m"
    
    @staticmethod
    def rainbow(text):
        """Create rainbow effect for text"""
        colors = [Colors.RED, Colors.YELLOW, Colors.GREEN, Colors.CYAN, Colors.BLUE, Colors.MAGENTA]
        result = []
        k = 0
        for char in text:
            if char.strip():
                result.append(f"{colors[k % len(colors)]}{char}")
                k += 1
            else:
                result.append(char)
        return ''.join(result) + Colors.RESET

# --- Configuration ---
class Config:
    # Model checkpoints
    TRANSFORMER_CHECKPOINT = "best_ordered_transformer_v1.pt"
    ENCODER_CHECKPOINT = "Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"

    # Data path
    EVAL_H5 = None
    VAL_H5 = None

    @staticmethod
    def get_data_path():
        if os.path.exists(Config.EVAL_H5):
            return Config.EVAL_H5
        return Config.VAL_H5
    
    # Device
    DEVICE = "auto"
    
    # Dimensions
    LATENT_DIM = 47
    NUM_X = 26
    NUM_TIME = 80
    SEQ_LEN = NUM_X * NUM_TIME # 2080
    INPUT_DIM = 52
    
    # For reporting
    TRIPLET_IDX = 62 # 63rd triplet (0-indexed 62)
    
    # Batch size
    BATCH_SIZE = 8
    
    # Micro-batching for evaluation loops
    # If BATCH_SIZE > 1, some operations might still OOM.
    # We can further process the batch in smaller chunks.
    MICRO_BATCH_SIZE = 4
    NUM_WORKERS = 2
    PREFETCH_FACTOR = 2
    
    # Fast evaluation
    LIMIT_SAMPLES = None
    
    # Staircase settings
    STAIRCASE_CONTEXTS = [1, 10, 20, 40, 60, 79]
    
    # Interleave Evaluation Settings
    # 1. Predict each even frame given the odd frame (T1->T2, T1-3->T4, ...)
    # 2. Predict every 2nd and 3rd only given 1 (C=1, P=2)
    # 3. Predict every 2nd, 3rd, 4th, 5th given 1 (C=1, P=4)
    # 4. Predict P=1 given C=2, P=2 given C=2, ...
    # 5. Collapse limit: RMSE > 0.05
    RMSE_LIMIT = 0.05
    
    # Runtime/report toggles
    RUN_CPU_PARALLEL = False
    ENABLE_METRICS = True
    ENABLE_STAIRCASE_EVAL = False
    ENABLE_INTERLEAVE_EVAL = False
    RESULTS_JSON = "evaluation_results.json"
    PRED_GT_PICKLE_PATH = "evaluation_pred_gt.pkl"
    # Number of final time steps to export for pointwise GT/Prediction.
    # <= 0 exports all predictable time steps (full autoregressive window).
    PRED_GT_EXPORT_TIME_STEPS = 0
    # Max flattened latent rows per AE decode call for export path.
    PRED_GT_DECODE_CHUNK = 4096

    @staticmethod
    def maybe_autocast(device):
        if device == "cuda":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            dtype = torch.bfloat16 if bf16_supported else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

def select_device(requested="auto"):
    requested = (requested or "auto").lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print(f"{Colors.YELLOW}CUDA was requested but is not available. Falling back to MPS/CPU.{Colors.RESET}")
        return "mps" if mps_available else "cpu"
    if requested == "mps" and not mps_available:
        print(f"{Colors.YELLOW}MPS was requested but is not available. Falling back to CPU.{Colors.RESET}")
        return "cpu"
    return requested

def refresh_derived_config():
    Config.SEQ_LEN = Config.NUM_X * Config.NUM_TIME

def configure(args):
    cfg = load_config(args.config)
    paths = cfg["paths"]
    data = cfg["data"]
    eval_cfg = cfg["evaluation"]

    Config.TRANSFORMER_CHECKPOINT = resolve_path(args.transformer_checkpoint or paths["transformer_checkpoint"])
    Config.ENCODER_CHECKPOINT = resolve_path(args.encoder_checkpoint or paths["encoder_checkpoint"])
    Config.EVAL_H5 = resolve_path(args.eval_h5 or paths["evaluation_h5"])
    Config.VAL_H5 = resolve_path(args.val_h5 or paths["validation_h5"])
    Config.RESULTS_JSON = resolve_path(args.results_json or paths["evaluation_results_json"])
    Config.PRED_GT_PICKLE_PATH = resolve_path(args.pred_gt_pickle or paths["pred_gt_pickle"])

    Config.LATENT_DIM = data.get("latent_dim", Config.LATENT_DIM)
    Config.NUM_X = data.get("num_x", Config.NUM_X)
    Config.NUM_TIME = args.num_time if args.num_time is not None else data.get("num_time", Config.NUM_TIME)
    Config.INPUT_DIM = data.get("input_dim", Config.INPUT_DIM)
    Config.BATCH_SIZE = args.batch_size if args.batch_size is not None else eval_cfg["batch_size"]
    Config.MICRO_BATCH_SIZE = args.micro_batch_size if args.micro_batch_size is not None else eval_cfg["micro_batch_size"]
    Config.NUM_WORKERS = args.num_workers if args.num_workers is not None else eval_cfg["num_workers"]
    Config.PREFETCH_FACTOR = args.prefetch_factor if args.prefetch_factor is not None else eval_cfg["prefetch_factor"]
    Config.LIMIT_SAMPLES = optional_int(args.limit_samples) if args.limit_samples is not None else optional_int(eval_cfg.get("limit_samples"))
    Config.RUN_CPU_PARALLEL = str_to_bool(args.run_cpu_parallel) if args.run_cpu_parallel is not None else eval_cfg["run_cpu_parallel"]
    Config.ENABLE_METRICS = str_to_bool(args.enable_metrics) if args.enable_metrics is not None else eval_cfg["enable_metrics"]
    Config.ENABLE_STAIRCASE_EVAL = str_to_bool(args.enable_staircase) if args.enable_staircase is not None else eval_cfg["enable_staircase"]
    Config.ENABLE_INTERLEAVE_EVAL = str_to_bool(args.enable_interleave) if args.enable_interleave is not None else eval_cfg["enable_interleave"]
    Config.PRED_GT_EXPORT_TIME_STEPS = (
        args.pred_gt_export_time_steps
        if args.pred_gt_export_time_steps is not None
        else eval_cfg["pred_gt_export_time_steps"]
    )
    Config.PRED_GT_DECODE_CHUNK = (
        args.pred_gt_decode_chunk
        if args.pred_gt_decode_chunk is not None
        else eval_cfg["pred_gt_decode_chunk"]
    )
    Config.TRIPLET_IDX = args.triplet_idx if args.triplet_idx is not None else eval_cfg["triplet_idx"]
    Config.RMSE_LIMIT = args.rmse_limit if args.rmse_limit is not None else eval_cfg["rmse_limit"]
    Config.DEVICE = select_device(args.device or eval_cfg.get("device", "auto"))
    Config.MICRO_BATCH_SIZE = max(1, min(Config.MICRO_BATCH_SIZE, Config.BATCH_SIZE))
    refresh_derived_config()

# --- Dataset ---
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, max_samples=None):
        self.h5_path = h5_path
        self._file = None
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            data_ds = f['data']
            total_available = data_ds.shape[0]
            self.has_originals = 'originals' in f
            self.has_start_time = 'start_time' in f
            self.has_start_t = 'start_t' in f
            self.sample_shape = data_ds.shape[1:]

            # Support both (N, T, X, F) and pre-flattened (N, T*X, F) layouts.
            if len(self.sample_shape) == 3:
                t_dim, x_dim, f_dim = self.sample_shape
                if x_dim != Config.NUM_X or f_dim != Config.INPUT_DIM:
                    raise ValueError(
                        f"Unexpected sample shape {self.sample_shape}. "
                        f"Expected (*, {Config.NUM_X}, {Config.INPUT_DIM})."
                    )
                self.num_time = t_dim
                self.seq_len = t_dim * x_dim
            elif len(self.sample_shape) == 2:
                seq_dim, f_dim = self.sample_shape
                if f_dim != Config.INPUT_DIM:
                    raise ValueError(
                        f"Unexpected sample shape {self.sample_shape}. "
                        f"Expected (*, {Config.INPUT_DIM}) for flattened data."
                    )
                if seq_dim % Config.NUM_X != 0:
                    raise ValueError(
                        f"Flattened sequence length {seq_dim} is not divisible by NUM_X={Config.NUM_X}."
                    )
                self.seq_len = seq_dim
                self.num_time = seq_dim // Config.NUM_X
            else:
                raise ValueError(
                    f"Unsupported sample layout {self.sample_shape}. "
                    "Expected 3D sample (T, X, F) or 2D sample (T*X, F)."
                )
            
            if max_samples is not None:
                self.length = min(max_samples, total_available)
                # Randomly pick indices from the whole dataset
                self.indices = np.random.choice(total_available, self.length, replace=False)
                # Sort indices to improve HDF5 access performance
                self.indices.sort()
            else:
                self.length = total_available
                self.indices = np.arange(total_available)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        
        # Map the requested idx to our random index
        actual_idx = self.indices[idx]
        data = self._file['data'][actual_idx] # (NUM_TIME, NUM_X, INPUT_DIM) or flattened
        # Flatten time and space: (2080, 52)
        data = data.reshape(self.seq_len, Config.INPUT_DIM)

        if self.has_start_time:
            start_time = float(self._file['start_time'][actual_idx])
        elif self.has_start_t:
            start_time = float(self._file['start_t'][actual_idx])
        else:
            start_time = 0.0
        start_time_tensor = torch.tensor(start_time, dtype=torch.float32)
        
        if self.has_originals:
            orig = self._file['originals'][actual_idx] # (26, 3)
            return torch.from_numpy(data).float(), torch.from_numpy(orig).float(), start_time_tensor
            
        return torch.from_numpy(data).float(), torch.zeros((Config.NUM_X, 3)), start_time_tensor

def load_models(device=None):
    if device is None:
        device = Config.DEVICE
        
    # Patch for torch._dynamo compatibility issues (e.g., missing ConvertFrameBox)
    try:
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class DummyConvertFrameBox:
                def __setstate__(self, state):
                    self.__dict__.update(state)
            torch._dynamo.convert_frame.ConvertFrameBox = DummyConvertFrameBox
    except (ImportError, AttributeError):
        pass

    # 1. Load Transformer
    print(f"Loading Transformer to {Colors.MAGENTA}{device}{Colors.RESET} from: {Colors.CYAN}{Config.TRANSFORMER_CHECKPOINT}{Colors.RESET}")
    checkpoint = torch.load(Config.TRANSFORMER_CHECKPOINT, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # Use the embedded model object for maximum compatibility
        transformer = checkpoint['model']
        
        # If it's a compiled model (OptimizedModule), get the original model
        if hasattr(transformer, '_orig_mod'):
            print(f"Detected compiled model for {device}, extracting original module...")
            transformer = transformer._orig_mod
        elif hasattr(transformer, 'module'):
            # In some cases it might be wrapped in DataParallel/DistributedDataParallel
            transformer = transformer.module
    else:
        # Reconstruct if necessary (using config in checkpoint)
        print(f"Reconstructing Transformer model for {device} from checkpoint config...")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        
    transformer.to(device)
    transformer.eval()
    
    # 2. Load Encoder/Decoder (TorchScript "one file" approach)
    print(f"Loading Scripted AE to {Colors.MAGENTA}{device}{Colors.RESET} from: {Colors.CYAN}{Config.ENCODER_CHECKPOINT}{Colors.RESET}")
    ae = torch.jit.load(Config.ENCODER_CHECKPOINT, map_location=device)
    ae.to(device)
    ae.eval()
    
    return transformer, ae

def evaluate_permutation(transformer, ae, batch, num_context_t, num_predict_t, scale_t, shift_t, triplet_idx=62, device='cpu'):
    """
    Evaluates a single permutation: (Context Time Steps, Prediction Time Steps).
    Returns the average RMSE across the prediction window.
    Processes in micro-batches to avoid MPS OOM.
    """
    B_full = batch.shape[0]
    num_x = 26
    latent_dim = 47
    
    context_len = num_context_t * num_x
    predict_len = num_predict_t * num_x
    total_len = context_len + predict_len
    
    if total_len > batch.shape[1]:
        return None
    
    # Slice for the prediction window (time steps starting from context_len)
    pred_slice = slice(context_len, total_len)
    
    all_rmse = []
    
    # Process each sample in the batch individually (Micro-batching)
    for i in range(0, B_full, Config.MICRO_BATCH_SIZE):
        micro_batch = batch[i:i+Config.MICRO_BATCH_SIZE]
        B = micro_batch.shape[0]
        
        # Autoregressive prediction
        current_seq = micro_batch[:, :context_len, :].clone()
        
        for step in range(context_len, total_len):
            with Config.maybe_autocast(device):
                step_out = transformer(current_seq)
            next_latent = step_out[:, -1, :] # (B, 47)
            
            # Prepare next token using metadata from 'micro_batch'
            new_token = micro_batch[:, step:step+1, :].clone()
            new_token[:, 0, :latent_dim] = next_latent
            current_seq = torch.cat([current_seq, new_token], dim=1)
            
        # Extract predicted and ground truth latents for the prediction window
        pred_latents = current_seq[:, pred_slice, :latent_dim]
        gt_latents = micro_batch[:, pred_slice, :latent_dim]
        
        # Decode and denormalize
        pred_latents_flat = pred_latents.reshape(-1, latent_dim)
        gt_latents_flat = gt_latents.reshape(-1, latent_dim)
        
        with Config.maybe_autocast(device):
            pred_dec_v = ae.decode(pred_latents_flat) # (B*pred_len, 375)
            gt_dec_v = ae.decode(gt_latents_flat) # (B*pred_len, 375)
        
        # Extract 63rd triplet
        pred_v_63 = pred_dec_v.reshape(B, num_predict_t, num_x, 125, 3)[:, :, :, triplet_idx, :]
        gt_v_63 = gt_dec_v.reshape(B, num_predict_t, num_x, 125, 3)[:, :, :, triplet_idx, :]
        
        # Denormalize on device to avoid CPU synchronization in inner loops.
        pred_denorm = (pred_v_63 - shift_t) / scale_t
        gt_denorm = (gt_v_63 - shift_t) / scale_t
        
        # Calculate RMSE
        sq_err = torch.sum((gt_denorm - pred_denorm) ** 2, dim=-1) # (B, P, 26)
        rmse_per_sample = torch.sqrt(torch.mean(sq_err, dim=(1, 2))) # (B,)
        all_rmse.extend(rmse_per_sample.detach().float().cpu().tolist())
        
        # Explicit memory cleanup
        if device == "mps":
            torch.mps.empty_cache()
    
    return float(np.mean(all_rmse))

def evaluate_on_device(device, indices, data_path, has_originals):
    """Run evaluation on a specific device using a subset of data indices."""
    print(f"{Colors.BOLD}Starting evaluation on {Colors.CYAN}{device}{Colors.RESET} for {len(indices)} samples")
    
    try:
        transformer, ae = load_models(device)
        converter = FloatConverter()
        # Keep native converter shapes (scalar or length-3) and rely on broadcasting.
        scale_t = torch.as_tensor(converter.scale, device=device, dtype=torch.float32)
        shift_t = torch.as_tensor(converter.shift, device=device, dtype=torch.float32)
    except Exception as e:
        print(f"{Colors.RED}Error loading models on {device}: {e}{Colors.RESET}")
        return None

    # Custom subset loader
    class IndexedEvalDataset(EvalDataset):
        def __init__(self, h5_path, indices):
            super().__init__(h5_path)
            self.indices = indices
            self.length = len(indices)

    dataset = IndexedEvalDataset(data_path, indices)
    print(
        f"{device}: inferred sample layout shape={dataset.sample_shape}, num_time={dataset.num_time}, "
        f"seq_len={dataset.seq_len}, has_start_time={dataset.has_start_time}, has_start_t={dataset.has_start_t}"
    )
    if dataset.num_time < 2:
        print(f"{Colors.RED}Need at least 2 time steps for autoregressive evaluation on {device}; got {dataset.num_time}.{Colors.RESET}")
        return None

    loader_kwargs = {"batch_size": Config.BATCH_SIZE, "shuffle": False}
    if device == "cuda":
        loader_kwargs.update({
            "num_workers": Config.NUM_WORKERS,
            "pin_memory": True,
        })
        if Config.NUM_WORKERS > 0:
            loader_kwargs.update({
                "persistent_workers": True,
                "prefetch_factor": Config.PREFETCH_FACTOR,
            })
    elif device == "cpu" and Config.NUM_WORKERS > 0:
        loader_kwargs.update({
            "num_workers": Config.NUM_WORKERS,
            "persistent_workers": True,
            "prefetch_factor": Config.PREFETCH_FACTOR,
        })
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    staircase_data = []
    interleave_results = []
    sqerr_sum_per_pos = np.zeros(Config.NUM_X, dtype=np.float64)
    sqerr_count_per_pos = np.zeros(Config.NUM_X, dtype=np.int64)
    sqerr_sum_overall = 0.0
    sqerr_count_overall = 0
    sqerr_sum_l4 = 0.0
    sqerr_count_l4 = 0
    sqerr_sum_l8 = 0.0
    sqerr_count_l8 = 0
    sqerr_sum_l16 = 0.0
    sqerr_count_l16 = 0
    # Dict[str(float)] -> [sum_sq_error, count]
    param_agg = {}
    # Dict[(y, z)] -> [sum_sq_error, count]
    yz_agg = {}
    pred_gt_data = {
        "x": [],
        "y": [],
        "z": [],
        "time": [],
        "vx_original": [],
        "vy_original": [],
        "vz_original": [],
        "vx_predicted": [],
        "vy_predicted": [],
        "vz_predicted": [],
    }
    decode_chunk = max(1, Config.PRED_GT_DECODE_CHUNK)

    def decode_triplet_velocities(latents_bt):
        """Decode latent tokens and keep only the configured velocity triplet."""
        bsz, n_tokens, _ = latents_bt.shape
        latents_flat = latents_bt.reshape(-1, Config.LATENT_DIM)
        triplet_chunks = []

        for start in range(0, latents_flat.shape[0], decode_chunk):
            chunk = latents_flat[start:start + decode_chunk]
            with Config.maybe_autocast(device):
                decoded = ae.decode(chunk)
            triplet_chunks.append(decoded.reshape(-1, 125, 3)[:, Config.TRIPLET_IDX, :])
            if device == "mps":
                torch.mps.empty_cache()

        return torch.cat(triplet_chunks, dim=0).reshape(bsz, n_tokens, 3)

    total_samples_processed = 0

    with torch.inference_mode():
        for batch_idx, (batch, originals_batch, start_time_batch) in enumerate(tqdm(loader, desc=f"Eval {device}")):
            non_blocking = device == "cuda"
            batch = batch.to(device, non_blocking=non_blocking)
            originals_batch = originals_batch.to(device, non_blocking=non_blocking)
            start_time_batch = start_time_batch.to(device, non_blocking=non_blocking)
            B = batch.shape[0]
            
            # 1. Standard Transformer Prediction
            outputs_list = []
            for i in range(0, B, Config.MICRO_BATCH_SIZE):
                micro_inputs = batch[i:i+Config.MICRO_BATCH_SIZE, :-1, :]
                with Config.maybe_autocast(device):
                    micro_outputs = transformer(micro_inputs)
                outputs_list.append(micro_outputs)
                if device == "mps":
                    torch.mps.empty_cache()
            
            outputs = torch.cat(outputs_list, dim=0)
            
            # 2. Select export window and extract Tlast predictions for metrics.
            seq_len = batch.shape[1]
            num_time = seq_len // Config.NUM_X
            max_export_steps = max(1, num_time - 1)
            if Config.PRED_GT_EXPORT_TIME_STEPS <= 0:
                export_time_steps = max_export_steps
            else:
                export_time_steps = min(max_export_steps, max(1, Config.PRED_GT_EXPORT_TIME_STEPS))
            export_start_idx = seq_len - (export_time_steps * Config.NUM_X)
            export_target_slice = slice(export_start_idx, seq_len)
            export_output_slice = slice(export_start_idx - 1, seq_len - 1)

            t80_start_idx = seq_len - Config.NUM_X
            t80_target_indices = range(t80_start_idx, seq_len)
            t80_output_indices = [i - 1 for i in t80_target_indices]
            
            pred_latents_t80 = outputs[:, t80_output_indices, :]
            gt_latents_t80 = batch[:, t80_target_indices, :Config.LATENT_DIM]
            
            coords_t80 = batch[:, t80_target_indices, 47:50]
            time_t80 = batch[:, t80_target_indices, 50]
            param_t80 = batch[:, t80_target_indices, 51] if Config.ENABLE_METRICS else None
            
            # 3. Decode Latents to Velocities
            pred_v_63_list = []
            decode_gt_t80 = not has_originals
            gt_v_63_list = [] if decode_gt_t80 else None
            
            for i in range(0, B, Config.MICRO_BATCH_SIZE):
                m_pred_latents = pred_latents_t80[i:i+Config.MICRO_BATCH_SIZE]
                mB = m_pred_latents.shape[0]
                
                m_pred_latents_flat = m_pred_latents.reshape(-1, Config.LATENT_DIM)
                
                with Config.maybe_autocast(device):
                    m_pred_velocities_full = ae.decode(m_pred_latents_flat)
                
                m_pred_v_63 = m_pred_velocities_full.reshape(mB, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
                
                pred_v_63_list.append(m_pred_v_63)
                if decode_gt_t80:
                    m_gt_latents = gt_latents_t80[i:i+Config.MICRO_BATCH_SIZE]
                    m_gt_latents_flat = m_gt_latents.reshape(-1, Config.LATENT_DIM)
                    with Config.maybe_autocast(device):
                        m_gt_velocities_full = ae.decode(m_gt_latents_flat)
                    m_gt_v_63 = m_gt_velocities_full.reshape(mB, 26, 125, 3)[:, :, Config.TRIPLET_IDX, :]
                    gt_v_63_list.append(m_gt_v_63)
                
                if device == "mps":
                    torch.mps.empty_cache()

            pred_v_63 = torch.cat(pred_v_63_list, dim=0)
            
            # 4. Denormalize on device
            pred_denorm_v = (pred_v_63 - shift_t) / scale_t
            
            if has_originals:
                gt_denorm_v = originals_batch
            else:
                gt_v_63 = torch.cat(gt_v_63_list, dim=0)
                gt_denorm_v = (gt_v_63 - shift_t) / scale_t
            
            # Prepare vectorized arrays for stats + pickle export once per batch.
            coords_np = coords_t80.detach().float().cpu().numpy()
            gt_denorm_np = gt_denorm_v.detach().float().cpu().numpy()
            pred_denorm_np = pred_denorm_v.detach().float().cpu().numpy()

            if Config.ENABLE_METRICS:
                sq_errors = torch.sum((gt_denorm_v - pred_denorm_v) ** 2, dim=2)
                sq_errors_np = sq_errors.detach().float().cpu().numpy()
                params = param_t80[:, 0].detach().float().cpu().numpy()
                y_np = coords_np[:, :, 1]
                z_np = coords_np[:, :, 2]
            if export_start_idx == t80_start_idx:
                # Fast path: export window is exactly Tlast.
                export_coords_np = coords_np
                export_rel_time_np = time_t80.detach().float().cpu().numpy()
                export_gt_denorm_np = gt_denorm_np
                export_pred_denorm_np = pred_denorm_np
            else:
                pred_latents_export = outputs[:, export_output_slice, :]
                gt_latents_export = batch[:, export_target_slice, :Config.LATENT_DIM]
                coords_export = batch[:, export_target_slice, 47:50]
                time_export = batch[:, export_target_slice, 50]

                pred_export_v = decode_triplet_velocities(pred_latents_export)
                gt_export_v = decode_triplet_velocities(gt_latents_export)

                pred_export_denorm = (pred_export_v - shift_t) / scale_t
                gt_export_denorm = (gt_export_v - shift_t) / scale_t

                export_coords_np = coords_export.detach().float().cpu().numpy()
                export_rel_time_np = time_export.detach().float().cpu().numpy()
                export_gt_denorm_np = gt_export_denorm.detach().float().cpu().numpy()
                export_pred_denorm_np = pred_export_denorm.detach().float().cpu().numpy()

            # Source absolute time = sample start_time + relative time feature.
            sample_start_time_np = start_time_batch.detach().float().cpu().numpy().reshape(B, 1)
            export_time_np = export_rel_time_np + sample_start_time_np

            pred_gt_data["x"].append(export_coords_np[:, :, 0].reshape(-1))
            pred_gt_data["y"].append(export_coords_np[:, :, 1].reshape(-1))
            pred_gt_data["z"].append(export_coords_np[:, :, 2].reshape(-1))
            pred_gt_data["time"].append(export_time_np.reshape(-1))
            pred_gt_data["vx_original"].append(export_gt_denorm_np[:, :, 0].reshape(-1))
            pred_gt_data["vy_original"].append(export_gt_denorm_np[:, :, 1].reshape(-1))
            pred_gt_data["vz_original"].append(export_gt_denorm_np[:, :, 2].reshape(-1))
            pred_gt_data["vx_predicted"].append(export_pred_denorm_np[:, :, 0].reshape(-1))
            pred_gt_data["vy_predicted"].append(export_pred_denorm_np[:, :, 1].reshape(-1))
            pred_gt_data["vz_predicted"].append(export_pred_denorm_np[:, :, 2].reshape(-1))
            
            if Config.ENABLE_METRICS:
                # Vectorized online accumulation avoids building millions of Python dict objects.
                sqerr_sum_per_pos += sq_errors_np.sum(axis=0, dtype=np.float64)
                sqerr_count_per_pos += B
                sqerr_sum_overall += float(sq_errors_np.sum(dtype=np.float64))
                sqerr_count_overall += B * Config.NUM_X

                l4_start = max(0, Config.NUM_X - 4)
                l8_start = max(0, Config.NUM_X - 8)
                l16_start = max(0, Config.NUM_X - 16)
                sqerr_sum_l4 += float(sq_errors_np[:, l4_start:].sum(dtype=np.float64))
                sqerr_count_l4 += B * (Config.NUM_X - l4_start)
                sqerr_sum_l8 += float(sq_errors_np[:, l8_start:].sum(dtype=np.float64))
                sqerr_count_l8 += B * (Config.NUM_X - l8_start)
                sqerr_sum_l16 += float(sq_errors_np[:, l16_start:].sum(dtype=np.float64))
                sqerr_count_l16 += B * (Config.NUM_X - l16_start)

                # Parameter aggregation: group by parameter value.
                sample_sq_sums = sq_errors_np.sum(axis=1, dtype=np.float64)
                unique_params, inv_idx = np.unique(params.astype(np.float32), return_inverse=True)
                grouped_sq = np.bincount(inv_idx, weights=sample_sq_sums)
                grouped_n = np.bincount(inv_idx) * Config.NUM_X
                for p_val, sum_sq, cnt in zip(unique_params.tolist(), grouped_sq.tolist(), grouped_n.tolist()):
                    key = float(p_val)
                    if key in param_agg:
                        param_agg[key][0] += sum_sq
                        param_agg[key][1] += int(cnt)
                    else:
                        param_agg[key] = [sum_sq, int(cnt)]

                # Y/Z aggregation: group by position-wise coordinates.
                for j in range(Config.NUM_X):
                    yz_pairs = np.stack([y_np[:, j], z_np[:, j]], axis=1).astype(np.float32, copy=False)
                    unique_yz, yz_inv = np.unique(yz_pairs, axis=0, return_inverse=True)
                    yz_sq = np.bincount(yz_inv, weights=sq_errors_np[:, j].astype(np.float64, copy=False))
                    yz_n = np.bincount(yz_inv)
                    for idx_u, (y_val, z_val) in enumerate(unique_yz.tolist()):
                        key = (float(y_val), float(z_val))
                        if key in yz_agg:
                            yz_agg[key][0] += yz_sq[idx_u]
                            yz_agg[key][1] += int(yz_n[idx_u])
                        else:
                            yz_agg[key] = [yz_sq[idx_u], int(yz_n[idx_u])]

            # --- Staircase Evaluation ---
            if Config.ENABLE_METRICS and Config.ENABLE_STAIRCASE_EVAL:
                staircase_contexts = [c for c in Config.STAIRCASE_CONTEXTS if 1 <= c < num_time]
                if not staircase_contexts:
                    staircase_contexts = [max(1, num_time - 1)]
                for c in staircase_contexts:
                    p = num_time - c
                    rmse = evaluate_permutation(transformer, ae, batch, c, p, scale_t, shift_t, device=device)
                    staircase_data.append({'context_time_steps': c, 'rmse': rmse})

            # --- New Permutations Evaluation ---
            if Config.ENABLE_METRICS and Config.ENABLE_INTERLEAVE_EVAL:
                for n in range(1, (num_time // 2) + 1):
                    c = 2*n - 1
                    p = 1
                    if c + p > num_time: break
                    rmse = evaluate_permutation(transformer, ae, batch, c, p, scale_t, shift_t, device=device)
                    interleave_results.append({'mode': 'interleave', 'c': c, 'p': p, 'rmse': rmse})
                    if rmse > Config.RMSE_LIMIT: break

                p_jump = 2
                while p_jump < num_time:
                    rmse = evaluate_permutation(transformer, ae, batch, 1, p_jump, scale_t, shift_t, device=device)
                    interleave_results.append({'mode': 'jump_c1', 'c': 1, 'p': p_jump, 'rmse': rmse})
                    if rmse > Config.RMSE_LIMIT: break
                    p_jump *= 2

                for c_var in [2, 5, 10, 20]:
                    for p_var in [1, 2, 5, 10, 20]:
                        if c_var + p_var > num_time: break
                        rmse = evaluate_permutation(transformer, ae, batch, c_var, p_var, scale_t, shift_t, device=device)
                        interleave_results.append({'mode': f'var_c{c_var}', 'c': c_var, 'p': p_var, 'rmse': rmse})
                        if rmse > Config.RMSE_LIMIT: break
            
            total_samples_processed += B

    return {
        'staircase_data': staircase_data,
        'interleave_results': interleave_results,
        'sqerr_sum_per_pos': sqerr_sum_per_pos,
        'sqerr_count_per_pos': sqerr_count_per_pos,
        'sqerr_sum_overall': sqerr_sum_overall,
        'sqerr_count_overall': sqerr_count_overall,
        'sqerr_sum_l4': sqerr_sum_l4,
        'sqerr_count_l4': sqerr_count_l4,
        'sqerr_sum_l8': sqerr_sum_l8,
        'sqerr_count_l8': sqerr_count_l8,
        'sqerr_sum_l16': sqerr_sum_l16,
        'sqerr_count_l16': sqerr_count_l16,
        'param_agg': param_agg,
        'yz_agg': yz_agg,
        'pred_gt_data': pred_gt_data,
        'total_samples_processed': total_samples_processed
    }

def main():
    # Big Rainbow Message
    msg = f"INTERLEAVED EVALUATION: {Config.DEVICE.upper()}"
    print("\n" + "="*80)
    print(Colors.rainbow(f"  {msg}  "))
    print("="*80 + "\n")

    if Config.DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print(f"{Colors.GREEN}CUDA optimization enabled (TF32 + cuDNN benchmark + autocast).{Colors.RESET}")
    
    # Load dataset to get indices
    try:
        data_path = Config.get_data_path()
        print(f"Using dataset: {Colors.YELLOW}{data_path}{Colors.RESET}")
        
        # Initial access to get total available and check for originals
        with h5py.File(data_path, 'r') as f:
            total_available = f['data'].shape[0]
            has_originals = 'originals' in f
            
        limit = total_available if Config.LIMIT_SAMPLES is None else min(Config.LIMIT_SAMPLES, total_available)
        all_indices = np.random.choice(total_available, limit, replace=False)
        all_indices.sort()
        
        run_cpu_parallel = Config.RUN_CPU_PARALLEL and Config.DEVICE == "cuda"
        if run_cpu_parallel:
            mid = len(all_indices) // 2
            gpu_indices = all_indices[:mid]
            cpu_indices = all_indices[mid:]
            print(f"Total samples: {len(all_indices)} (GPU: {len(gpu_indices)}, CPU: {len(cpu_indices)})")
        else:
            gpu_indices = all_indices if Config.DEVICE != "cpu" else np.array([], dtype=np.int64)
            cpu_indices = all_indices if Config.DEVICE == "cpu" else np.array([], dtype=np.int64)
            active = "CPU" if Config.DEVICE == "cpu" else Config.DEVICE.upper()
            print(f"Total samples: {len(all_indices)} ({active}: {len(all_indices)})")
    except Exception as e:
        print(f"{Colors.RED}Error preparing dataset: {e}{Colors.RESET}")
        return

    # Run evaluation
    results_list = []
    if Config.RUN_CPU_PARALLEL and Config.DEVICE == "cuda" and len(cpu_indices) > 0:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            if len(gpu_indices) > 0:
                futures.append(executor.submit(evaluate_on_device, Config.DEVICE, gpu_indices, data_path, has_originals))
            if len(cpu_indices) > 0:
                futures.append(executor.submit(evaluate_on_device, 'cpu', cpu_indices, data_path, has_originals))
            for future in futures:
                res = future.result()
                if res:
                    results_list.append(res)
    else:
        target_device = "cpu" if Config.DEVICE == "cpu" else Config.DEVICE
        target_indices = cpu_indices if target_device == "cpu" else gpu_indices
        if len(target_indices) > 0:
            res = evaluate_on_device(target_device, target_indices, data_path, has_originals)
            if res:
                results_list.append(res)

    # Merge results
    staircase_data = []
    interleave_results = []
    sqerr_sum_per_pos = np.zeros(Config.NUM_X, dtype=np.float64)
    sqerr_count_per_pos = np.zeros(Config.NUM_X, dtype=np.int64)
    sqerr_sum_overall = 0.0
    sqerr_count_overall = 0
    sqerr_sum_l4 = 0.0
    sqerr_count_l4 = 0
    sqerr_sum_l8 = 0.0
    sqerr_count_l8 = 0
    sqerr_sum_l16 = 0.0
    sqerr_count_l16 = 0
    param_agg = {}
    yz_agg = {}
    pred_gt_data = {
        "x": [],
        "y": [],
        "z": [],
        "time": [],
        "vx_original": [],
        "vy_original": [],
        "vz_original": [],
        "vx_predicted": [],
        "vy_predicted": [],
        "vz_predicted": [],
    }
    total_samples_processed = 0
    
    for res in results_list:
        staircase_data.extend(res['staircase_data'])
        interleave_results.extend(res['interleave_results'])
        sqerr_sum_per_pos += res['sqerr_sum_per_pos']
        sqerr_count_per_pos += res['sqerr_count_per_pos']
        sqerr_sum_overall += res['sqerr_sum_overall']
        sqerr_count_overall += res['sqerr_count_overall']
        sqerr_sum_l4 += res['sqerr_sum_l4']
        sqerr_count_l4 += res['sqerr_count_l4']
        sqerr_sum_l8 += res['sqerr_sum_l8']
        sqerr_count_l8 += res['sqerr_count_l8']
        sqerr_sum_l16 += res['sqerr_sum_l16']
        sqerr_count_l16 += res['sqerr_count_l16']
        for p_key, (sum_sq, cnt) in res['param_agg'].items():
            if p_key in param_agg:
                param_agg[p_key][0] += sum_sq
                param_agg[p_key][1] += cnt
            else:
                param_agg[p_key] = [sum_sq, cnt]
        for yz_key, (sum_sq, cnt) in res['yz_agg'].items():
            if yz_key in yz_agg:
                yz_agg[yz_key][0] += sum_sq
                yz_agg[yz_key][1] += cnt
            else:
                yz_agg[yz_key] = [sum_sq, cnt]
        for key in pred_gt_data:
            pred_gt_data[key].extend(res['pred_gt_data'][key])
        total_samples_processed += res['total_samples_processed']

    if total_samples_processed == 0:
        print(f"{Colors.RED}No samples were processed.{Colors.RESET}")
        return

    # --- Statistics Calculation ---
    print(f"\n{Colors.BOLD}CALCULATING SUMMARY STATISTICS...{Colors.RESET}")
    if Config.ENABLE_METRICS:
        rmse_per_pos_values = np.sqrt(
            np.divide(
                sqerr_sum_per_pos,
                np.maximum(sqerr_count_per_pos, 1),
                dtype=np.float64,
            )
        )
        rmse_per_pos = pd.Series(rmse_per_pos_values, index=np.arange(Config.NUM_X))

        if param_agg:
            rmse_per_param = pd.Series(
                {
                    p_val: float(np.sqrt(sum_sq / max(cnt, 1)))
                    for p_val, (sum_sq, cnt) in param_agg.items()
                }
            ).sort_index()
        else:
            rmse_per_param = pd.Series(dtype=np.float64)

        if staircase_data:
            df_staircase = pd.DataFrame(staircase_data)
            rmse_staircase = df_staircase.groupby('context_time_steps')['rmse'].mean()
        else:
            rmse_staircase = pd.Series(dtype=np.float64)

        rmse_l4 = float(np.sqrt(sqerr_sum_l4 / max(sqerr_count_l4, 1)))
        rmse_l8 = float(np.sqrt(sqerr_sum_l8 / max(sqerr_count_l8, 1)))
        rmse_l16 = float(np.sqrt(sqerr_sum_l16 / max(sqerr_count_l16, 1)))
        rmse_overall = float(np.sqrt(sqerr_sum_overall / max(sqerr_count_overall, 1)))
        yz_stats = [
            {'y': y_val, 'z': z_val, 'rmse': float(np.sqrt(sum_sq / max(cnt, 1)))}
            for (y_val, z_val), (sum_sq, cnt) in yz_agg.items()
        ]
    else:
        rmse_per_pos = pd.Series(dtype=np.float64)
        rmse_per_param = pd.Series(dtype=np.float64)
        rmse_staircase = pd.Series(dtype=np.float64)
        rmse_l4 = float("nan")
        rmse_l8 = float("nan")
        rmse_l16 = float("nan")
        rmse_overall = float("nan")
        yz_stats = []
        print(f"{Colors.YELLOW}Metrics disabled (EVAL_ENABLE_METRICS=0): skipping RMSE/statistics aggregation.{Colors.RESET}")

    # --- Export Results ---
    print(f"\n{Colors.BOLD}EXPORTING RESULTS...{Colors.RESET}")

    if interleave_results:
        df_interleave = pd.DataFrame(interleave_results)
        summary_interleave = df_interleave.groupby(['mode', 'c', 'p'])['rmse'].mean().reset_index()
    else:
        summary_interleave = pd.DataFrame(columns=['mode', 'c', 'p', 'rmse'])

    results = {
        'rmse_per_pos': rmse_per_pos.to_dict(),
        'rmse_staircase': rmse_staircase.to_dict(),
        'rmse_per_param': rmse_per_param.to_dict(),
        'yz_stats': yz_stats,
        'rmse_l4': rmse_l4,
        'rmse_l8': rmse_l8,
        'rmse_l16': rmse_l16,
        'rmse_overall': rmse_overall,
        'interleave_summary': summary_interleave.to_dict(orient='records')
    }

    def default_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    os.makedirs(os.path.dirname(Config.RESULTS_JSON) or ".", exist_ok=True)
    with open(Config.RESULTS_JSON, 'w') as f:
        json.dump(results, f, default=default_converter)
    print(f"Results exported to: {Colors.CYAN}{Config.RESULTS_JSON}{Colors.RESET}")

    # Export point-wise ground-truth vs predicted values for downstream analysis.
    pred_gt_columns = [
        "x",
        "y",
        "z",
        "time",
        "vx_original",
        "vy_original",
        "vz_original",
        "vx_predicted",
        "vy_predicted",
        "vz_predicted",
    ]
    pred_gt_df = pd.DataFrame({
        key: np.concatenate(pred_gt_data[key]).astype(np.float32, copy=False)
        for key in pred_gt_columns
    })[pred_gt_columns]
    pred_gt_df = pred_gt_df.drop_duplicates()
    os.makedirs(os.path.dirname(Config.PRED_GT_PICKLE_PATH) or ".", exist_ok=True)
    pred_gt_df.to_pickle(Config.PRED_GT_PICKLE_PATH)
    time_min = float(pred_gt_df["time"].min())
    time_max = float(pred_gt_df["time"].max())
    print(
        f"Pointwise GT/Prediction exported to: {Colors.CYAN}{Config.PRED_GT_PICKLE_PATH}{Colors.RESET} "
        f"({len(pred_gt_df)} rows, time range: [{time_min:.1f}, {time_max:.1f}])"
    )

    # --- Final Report ---
    print(f"\n{Colors.BOLD}SUMMARY STATISTICS (Velocity Units){Colors.RESET}")
    print("-" * 50)
    if Config.ENABLE_METRICS:
        print(f"Overall RMSE:       {Colors.GREEN}{rmse_overall:.4e}{Colors.RESET}")
        print(f"RMSE (Last 4):      {Colors.YELLOW}{rmse_l4:.4e}{Colors.RESET}")
        print(f"RMSE (Last 8):      {Colors.YELLOW}{rmse_l8:.4e}{Colors.RESET}")
        print(f"RMSE (Last 16):     {Colors.YELLOW}{rmse_l16:.4e}{Colors.RESET}")
        print("-" * 50)
        print(f"STAIRCASE EVALUATION (Velocity Units):")
        for k, val in rmse_staircase.items():
            print(f"Given {k} time steps -> Tlast RMSE: {Colors.CYAN}{val:.4e}{Colors.RESET}")
        print("-" * 50)
        print(f"RMSE per Experiment (First 10 Params):\n{rmse_per_param.head(10).apply(lambda x: f'{x:.4e}')}")
    else:
        print("Metrics disabled.")
    print("-" * 50)

    print(f"{Colors.GREEN}Evaluation complete!{Colors.RESET}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the ordered transformer.")
    add_config_arg(parser)
    parser.add_argument("--transformer_checkpoint", "--transformer-checkpoint", dest="transformer_checkpoint", default=None)
    parser.add_argument("--encoder_checkpoint", "--encoder-checkpoint", dest="encoder_checkpoint", default=None)
    parser.add_argument("--eval_h5", "--eval-h5", dest="eval_h5", default=None)
    parser.add_argument("--val_h5", "--val-h5", dest="val_h5", default=None)
    parser.add_argument("--results_json", "--results-json", dest="results_json", default=None)
    parser.add_argument("--pred_gt_pickle", "--pred-gt-pickle", dest="pred_gt_pickle", default=None)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", "--micro-batch-size", dest="micro_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", "--prefetch-factor", dest="prefetch_factor", type=int, default=None)
    parser.add_argument("--limit_samples", "--limit-samples", dest="limit_samples", default=None, help="Limit samples. Use none/all/0 for full dataset.")
    parser.add_argument("--run_cpu_parallel", "--run-cpu-parallel", dest="run_cpu_parallel", default=None)
    parser.add_argument("--enable_metrics", "--enable-metrics", dest="enable_metrics", default=None)
    parser.add_argument("--enable_staircase", "--enable-staircase", dest="enable_staircase", default=None)
    parser.add_argument("--enable_interleave", "--enable-interleave", dest="enable_interleave", default=None)
    parser.add_argument("--pred_gt_export_time_steps", "--pred-gt-export-time-steps", dest="pred_gt_export_time_steps", type=int, default=None)
    parser.add_argument("--pred_gt_decode_chunk", "--pred-gt-decode-chunk", dest="pred_gt_decode_chunk", type=int, default=None)
    parser.add_argument("--triplet_idx", "--triplet-idx", dest="triplet_idx", type=int, default=None)
    parser.add_argument("--rmse_limit", "--rmse-limit", dest="rmse_limit", type=float, default=None)
    parser.add_argument("--num_time", "--num-time", dest="num_time", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    return parser.parse_args()

if __name__ == "__main__":
    configure(parse_args())
    main()
