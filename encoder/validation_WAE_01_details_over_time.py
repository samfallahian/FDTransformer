"""
Compute WAE residuals over time for vx, vy, vz and summarize by percentiles.

What it does
- Walk a directory of time-sliced pickle DataFrames (one file per second, named 1..N).
- Each DataFrame must contain columns: x, y, z, vx, vy, vz, with identical coordinate rows across time.
- For each time step and for each coordinate present in the FIRST file, build the 3×125 neighborhood vector
  using the same neighborhood definition used by production/encode_all_possible_input.py (3p6, etc.).
- Run the WAE (encode→decode) to reconstruct the 375-dim neighborhood vector.
- For each component separately (vx, vy, vz), compute RMSE across the 125 neighborhood values.
- Aggregate across all coordinates at that time and record the 25th/50th/75th percentiles for each component.
- Save cached CSV of the time series of percentiles and generate per-component plots over time.

Notes
- Uses tqdm progress bars for file processing.
- Prints a green diagnostic confirming we see 125 triplets per neighborhood (via adjacent_coordinate).
"""

import os
import sys
import re
import math
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# Resolve project root so we can import local modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from encoder.model_WAE_01 import WAE  # noqa: E402
from encoder.train_WAE_01_cached import _load_preferences  # noqa: E402
from production.adjacent_coordinate import adjacent_coordinate  # noqa: E402


# ------------------------------
# CONFIG DEFAULTS
# ------------------------------
MODEL_CHECKPOINT_PATH = \
    "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_01_epoch_2870.pt"
# Default meta-project configuration file (updated per request)
# Previously: configs/Umass_experiments.txt
# Now use the JSON metadata file instead
META_PROJECT_JSON = "/Users/kkreth/PycharmProjects/cgan/configs/Experiment_MetaData.json"
ABSOLUTE_RESIDUALS = False  # if True, RMSE is computed on absolute differences (no real effect for RMSE)
OUTPUT_DIR = os.path.join(PARENT_DIR, 'encoder', 'position_error_analysis')
BATCH_SIZE = 2048  # reserved for potential batching; we process coordinate-by-coordinate here


logger = logging.getLogger(__name__)
# Default all logging here to DEBUG as requested
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _load_model(checkpoint_path: str, device: torch.device) -> WAE:
    model = WAE().to(device)
    model.eval()
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    return model


def _discover_time_files(directory: str) -> List[Tuple[int, str]]:
    """Return list of (time_index, path) sorted by time_index.
    Accepts files whose basenames start with an integer (e.g., 1.pkl, 2.pkl, 003.parquet, 10).
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(directory)
    out = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        m = re.match(r"^(\d+)", os.path.splitext(name)[0])
        if m:
            t = int(m.group(1))
            out.append((t, path))
    out.sort(key=lambda x: x[0])
    if not out:
        raise FileNotFoundError(f"No time-indexed files found in {directory}")
    logger.debug(f"Discovered {len(out)} time files in {directory}. First 5: {[os.path.basename(p) for _, p in out[:5]]}")
    return out


def _read_frame(path: str) -> pd.DataFrame:
    """Backward-compatible simple reader by extension (legacy)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.pkl', '.pickle'):
        return pd.read_pickle(path)
    if ext in ('.parquet',):
        return pd.read_parquet(path)
    if ext in ('.feather',):
        return pd.read_feather(path)
    if ext in ('.csv',):
        return pd.read_csv(path)
    return pd.read_pickle(path)


def _detect_and_build_reader(path: str, assume_format: str | None = None):
    """Detect file format for a single representative file and return a callable reader.
    Shows attempts in the console and reuses the discovered reader for the rest.

    Supported assume_format values: auto/None, 'pickle', 'gzip_pickle', 'bz2_pickle', 'xz_pickle',
    'zip_pickle', 'zstd_pickle', 'parquet', 'feather', 'csv'.
    """
    import gzip
    import bz2
    import lzma
    import zipfile
    from io import BytesIO

    def _log(msg: str):
        logger.info(msg)

    def _reader_for(strategy: str):
        if strategy == 'pickle':
            return lambda p: pd.read_pickle(p)
        if strategy == 'gzip_pickle':
            return lambda p: pd.read_pickle(gzip.open(p, 'rb'))
        if strategy == 'bz2_pickle':
            return lambda p: pd.read_pickle(bz2.open(p, 'rb'))
        if strategy == 'xz_pickle':
            return lambda p: pd.read_pickle(lzma.open(p, 'rb'))
        if strategy == 'zip_pickle':
            # Attempt to open single-file zip and read inner as pickle
            def _zip_read(p):
                with zipfile.ZipFile(p, 'r') as zf:
                    names = zf.namelist()
                    if not names:
                        raise ValueError('Empty ZIP archive')
                    with zf.open(names[0], 'r') as fh:
                        data = fh.read()
                return pd.read_pickle(BytesIO(data))
            return _zip_read
        if strategy == 'zstd_pickle':
            # Try pandas built-in compression param if available
            return lambda p: pd.read_pickle(p, compression='zstd')
        if strategy == 'parquet':
            return lambda p: pd.read_parquet(p)
        if strategy == 'feather':
            return lambda p: pd.read_feather(p)
        if strategy == 'csv':
            return lambda p: pd.read_csv(p)
        raise ValueError(f'Unknown strategy: {strategy}')

    # If user forced a format, try that directly
    if assume_format and assume_format not in ('auto', 'AUTO', 'Auto', 'none', 'None'):
        strategy = assume_format.lower()
        _log(f"Assuming format '{strategy}' per CLI; using this reader without detection.")
        reader = _reader_for(strategy)
        # Validate once on the sample path to fail fast with clear message
        try:
            df = reader(path)
        except Exception as e:
            raise RuntimeError(f"Assumed format '{strategy}' failed to read file '{path}': {e}") from e
        _log(f"Assumed format '{strategy}' successfully read sample file.")
        return strategy, reader, df

    # Auto-detect: sniff header and then try a sequence of strategies; log tries
    try:
        with open(path, 'rb') as f:
            head = f.read(12)
    except Exception as e:
        raise RuntimeError(f"Failed to open '{path}' for detection: {e}") from e

    head_hex = head[:8].hex(' ')
    _log(f"Detecting format for '{os.path.basename(path)}' | header bytes: {head_hex}")

    # Magic-byte guided first guess
    guided: list[str] = []
    if len(head) >= 2 and head[0] == 0x1F and head[1] == 0x8B:
        guided.append('gzip_pickle')  # GZIP
    if head[:3] == b'BZh':
        guided.append('bz2_pickle')   # BZip2
    # XZ/LZMA (xz) magic bytes: FD 37 7A 58 5A 00
    if len(head) >= 6 and head.startswith(b"\xFD7zXZ\x00"):
        guided.append('xz_pickle')    # XZ
    if head[:4] == b'PK\x03\x04':
        guided.append('zip_pickle')   # ZIP
    if head[:4] == b'PAR1':
        guided.append('parquet')
    if head[:5] == b'ARROW' or head[:4] == b'FEA1':
        guided.append('feather')
    if head and head[0] == 0x80:
        guided.append('pickle')       # Pickle protocol header

    # Fallback generic order if no strong signal
    generic_order = ['pickle', 'gzip_pickle', 'bz2_pickle', 'xz_pickle', 'zip_pickle', 'zstd_pickle', 'parquet', 'feather', 'csv']
    tried = set()
    attempts: list[str] = []
    for s in guided + [g for g in generic_order if g not in guided]:
        if s in tried:
            continue
        tried.add(s)
        attempts.append(s)
        try:
            _log(f"  ▶ Trying reader: {s} ...")
            reader = _reader_for(s)
            df = reader(path)
            _log(f"  ✔ Success with reader: {s}")
            return s, reader, df
        except Exception as e:
            _log(f"  ✖ Reader '{s}' failed: {e}")

    raise RuntimeError(f"Unable to detect a working reader for '{path}'. Attempts: {attempts}")


def _has_inline_neighbors(df: pd.DataFrame) -> bool:
    """Check if DataFrame has inline neighbor columns vx_1..vx_125, vy_*, vz_*.
    Requires at least vx_1, vy_1, vz_1 and vx_125, vy_125, vz_125 to avoid expensively scanning all.
    """
    cols = set(df.columns)
    return (
        'vx_1' in cols and 'vy_1' in cols and 'vz_1' in cols and
        'vx_125' in cols and 'vy_125' in cols and 'vz_125' in cols
    )


def _build_vector_from_inline_row(row: pd.Series) -> np.ndarray:
    """Assemble a 375-dim vector from a single row that already contains neighbor values
    as columns vx_1..vx_125, vy_1..vy_125, vz_1..vz_125, ordered by neighbor index.
    """
    values: List[float] = []
    for i in range(1, 126):
        try:
            values.extend([float(row[f'vx_{i}']), float(row[f'vy_{i}']), float(row[f'vz_{i}'])])
        except KeyError as e:
            raise KeyError(f"Missing inline neighbor column(s) for index {i}: {e}")
    v = np.asarray(values, dtype=np.float32)
    if v.size != 375:
        raise ValueError(f"Inline vector length {v.size} != 375")
    return v


def _build_neighborhood_vector(df: pd.DataFrame, finder: adjacent_coordinate, x: int, y: int, z: int) -> np.ndarray:
    """Return 375-dim vector as [vx,vy,vz] repeated for 125 coordinates in the 3p6 neighborhood order.
    If a neighbor is missing in df, raises KeyError.
    """
    coords = finder.find_adjacent_coordinates(int(x), int(y), int(z))
    values: List[float] = []
    # Speed up lookups by indexing
    # Assumes df has x,y,z columns and vx,vy,vz values
    # Create a MultiIndex view for fast exact match
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['x', 'y', 'z']:
        df = df.set_index(['x', 'y', 'z'], drop=False)
    for cx, cy, cz in coords:
        try:
            row = df.loc[(int(cx), int(cy), int(cz))]
        except KeyError:
            raise KeyError(f"Missing neighbor coordinate ({cx},{cy},{cz}) in current frame")
        values.extend([float(row['vx']), float(row['vy']), float(row['vz'])])
    v = np.array(values, dtype=np.float32)
    if v.shape[0] != 375:
        raise ValueError(f"Neighborhood length {v.shape[0]} != 375 (expected 125 triplets)")
    return v


def _rmse_components(orig_vec: np.ndarray, recon_vec: np.ndarray) -> Tuple[float, float, float]:
    """Compute RMSE for vx, vy, vz across the 125 neighbors.
    Input vectors are flat [375], with triplets [vx, vy, vz] per neighbor.
    Returns (rmse_vx, rmse_vy, rmse_vz).
    """
    a = orig_vec.reshape(-1, 3)
    b = recon_vec.reshape(-1, 3)
    dif = a - b
    if ABSOLUTE_RESIDUALS:
        dif = np.abs(dif)
    mse = np.mean(dif * dif, axis=0)  # shape (3,)
    rmse = np.sqrt(mse)
    return float(rmse[0]), float(rmse[1]), float(rmse[2])


def _compute_percentiles(vals: np.ndarray, qs=(25, 50, 75)) -> Tuple[float, float, float]:
    p = np.percentile(vals, qs, method='linear') if hasattr(np, 'percentile') else np.quantile(vals, np.array(qs)/100.0)
    return float(p[0]), float(p[1]), float(p[2])


def _green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"


def process_over_time(
    time_dir: str,
    model_path: str = MODEL_CHECKPOINT_PATH,
    meta_project_json: str = META_PROJECT_JSON,
    experiment_name: str | None = None,
    cache_csv_path: str | None = None,
    assume_format: str | None = None,
    inner_progress_mode: str = "oneline",  # 'oneline' | 'tqdm' | 'off'
    inner_update_every: int = 50,
) -> str:
    """Process all time files and return path to the cached CSV with percentile time series."""
    logger.debug("Starting process_over_time with parameters:")
    logger.debug(f"  time_dir={time_dir}")
    logger.debug(f"  model_path={model_path}")
    logger.debug(f"  meta_project_json={meta_project_json}")
    logger.debug(f"  experiment_name={experiment_name}")
    logger.debug(f"  cache_csv_path={cache_csv_path}")
    logger.debug(f"  assume_format={assume_format}")
    logger.debug(f"  inner_progress_mode={inner_progress_mode}")
    logger.debug(f"  inner_update_every={inner_update_every}")

    files = _discover_time_files(time_dir)
    t_indices = [t for t, _ in files]
    # Detect reader and load first frame to establish coordinate set
    strategy, reader, df0 = _detect_and_build_reader(files[0][1], assume_format=assume_format)
    logger.info(f"Using reader strategy '{strategy}' for all subsequent files.")
    logger.debug(f"First frame shape: {df0.shape}; columns: {list(df0.columns)}")
    required_cols = {'x', 'y', 'z', 'vx', 'vy', 'vz'}
    missing = required_cols - set(df0.columns)
    if missing:
        raise ValueError(f"First frame missing columns: {missing}")
    coords0 = df0[['x', 'y', 'z']].astype(int).to_records(index=False).tolist()
    coords0_set = set(coords0)
    logger.debug(f"Number of base coordinates from first frame: {len(coords0)}")

    # Determine whether frames include inline neighbor columns (preferred, avoids heavy neighbor lookups)
    inline_mode = _has_inline_neighbors(df0)
    logger.info(f"Vector build mode: {'inline_columns' if inline_mode else 'neighbor_finder'}")
    if inline_mode:
        # Quick diagnostic: ensure the expected column range appears to exist
        missing_any = [name for name in ("vx_1","vy_1","vz_1","vx_125","vy_125","vz_125") if name not in df0.columns]
        if missing_any:
            logger.warning(f"Inline mode selected but some sentinel columns missing: {missing_any}")

    # Determine experiment name from directory name if not provided
    if not experiment_name:
        experiment_name = os.path.basename(os.path.abspath(time_dir))
    logger.debug(f"Effective experiment_name={experiment_name}")

    # Build the neighborhood finder (e.g., '3p6') only if needed
    if not inline_mode:
        # Log effective metadata path and attempt to instantiate the neighborhood finder
        logger.info(f"Using meta_project_json: {meta_project_json}")
        finder = adjacent_coordinate(meta_project_json, experiment_name)
    else:
        finder = None  # not used in inline mode

    # Diagnostic: confirm neighborhood length
    try:
        sample_vec = _build_neighborhood_vector(df0.copy(), finder, int(coords0[0][0]), int(coords0[0][1]), int(coords0[0][2]))
        print(_green(f"Diagnostic: neighborhood vector length = {sample_vec.size} (expect 375: 125 triplets). OK."))
    except Exception as e:
        print(_green(f"Diagnostic failed to confirm neighborhood length: {e}"))

    device = _pick_device()
    model = _load_model(model_path, device)
    logger.info(f"Using device: {device}; Loaded model: {model_path}")

    # Prepare arrays to collect per-time percentiles for each component
    p25_vx, p50_vx, p75_vx = [], [], []
    p25_vy, p50_vy, p75_vy = [], [], []
    p25_vz, p50_vz, p75_vz = [], [], []

    # Prepare wide per-coordinate, per-time storage (what the user requested)
    # We will build big matrices of shape (n_coords, n_times) for each component (vx, vy, vz),
    # fill them as we process each time step, and write a single wide CSV at the end:
    # columns: x, y, z, rmse_vx_0001..rmse_vx_N, rmse_vy_0001..rmse_vy_N, rmse_vz_0001..rmse_vz_N
    # Note: use float32 to keep memory reasonable; values are RMSEs.
    n_coords = len(coords0)
    n_times = len(t_indices)
    rmse_vx_wide = np.full((n_coords, n_times), np.nan, dtype=np.float32)
    rmse_vy_wide = np.full((n_coords, n_times), np.nan, dtype=np.float32)
    rmse_vz_wide = np.full((n_coords, n_times), np.nan, dtype=np.float32)
    # Stable base coordinate order and a mapping to row index
    base_coords = list(coords0)
    coord_to_row = {tuple(map(int, c)): i for i, c in enumerate(base_coords)}
    # Map actual time value (e.g., 1..N) to column index 0..n_times-1
    time_to_col = {int(t): j for j, t in enumerate(t_indices)}

    # Resolve output CSV path for per-coordinate running RMSE
    if cache_csv_path is None:
        base = os.path.basename(os.path.abspath(time_dir))
        cache_csv_path = os.path.join(
            OUTPUT_DIR,
            f"validation_WAE_01_details_over_time_{base}.csv"
        )
    os.makedirs(os.path.dirname(cache_csv_path), exist_ok=True)

    # (Optional future) Resume from an existing wide CSV is non-trivial and is not implemented here.
    # We will always recompute the wide matrices on a new run to guarantee consistency.

    # Pre-index all frames lazily (load on the fly to limit mem)
    skipped_files = 0
    for t, path in tqdm(files, desc="Processing time files", unit="file"):
        logger.debug(f"Time t={t} file='{os.path.basename(path)}' — begin")
        # Read using detected/assumed reader; skip on failure with NaNs
        try:
            df_t = reader(path)
        except Exception as e:
            logger.warning(f"Failed to read time file '{os.path.basename(path)}' with strategy '{strategy}': {e}. Marking NaNs for t={t}.")
            skipped_files += 1
            p25_vx.append(float('nan')); p50_vx.append(float('nan')); p75_vx.append(float('nan'))
            p25_vy.append(float('nan')); p50_vy.append(float('nan')); p75_vy.append(float('nan'))
            p25_vz.append(float('nan')); p50_vz.append(float('nan')); p75_vz.append(float('nan'))
            continue
        # Index for fast coordinate lookup
        logger.debug(f"Loaded frame t={t} shape: {df_t.shape}; columns: {list(df_t.columns)}")
        df_t = df_t.set_index(['x', 'y', 'z'], drop=False)

        # Determine coordinates to process for this frame (intersection with base set)
        current_coords = set(df_t.index.to_list())
        coords_to_process = list(coords0_set & current_coords)
        if not coords_to_process:
            logger.warning(f"t={t}: No coordinate intersection between base frame and current frame. Marking NaNs for this time.")
            p25_vx.append(float('nan')); p50_vx.append(float('nan')); p75_vx.append(float('nan'))
            p25_vy.append(float('nan')); p50_vy.append(float('nan')); p75_vy.append(float('nan'))
            p25_vz.append(float('nan')); p50_vz.append(float('nan')); p75_vz.append(float('nan'))
            continue

        rmse_vx_list = []
        rmse_vy_list = []
        rmse_vz_list = []

        attempted = 0
        successes = 0
        missing_neighbors = 0
        debug_samples_logged = 0
        # Track previous medians for colored delta display
        prev_vx_med: float | None = None
        prev_vy_med: float | None = None
        prev_vz_med: float | None = None
        _EPS = 1e-12
        # Helper: single-line status printer
        class _OneLineStatus:
            def __init__(self):
                self._last_len = 0
                # detect terminal width for safe truncation
                try:
                    self._width = shutil.get_terminal_size((120, 20)).columns
                except Exception:
                    self._width = 120
            def print(self, s: str):
                # ensure string fits into width; keep some room
                maxw = max(20, self._width - 1)
                if len(s) > maxw:
                    s = s[:maxw - 3] + "..."
                pad = max(0, self._last_len - len(s))
                sys.stdout.write("\r" + s + (" " * pad))
                sys.stdout.flush()
                self._last_len = len(s)
            def newline(self):
                sys.stdout.write("\n")
                sys.stdout.flush()

        use_oneline = (inner_progress_mode or "").lower() == "oneline"
        use_tqdm_inner = (inner_progress_mode or "").lower() == "tqdm"
        disable_inner = (inner_progress_mode or "").lower() == "off"

        if use_tqdm_inner:
            inner_bar = tqdm(
                total=len(coords_to_process), desc=f"t={t}", unit="row",
                leave=False, position=1, dynamic_ncols=True, miniters=max(1, inner_update_every), mininterval=0.3
            )
        else:
            inner_bar = None
        status = _OneLineStatus() if use_oneline else None

        # Reduce log noise during inner loop to avoid breaking single-line rendering
        saved_level = logger.level
        if use_oneline:
            try:
                logger.setLevel(logging.INFO)
            except Exception:
                pass
        for (x, y, z) in coords_to_process:
            try:
                if inline_mode:
                    row = df_t.loc[(int(x), int(y), int(z))]
                    v_vec = _build_vector_from_inline_row(row)
                else:
                    v_vec = _build_neighborhood_vector(df_t, finder, int(x), int(y), int(z))
            except KeyError:
                missing_neighbors += 1
                if inner_bar is not None:
                    inner_bar.update(1)
                continue
            attempted += 1
            x_t = torch.from_numpy(v_vec[None, :]).to(device)
            with torch.no_grad():
                recon, _ = model(x_t)
            recon_np = recon.detach().cpu().numpy().reshape(-1)
            rvx, rvy, rvz = _rmse_components(v_vec, recon_np)
            rmse_vx_list.append(rvx)
            rmse_vy_list.append(rvy)
            rmse_vz_list.append(rvz)
            successes += 1
            if debug_samples_logged < 3:
                logger.debug(
                    f"t={t} coord=({int(x)},{int(y)},{int(z)}) v_vec.len={len(v_vec)} recon.len={len(recon_np)} "
                    f"rmse=(vx={rvx:.6f}, vy={rvy:.6f}, vz={rvz:.6f})"
                )
                debug_samples_logged += 1
            # Fill wide matrices at the row of this coordinate and the column of this time
            row_idx = coord_to_row.get((int(x), int(y), int(z)))
            col_idx = time_to_col.get(int(t))
            if row_idx is not None and col_idx is not None:
                rmse_vx_wide[row_idx, col_idx] = rvx
                rmse_vy_wide[row_idx, col_idx] = rvy
                rmse_vz_wide[row_idx, col_idx] = rvz

            # Always show a live median snapshot (after at least 1 success)
            try:
                import numpy as _np
                if successes >= 1:
                    cur_vx = float(_np.median(rmse_vx_list))
                    cur_vy = float(_np.median(rmse_vy_list))
                    cur_vz = float(_np.median(rmse_vz_list))
                    def _fmt_colored(cur: float, prev: float | None) -> str:
                        # 8 significant digits as requested
                        txt = f"{cur:.8g}"
                        if prev is None:
                            return txt
                        if cur < prev - _EPS:
                            return f"\033[92m{txt}\033[0m"  # green
                        if cur > prev + _EPS:
                            return f"\033[91m{txt}\033[0m"  # red
                        return txt
                    vx_txt = _fmt_colored(cur_vx, prev_vx_med)
                    vy_txt = _fmt_colored(cur_vy, prev_vy_med)
                    vz_txt = _fmt_colored(cur_vz, prev_vz_med)
                else:
                    vx_txt = vy_txt = vz_txt = "n/a"
                msg = f"t={t}: {successes}/{len(coords_to_process)} ({successes/len(coords_to_process):.1%}) " \
                      f"vx_med={vx_txt}, vy_med={vy_txt}, vz_med={vz_txt}"
                # Update inner progress display according to mode
                if use_tqdm_inner and inner_bar is not None:
                    inner_bar.set_postfix_str(f"vx_med={vx_txt}, vy_med={vy_txt}, vz_med={vz_txt}")
                elif use_oneline and status is not None and (successes % max(1, inner_update_every) == 0):
                    status.print(msg)
                if successes >= 1:
                    prev_vx_med, prev_vy_med, prev_vz_med = cur_vx, cur_vy, cur_vz
            except Exception:
                pass
            if inner_bar is not None:
                inner_bar.update(1)
            elif use_oneline and status is not None and (successes % max(1, inner_update_every) == 0):
                # if we didn't meet the >=3 condition (early rows), still show coarse progress
                status.print(f"t={t}: {successes}/{len(coords_to_process)} ({successes/len(coords_to_process):.1%})")
        # finalize inner progress line
        if inner_bar is not None:
            # finalize postfix with last medians if available
            if prev_vx_med is not None:
                vx_txt = f"{prev_vx_med:.8g}"
                vy_txt = f"{prev_vy_med:.8g}"
                vz_txt = f"{prev_vz_med:.8g}"
                inner_bar.set_postfix_str(f"vx_med={vx_txt}, vy_med={vy_txt}, vz_med={vz_txt}")
            inner_bar.close()
        elif use_oneline and status is not None:
            # ensure a newline so the outer bar continues on next line cleanly
            status.newline()
        # restore logger level
        if use_oneline:
            try:
                logger.setLevel(saved_level)
            except Exception:
                pass

        # Compute percentiles across all coordinates for this time
        if rmse_vx_list:
            p25, p50, p75 = _compute_percentiles(np.asarray(rmse_vx_list))
        else:
            p25 = p50 = p75 = float('nan')
        p25_vx.append(p25); p50_vx.append(p50); p75_vx.append(p75)

        if rmse_vy_list:
            p25, p50, p75 = _compute_percentiles(np.asarray(rmse_vy_list))
        else:
            p25 = p50 = p75 = float('nan')
        p25_vy.append(p25); p50_vy.append(p50); p75_vy.append(p75)

        if rmse_vz_list:
            p25, p50, p75 = _compute_percentiles(np.asarray(rmse_vz_list))
        else:
            p25 = p50 = p75 = float('nan')
        p25_vz.append(p25); p50_vz.append(p50); p75_vz.append(p75)

        logger.debug(
            f"t={t} summary: total_base_coords={len(coords0)} attempted={attempted} "
            f"successes={successes} missing_neighbor_skips={missing_neighbors} "
            f"percentiles: vx=({p25_vx[-1]}, {p50_vx[-1]}, {p75_vx[-1]}), "
            f"vy=({p25_vy[-1]}, {p50_vy[-1]}, {p75_vy[-1]}), "
            f"vz=({p25_vz[-1]}, {p50_vz[-1]}, {p75_vz[-1]})"
        )

        # We no longer persist a per-time running CSV; the wide CSV is written once at the end.

    # After processing all times: write the requested WIDE CSV (one row per coord, three blocks of N time columns)
    try:
        # Build coordinate DataFrame in the base order
        coord_df = pd.DataFrame(base_coords, columns=['x', 'y', 'z'])
        # Build column names for each component using actual time indices (zero-padded to 4 digits)
        vx_cols = [f"rmse_vx_{int(t):04d}" for t in t_indices]
        vy_cols = [f"rmse_vy_{int(t):04d}" for t in t_indices]
        vz_cols = [f"rmse_vz_{int(t):04d}" for t in t_indices]
        vx_df = pd.DataFrame(rmse_vx_wide, columns=vx_cols)
        vy_df = pd.DataFrame(rmse_vy_wide, columns=vy_cols)
        vz_df = pd.DataFrame(rmse_vz_wide, columns=vz_cols)
        wide_df = pd.concat([coord_df, vx_df, vy_df, vz_df], axis=1)
        os.makedirs(os.path.dirname(cache_csv_path), exist_ok=True)
        wide_df.to_csv(cache_csv_path, index=False)
        logger.info(f"Saved wide per-coordinate CSV (rows={len(wide_df)}, cols={wide_df.shape[1]}): {cache_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write wide CSV '{cache_csv_path}': {e}")

    # After processing all times: also write a percentiles-over-time CSV alongside the wide CSV
    base, ext = os.path.splitext(cache_csv_path)
    percentiles_csv_path = base + "_percentiles" + ext
    percentiles_df = pd.DataFrame({
        'time': t_indices,
        'vx_p25': p25_vx, 'vx_p50': p50_vx, 'vx_p75': p75_vx,
        'vy_p25': p25_vy, 'vy_p50': p50_vy, 'vy_p75': p75_vy,
        'vz_p25': p25_vz, 'vz_p50': p50_vz, 'vz_p75': p75_vz,
    })
    percentiles_df.to_csv(percentiles_csv_path, index=False)
    if skipped_files:
        logger.warning(f"Skipped {skipped_files} files due to read errors. Corresponding times have NaNs in the percentiles CSV.")
    logger.info(f"Saved percentiles CSV: {percentiles_csv_path}")
    return cache_csv_path


def _plot_component_percentiles(csv_path: str, component: str, out_dir: str) -> str:
    df = pd.read_csv(csv_path)
    t = df['time'].values
    p25 = df[f'{component}_p25'].values
    p50 = df[f'{component}_p50'].values
    p75 = df[f'{component}_p75'].values
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(t, p25, label='p25', color='tab:blue')
    ax.plot(t, p50, label='p50', color='tab:orange')
    ax.plot(t, p75, label='p75', color='tab:green')
    ax.fill_between(t, p25, p75, color='gray', alpha=0.2, label='IQR')
    ax.set_xlabel('Time (file index)')
    ax.set_ylabel('RMSE')
    ax.set_title(f'RMSE percentiles over time — {component}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'over_time_{component}_percentiles.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute WAE residual RMSE percentiles over time for vx/vy/vz')
    parser.add_argument('--time_dir', type=str, required=True, help='Directory containing time-sliced pickle files (named 1..N.*)')
    parser.add_argument('--model_path', type=str, default=MODEL_CHECKPOINT_PATH, help='Path to WAE checkpoint')
    parser.add_argument('--meta_project_json', type=str, default=META_PROJECT_JSON, help='Meta project JSON for adjacent neighborhood rules')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (defaults to basename of time_dir, e.g., 3p6)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to write outputs')
    parser.add_argument('--cache_csv', type=str, default=None, help='Optional path for cached percentiles CSV')
    parser.add_argument('--assume_format', type=str, default='auto',
                        help="Assume a specific input format for time files: auto|pickle|gzip_pickle|bz2_pickle|xz_pickle|zip_pickle|zstd_pickle|parquet|feather|csv")
    args = parser.parse_args()

    csv_path = process_over_time(
        time_dir=args.time_dir,
        model_path=args.model_path,
        meta_project_json=args.meta_project_json,
        experiment_name=args.experiment_name,
        cache_csv_path=args.cache_csv,
        assume_format=args.assume_format,
    )

    # Plots: three separate per-component figures using the percentiles CSV
    base_out_dir = os.path.join(args.output_dir, 'over_time')
    base, ext = os.path.splitext(csv_path)
    percentiles_csv_path = base + "_percentiles" + ext
    vx_png = _plot_component_percentiles(percentiles_csv_path, 'vx', base_out_dir)
    vy_png = _plot_component_percentiles(percentiles_csv_path, 'vy', base_out_dir)
    vz_png = _plot_component_percentiles(percentiles_csv_path, 'vz', base_out_dir)
    logger.info(f"Saved plots: {vx_png}, {vy_png}, {vz_png}")


if __name__ == '__main__':
    main()
