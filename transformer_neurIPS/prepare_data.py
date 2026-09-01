import os
import hashlib
import h5py
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import random
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from typing import Dict, List, Tuple

# Paths
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
SOURCE_ROOT = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "transformer_neurIPS/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# NUM_TIME_NEW is the sequence length (context + forecast) in frames.
# v3 policy: ONLY 80-frame data is produced going forward (12 context + 68
# forecast, 566.7 ms forecast @ 120 Hz). The historical v1.0 40-frame path
# is no longer supported by this file -- do not resurrect it. Any 40-frame
# H5 (train_40.h5 / val_40.h5) still on disk is treated as a legacy artifact
# that must not be regenerated; regenerate as train_80.h5 / val_80.h5 only.
# WINDOWS_PER_COORD is derived so every coordinate contributes disjoint,
# non-overlapping windows tiling all TOTAL_TIMESTAMPS frames.
NUM_TIME_NEW = 80
TOTAL_TIMESTAMPS = 1200
WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW  # 15 for N=80

# X-coordinate policy -- constrained to |x| <= 20 (inclusive) as of v3.
#
# The full sensor line originally spanned x in [-29, +69], 26 samples per
# (y, z, t). Empirically the interesting wake dynamics -- vortex cores,
# reversal events, high-vorticity structure -- concentrate near the
# tunnel centreline, so v3 restricts the per-token x-sweep to the
# inclusive range [-20, +20]. That drops the count from 26 -> 10:
#
#     kept:   [-18, -14, -10, -6, -2, 1, 5, 9, 13, 17]
#     dropped (|x| > 20):
#             [-29, -26, -22]   (far upstream, mostly freestream)
#             [21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69]
#                              (far downstream, sparse structure)
#
# The wake coordinates (WAKE_COORDS below) live in (y, z) and are
# unaffected by this filter -- every wake region still contributes a
# full sequence, just with fewer x-samples per token row.
#
# Downstream trainer impact: SEQ_LEN = NUM_TIME * NUM_X changes from
# 80 * 26 = 2080 to 80 * 10 = 800. Any resume from a v2.0/v2.1 _latest.pt
# built at 2080 tokens will hit the same length-dependent shape-mismatch
# path in the trainer's resume block that v1.0 -> v2.0 already handled,
# and length-dependent tensors (time_embeddings.weight, positional
# embeddings) will be dropped and rebuilt. Cold-start is cleaner.
NUM_X = 10
X_COORDS = np.array([-18, -14, -10, -6, -2, 1, 5, 9, 13, 17], dtype='float32')

# --------------------------------------------------------------------------
# v3.4 (see OVERVIEW.md §15): wake seeds now come from the PHYSICS-DERIVED
# wake atlas built by `build_wake_atlas.py`. The atlas covers every
# `(param, step)` and includes spatial sliding-window neighbours, so a
# single param has thousands (not 24) of unique `(y_grid, z_grid)` seeds.
#
# The 24-tap hand-curated list below is retained ONLY as an opt-in
# diagnostic fallback -- if the atlas is missing/empty for a param AND
# the environment variable `PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK=1`
# is set, that param falls back to these 24 taps (loud red warning
# logged). Without the env var, missing atlas is a hard SystemExit.
# --------------------------------------------------------------------------
_LEGACY_WAKE_COORDS_FOR_FALLBACK = [
    (-71, -1), (-67, -1), (-63, -21), (-59, -17), (-55, 2), (-47, -21),
    (-43, 22), (-31, -21), (-16, 22), (-12, 22), (-8, 18), (0, -1),
    (3, 10), (11, 22), (15, 22), (23, 22), (27, 22), (39, 10),
    (47, -21), (55, 2), (59, 2), (67, 10), (71, -13), (75, -5)
]
# Back-compat alias: some tests / external callers still reference the
# old public name. It points at the same legacy fallback list; do NOT
# use it as a live wake-seed source in new code -- always go through
# `load_wake_atlas(...)`.
WAKE_COORDS = _LEGACY_WAKE_COORDS_FOR_FALLBACK

# Atlas artifact locations (produced by build_wake_atlas.py).
WAKE_ATLAS_CSV = os.path.join(OUTPUT_DIR, "wake_atlas.csv.gz")
WAKE_ATLAS_SHARDS_DIR = os.path.join(OUTPUT_DIR, "wake_atlas_shards")
WAKE_ATLAS_SHA256_PATH = os.path.join(OUTPUT_DIR, "wake_atlas.sha256")


def _read_shard_file(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _iter_atlas_shards(shards_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(shards_dir):
        return out
    for p in sorted(os.listdir(shards_dir)):
        pdir = os.path.join(shards_dir, p)
        if not os.path.isdir(pdir):
            continue
        for f in sorted(os.listdir(pdir)):
            if f.startswith("step_") and (f.endswith(".parquet")
                                          or f.endswith(".csv.gz")):
                out.append(os.path.join(pdir, f))
    return out


def load_wake_atlas(atlas_path_or_shards_dir: str,
                    include_neighbours: bool = True
                    ) -> Dict[str, List[Tuple[int, int]]]:
    """Load the physics-derived wake atlas.

    Args:
      atlas_path_or_shards_dir: path to a merged `wake_atlas.csv.gz` file,
        OR to a directory containing per-`(param, step)` shard files.
        If a directory is given (or if the merged file is absent and
        a sibling shard tree exists), shards are loaded and merged in
        memory (with a warning).
      include_neighbours: if True (default), spatial sliding-window
        neighbour rows are included alongside the core rows.

    Returns:
      {param: [(y_grid, z_grid), ...]} with duplicates removed and
      stable-sorted by first appearance.

    Raises:
      FileNotFoundError if neither the merged file nor a non-empty
      shards directory exists.
    """
    # Resolve where to read from.
    df: pd.DataFrame
    if os.path.isfile(atlas_path_or_shards_dir):
        df = pd.read_csv(atlas_path_or_shards_dir)
    elif os.path.isdir(atlas_path_or_shards_dir):
        files = _iter_atlas_shards(atlas_path_or_shards_dir)
        if not files:
            raise FileNotFoundError(
                f"[wake-atlas] no shard files found under "
                f"{atlas_path_or_shards_dir}; run build_wake_atlas.py first.")
        print(f"  [wake-atlas] auto-merging {len(files)} shards from "
              f"{atlas_path_or_shards_dir} (no wake_atlas.csv.gz present).")
        frames = []
        for fp in files:
            try:
                shard = _read_shard_file(fp)
            except Exception as e:
                print(f"  [wake-atlas] WARNING: failed to read shard {fp}: {e}")
                continue
            if len(shard) == 0:
                continue
            frames.append(shard)
        if not frames:
            raise FileNotFoundError(
                f"[wake-atlas] shard tree {atlas_path_or_shards_dir} "
                f"contained zero readable rows.")
        df = pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError(
            f"[wake-atlas] neither a merged wake_atlas.csv.gz nor a shards "
            f"directory at {atlas_path_or_shards_dir}")

    # Drop empty-sentinel rows (n_core_points == 0) if present.
    if "n_core_points" in df.columns:
        df = df[df["n_core_points"] > 0]
    if not include_neighbours and "is_neighbour" in df.columns:
        df = df[df["is_neighbour"] == False]  # noqa: E712

    result: Dict[str, List[Tuple[int, int]]] = {}
    seen: Dict[str, set] = {}
    for _, row in df.iterrows():
        ps = str(row["param"])
        y = int(row["y_grid"])
        z = int(row["z_grid"])
        s = seen.setdefault(ps, set())
        if (y, z) in s:
            continue
        s.add((y, z))
        result.setdefault(ps, []).append((y, z))
    return result


def _atlas_source_and_sha(atlas_path: str) -> Tuple[str, str]:
    """Return (absolute_source_path, sha256) for the atlas file. If a
    committed sha256 file is present it is trusted; otherwise the file
    is hashed on the fly. Returns ('', '') if `atlas_path` is not a file.
    """
    if not os.path.isfile(atlas_path):
        return "", ""
    src = os.path.abspath(atlas_path)
    # If a sibling .sha256 exists, prefer it (matches build_wake_atlas).
    if os.path.isfile(WAKE_ATLAS_SHA256_PATH):
        try:
            with open(WAKE_ATLAS_SHA256_PATH, "r") as f:
                sha = f.read().strip().split()[0]
            if sha:
                return src, sha
        except Exception:
            pass
    h = hashlib.sha256()
    with open(atlas_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return src, h.hexdigest()


def resolve_wake_atlas_for_params(param_list: List[str]
                                  ) -> Tuple[Dict[str, List[Tuple[int, int]]],
                                             Dict[str, object]]:
    """Load the atlas and derive per-param wake-seed lists, with an
    opt-in legacy fallback for params that have zero atlas rows.

    Returns (atlas_yz_by_param, atlas_meta) where atlas_meta carries
    H5-attr provenance: 'wake_atlas_source', 'wake_atlas_sha256',
    'wake_atlas_rows' (per-param count, JSON string).
    """
    allow_fallback = os.environ.get(
        "PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK", "").strip() == "1"

    atlas: Dict[str, List[Tuple[int, int]]] = {}
    source = ""
    sha = ""
    if os.path.isfile(WAKE_ATLAS_CSV):
        atlas = load_wake_atlas(WAKE_ATLAS_CSV)
        source, sha = _atlas_source_and_sha(WAKE_ATLAS_CSV)
    elif os.path.isdir(WAKE_ATLAS_SHARDS_DIR) and _iter_atlas_shards(
            WAKE_ATLAS_SHARDS_DIR):
        atlas = load_wake_atlas(WAKE_ATLAS_SHARDS_DIR)
        source = os.path.abspath(WAKE_ATLAS_SHARDS_DIR)
        sha = "shard-tree-in-memory-merge"
    else:
        if not allow_fallback:
            raise SystemExit(
                "[wake-atlas] no wake_atlas.csv.gz and no shard tree found "
                f"under {OUTPUT_DIR}. Run:\n"
                "    python transformer_neurIPS/build_wake_atlas.py --merge\n"
                "to produce the atlas, or export "
                "PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK=1 to opt into the "
                "legacy hardcoded 24 taps for diagnostics.")
        print("\033[93m  [wake-atlas] LEGACY FALLBACK ACTIVE (env opt-in): "
              "no atlas artifacts on disk; using the hardcoded 24 taps for "
              "every param.\033[0m")
        atlas = {ps: list(_LEGACY_WAKE_COORDS_FOR_FALLBACK)
                 for ps in param_list}
        source = "legacy:_LEGACY_WAKE_COORDS_FOR_FALLBACK"
        sha = "legacy"

    # Fill in any missing param with fallback (opt-in) or empty (fail-fast).
    result: Dict[str, List[Tuple[int, int]]] = {}
    per_param_rows: Dict[str, int] = {}
    for ps in param_list:
        rows = atlas.get(ps, [])
        if not rows:
            if allow_fallback:
                print(f"\033[93m  [wake-atlas] LEGACY FALLBACK ACTIVE for "
                      f"'{ps}': atlas has zero rows; using 24 hardcoded "
                      f"taps for THIS PARAM only.\033[0m")
                rows = list(_LEGACY_WAKE_COORDS_FOR_FALLBACK)
            else:
                raise SystemExit(
                    f"[wake-atlas] param '{ps}' has zero rows in the atlas "
                    f"at {source}. Rebuild the atlas for that param via "
                    f"`python transformer_neurIPS/build_wake_atlas.py "
                    f"--params {ps} --merge`, or opt into the legacy "
                    f"fallback with "
                    f"PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK=1.")
        result[ps] = rows
        per_param_rows[ps] = len(rows)
        print(f"  [wake-atlas] '{ps}': {len(rows)} unique (y_grid, z_grid) "
              f"rows from {source}")

    meta = {
        "wake_atlas_source": source,
        "wake_atlas_sha256": sha,
        "wake_atlas_rows": json.dumps(per_param_rows, sort_keys=True),
    }
    return result, meta

def parse_param(p_str):
    mapping = {"3p6": 5.6, "4p4": 6.9, "4p6": 7.2, "5p2": 8.1, "6p4": 10.0, 
               "6p6": 10.3, "7p2": 11.3, "7p8": 12.2, "8p4": 13.1, "10p4": 16.3, "11p4": 17.8}
    return mapping.get(p_str, 0.0)

def get_file_path(param_set, step):
    return os.path.join(SOURCE_ROOT, param_set, f"{step:04d}.pkl.gz")

def extract_from_file(args):
    """Worker function to extract multiple coordinates from a single file.

    Returns a dict {(y,z): features_or_None} on success, or a dict
    {'__error__': reason_str} / {'__missing__': path} on skip, so the caller
    can log why a file/step contributed nothing rather than silently
    dropping the whole timestep. (The previous return-None-on-anything
    behavior is exactly what hid the '4p4/ folder missing on disk' bug
    behind a suspiciously fast tqdm rate.)
    """
    f_path, coords_to_extract, param_val = args
    if not os.path.exists(f_path):
        return {'__missing__': f_path}

    try:
        df = pd.read_pickle(f_path, compression='gzip')
        latent_cols = [c for c in df.columns if 'latent' in c.lower()]
        if not latent_cols:
             raise ValueError(f"No 'latent' columns found in {f_path}")
        
        results = {}
        # Pre-filter dataframe for all requested coordinates at once to speed up lookups
        ys = [c[0] for c in coords_to_extract]
        zs = [c[1] for c in coords_to_extract]
        subset = df[df['y'].isin(ys) & df['z'].isin(zs)]
        
        for y_val, z_val in coords_to_extract:
            rows = subset[(subset['y'] == y_val) & (subset['z'] == z_val)]
            if rows.empty:
                results[(y_val, z_val)] = None
                continue
            
            rows = rows.set_index('x').reindex(X_COORDS).reset_index()
            lats = np.nan_to_num(rows[latent_cols].values)
            
            # Stack features: latents(47), x(1), y(1), z(1), t_idx(placeholder), param(1)
            # Coordinates y and z are stored as int32
            extracted = np.column_stack([
                lats.astype('float32'), 
                rows['x'].values.astype('float32'),
                np.full(NUM_X, y_val, dtype='int32'),
                np.full(NUM_X, z_val, dtype='int32'),
                np.zeros(NUM_X, dtype='float32'), # t_idx placeholder
                np.full(NUM_X, param_val, dtype='float32')
            ])
            results[(y_val, z_val)] = extracted
        return results
    except Exception as e:
        # Surface the failure reason so a corrupt/unexpected pickle stops
        # being invisible; the caller logs a summary rather than one line
        # per file to keep the console readable at 1200-step scale.
        return {'__error__': f"{type(e).__name__}: {e} @ {f_path}"}


def _format_timeout_seconds(seconds):
    if seconds is None:
        return "disabled"
    return f"{seconds:.1f}s"


def _make_wall_clock_deadline(wall_clock_sec):
    if wall_clock_sec is None or wall_clock_sec <= 0:
        return None
    return time.monotonic() + float(wall_clock_sec)


def _check_wall_clock_deadline(deadline, started_at, phase, detail=""):
    if deadline is None:
        return
    now = time.monotonic()
    if now <= deadline:
        return
    elapsed = float(now - started_at)
    limit = float(deadline - started_at)
    extra = f" {detail}" if detail else ""
    raise SystemExit(
        f"[timeout] wall-clock limit exceeded during {phase} after "
        f"{elapsed:.1f}s (limit={limit:.1f}s).{extra}")


def _remaining_wall_clock_seconds(deadline):
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _build_random_plans(param_list, atlas_yz_by_param, num_wake_plans,
                        param_to_coords):
    random.seed(42)

    ps_to_valid_coords = {}
    ps_to_allow_atlas_coords = {}
    fallback_used = False
    atlas_probe_fallback_used = False
    for ps in param_list:
        per_ps_wake_set = set(atlas_yz_by_param.get(ps, []))
        probe = get_file_path(ps, 1)
        valid = None
        allow_atlas_coords = False
        if os.path.exists(probe):
            try:
                sdf = pd.read_pickle(probe, compression='gzip')
                yz_all = {(int(y), int(z))
                          for y, z in zip(sdf['y'].values, sdf['z'].values)}
                valid = sorted(yz_all - per_ps_wake_set)
                if not valid:
                    # Fallback path: atlas exclusion has emptied the
                    # pool. Fall back to `probe_yz` (i.e. allow atlas
                    # rows as random candidates for this param). Loud
                    # log so the shrinkage doesn't hide.
                    valid = sorted(yz_all)
                    allow_atlas_coords = True
                    atlas_probe_fallback_used = True
                    print(f"  [random-plan] WARNING: atlas exclusion "
                          f"emptied the non-wake pool for '{ps}'. "
                          f"Falling back to full probe_yz "
                          f"({len(valid)} coords) so random plans "
                          f"can populate at all.")
            except Exception as _e:
                valid = None
        if not valid:
            # Fallback -- keep behaviour deterministic even if a
            # param's probe is unreadable. The random-plan generator
            # falls back to the old regular grid, which still
            # exhibits the pre-v3.3 near-zero hit rate; the loud
            # log lets the operator see when the honest path was
            # taken.
            fallback_used = True
            fb_y = np.arange(-80, 81, 4)
            fb_z = np.arange(-80, 81, 4)
            valid = [(int(y), int(z))
                     for y in fb_y for z in fb_z
                     if (int(y), int(z)) not in per_ps_wake_set]
            print(f"  [random-plan] probe '{probe}' unavailable/unreadable "
                  f"for '{ps}'; falling back to regular step-4 grid "
                  f"(pre-v3.3 behaviour for this param).")
        ps_to_valid_coords[ps] = valid
        ps_to_allow_atlas_coords[ps] = allow_atlas_coords

    # Log a summary so the operator can see the random-plan pool size.
    pool_sizes = {ps: len(v) for ps, v in ps_to_valid_coords.items()}
    print(f"  [random-plan] valid non-wake (y,z) pool per param "
          f"(v3.4 = probe_yz - atlas_yz): {pool_sizes}")
    if fallback_used:
        print(f"  [random-plan] NOTE: at least one param used the pre-v3.3 "
              f"fallback grid -- expect elevated random-plan dropoff for it.")
    if atlas_probe_fallback_used:
        print("  [random-plan] NOTE: at least one param has no remaining "
              "probe_yz - atlas_yz pool, so atlas rows are being reused as "
              "random candidates for that param to preserve finite execution.")

    eligible_params = [ps for ps in param_list if ps_to_valid_coords.get(ps)]
    if not eligible_params and num_wake_plans > 0:
        raise SystemExit(
            "[random-plan] unable to build any random-plan candidate pool. "
            "Every param has zero admissible (y,z) candidates after applying "
            "probe/atlas fallbacks. Check the atlas coverage and source probe "
            "pickles before rerunning.")

    random_plans = []
    for _ in range(num_wake_plans):
        ps = random.choice(eligible_params)
        pool = ps_to_valid_coords[ps]
        y, z = random.choice(pool)
        if (not ps_to_allow_atlas_coords[ps]
                and (y, z) in set(atlas_yz_by_param.get(ps, []))):
            raise SystemExit(
                f"[random-plan] internal error: pooled candidate {(y, z)} for "
                f"'{ps}' is still atlas-covered. Pool construction is inconsistent.")
        w_idx = random.randint(0, WINDOWS_PER_COORD - 1)
        start_step = w_idx * NUM_TIME_NEW + 1
        random_plans.append((ps, y, z, start_step, False))
        param_to_coords[ps].add((y, z))
    return random_plans


def _collect_futures_with_timeout(futures_by_step, read_timeout_sec,
                                  wall_clock_deadline, run_started_at,
                                  phase_label):
    pending = set(futures_by_step.keys())
    results_by_step = {}
    while pending:
        _check_wall_clock_deadline(
            wall_clock_deadline,
            run_started_at,
            phase_label,
            detail=f"Pending futures={len(pending)}.")
        timeout_budget = float(read_timeout_sec)
        remaining = _remaining_wall_clock_seconds(wall_clock_deadline)
        if remaining is not None:
            timeout_budget = min(timeout_budget, max(0.0, remaining))
        done, pending = wait(
            pending,
            timeout=timeout_budget,
            return_when=FIRST_COMPLETED,
        )
        if not done:
            pending_steps = sorted(futures_by_step[f][0] for f in pending)
            pending_preview = pending_steps[:8]
            more = "" if len(pending_steps) <= len(pending_preview) else " ..."
            raise SystemExit(
                f"[timeout] no extraction worker finished within "
                f"{_format_timeout_seconds(timeout_budget)} during {phase_label}. "
                f"Likely stuck pickle read or hung worker. Pending steps for this "
                f"batch: {pending_preview}{more}. Increase --read-timeout-sec if "
                f"the dataset is merely slow, or inspect the corresponding .pkl.gz files.")
        for future in done:
            step, _path = futures_by_step[future]
            results_by_step[step] = future.result()
    return results_by_step


def _force_terminate_executor(executor, grace_sec=1.0):
    """Hard-stop a pool so hung workers cannot block process exit.

    ``ProcessPoolExecutor.shutdown(wait=True)`` joins the manager thread,
    which waits for in-flight tasks. A worker stuck in ``pd.read_pickle``
    therefore makes even a post-timeout ``with``-block exit hang forever.

    Snapshot worker process handles *before* ``shutdown()``: on current
    CPython, ``shutdown(wait=False, cancel_futures=True)`` can clear
    ``_processes`` while leaving already-running workers alive. We must
    keep those handles so terminate/kill still reaches them. Bound any
    manager-thread join so a later ``shutdown(wait=True)`` cannot
    deadlock the script.
    """
    # Snapshot first -- shutdown may empty executor._processes.
    processes = getattr(executor, "_processes", None)
    procs = list(processes.values()) if processes else []

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        # Very old Python without cancel_futures.
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
    except Exception:
        pass

    for proc in procs:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

    join_deadline = time.monotonic() + max(0.0, float(grace_sec))
    for proc in procs:
        try:
            proc.join(timeout=max(0.0, join_deadline - time.monotonic()))
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
        except Exception:
            pass

    # Bound the manager-thread join. If it still will not exit after workers
    # are gone, detach the reference so a subsequent shutdown(wait=True)
    # cannot hang the main process (e.g. SystemExit propagation).
    manager_join_timeout = max(0.1, float(grace_sec))
    for attr in ("_executor_manager_thread", "_queue_management_thread"):
        thread = getattr(executor, attr, None)
        if thread is None:
            continue
        try:
            if thread.is_alive():
                thread.join(timeout=manager_join_timeout)
        except Exception:
            pass
        try:
            if thread.is_alive():
                setattr(executor, attr, None)
        except Exception:
            pass
        break


def _stream_sequences_to_hdf5(out_path, sequence_iter, *, expected_plan_count,
                              wall_clock_deadline, run_started_at,
                              phase_label):
    """Incrementally write sequences so large runs stay bounded and visible."""
    temp_path = out_path + '.tmp'
    write_chunk = max(
        1,
        min(256, int(math.ceil(expected_plan_count / 100.0)) if expected_plan_count else 64)
    )
    progress_every = max(1, min(250, write_chunk))
    rows_written = 0

    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        with h5py.File(temp_path, 'w') as f_out:
            dataset = None
            for plan_idx, seq in sequence_iter:
                if dataset is None:
                    dataset = f_out.create_dataset(
                        'data',
                        shape=(0, NUM_TIME_NEW, NUM_X, 52),
                        maxshape=(None, NUM_TIME_NEW, NUM_X, 52),
                        chunks=(1, NUM_TIME_NEW, NUM_X, 52),
                        compression='gzip',
                        dtype='float32')

                if plan_idx == 1 or plan_idx % progress_every == 0:
                    _check_wall_clock_deadline(
                        wall_clock_deadline, run_started_at, phase_label,
                        detail=f" Written {rows_written}/{expected_plan_count} valid sequences so far.")
                    print(f"  [write-progress] {os.path.basename(out_path)}: "
                          f"planned={plan_idx}/{expected_plan_count}, written={rows_written}")

                dataset.resize((rows_written + 1, NUM_TIME_NEW, NUM_X, 52))
                dataset[rows_written] = seq
                rows_written += 1

                if rows_written % write_chunk == 0:
                    f_out.flush()

            if dataset is None:
                f_out.create_dataset(
                    'data',
                    shape=(0, NUM_TIME_NEW, NUM_X, 52),
                    maxshape=(None, NUM_TIME_NEW, NUM_X, 52),
                    chunks=(1, NUM_TIME_NEW, NUM_X, 52),
                    compression='gzip',
                    dtype='float32')
            f_out.flush()
        os.replace(temp_path, out_path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    return rows_written


def process_set(param_list, out_name, atlas_yz_by_param, atlas_meta,
                sample_percent=100.0, test_mode=False,
                read_timeout_sec=60.0, wall_clock_deadline=None,
                run_started_at=None):
    """Build one H5 file (train_80.h5 or val_80.h5).

    v3.4 (see OVERVIEW.md §15): `atlas_yz_by_param` is per-param and
    typically thousands of `(y_grid, z_grid)` rows drawn from the
    physics-derived wake atlas. Both the wake_plans population AND the
    random-plan exclusion set are derived from THIS SAME dict, so the
    atlas doubles as "seed these" for wake plans and "avoid these" for
    random plans -- exactly the operator's explicit ask.
    """
    print(f"\n🚀 Building {out_name}...")
    if run_started_at is None:
        run_started_at = time.monotonic()
    _check_wall_clock_deadline(
        wall_clock_deadline, run_started_at, f"initial planning for {out_name}")

    # 1. Determine all unique coordinates needed per parameter set
    param_to_coords = {ps: set() for ps in param_list}
    wake_plans = []

    if test_mode:
        print("Running in TEST MODE: only first sequence for each experiment.")
        for ps in param_list:
            seeds = atlas_yz_by_param.get(ps, [])
            if not seeds:
                continue
            y, z = seeds[0]
            start_step = 1
            wake_plans.append((ps, y, z, start_step, True))
            param_to_coords[ps].add((y, z))
    else:
        for ps in param_list:
            for y, z in atlas_yz_by_param.get(ps, []):
                param_to_coords[ps].add((y, z))
                for w_idx in range(WINDOWS_PER_COORD):
                    start_step = w_idx * NUM_TIME_NEW + 1
                    wake_plans.append((ps, y, z, start_step, True))

        if sample_percent < 100.0:
            sample_size = max(1, int(len(wake_plans) * (sample_percent / 100.0)))
            print(f"Sampling {sample_percent}% of wake sequences ({sample_size}/{len(wake_plans)})")
            wake_plans = random.sample(wake_plans, sample_size)
            # Re-evaluate which coordinates we actually need
            param_to_coords = {ps: set() for ps in param_list}
            for ps, y, z, _, _ in wake_plans:
                param_to_coords[ps].add((y, z))

    num_wake_plans = len(wake_plans)
    
    # Generate random plans 1-for-1
    #
    # v3.3 fix (see OVERVIEW.md §14): random-plan (y, z) coordinates are
    # now sampled from the SET OF (y, z) PAIRS ACTUALLY PRESENT in a
    # probe pickle, minus WAKE_COORDS. Previously v3/v3.1/v3.2 built a
    # regular step-4 grid over `[y_min, y_max] x [z_min, z_max]` from
    # the probe file, then `random.choice`d on it. That grid does NOT
    # match the irregular real-data (y, z) grid the OG pipeline emits,
    # so essentially every random pick landed on an (y, z) that did
    # NOT exist as a row in ANY pickle; `extract_from_file` returned
    # `None` for it, and the assembly loop then dropped the whole
    # 80-frame window via `assembly_missing_step`. Net effect: random
    # plans were populated 1-for-1 with wake plans, then ~100% dropped,
    # producing the ~50% "many fewer choices" symptom the operator
    # reported (documented in
    # tests/test_sequence_dropoff_diagnosis.py).
    #
    # By sampling from the probe pickle's real (y, z) set instead, a
    # random plan now targets a coord that is guaranteed to exist as
    # a row in the source data, so the extraction returns real
    # features and only the all-zero-latent trap (§13) or true source
    # gaps can still drop it. Random plans are still the intended
    # "negative examples away from the wake" -- they just aren't
    # silently discarded en masse anymore.
    random_plans = []
    if not test_mode:
        _check_wall_clock_deadline(
            wall_clock_deadline, run_started_at,
            f"random-plan pool construction for {out_name}")
        random_plans = _build_random_plans(
            param_list, atlas_yz_by_param, num_wake_plans, param_to_coords)

    print(f"Planned {num_wake_plans} wake sequences and {len(random_plans)} random sequences.")
    
    # 2. Extract data for each parameter set
    all_extracted_data = {} # (ps, y, z, step) -> features
    
    for ps in param_list:
        _check_wall_clock_deadline(
            wall_clock_deadline, run_started_at,
            f"extraction setup for {out_name}",
            detail=f" Current param='{ps}'.")
        coords = list(param_to_coords[ps])
        param_val = parse_param(ps)
        
        tasks = []
        if test_mode:
            # In test mode, we only need steps for the first sequence (1 to NUM_TIME_NEW)
            steps_needed = range(1, NUM_TIME_NEW + 1)
        else:
            steps_needed = range(1, TOTAL_TIMESTAMPS + 1)

        for step in steps_needed:
            f_path = get_file_path(ps, step)
            tasks.append((f_path, coords, param_val))
        
        # Avoid `with ProcessPoolExecutor()` so a timeout path can force-kill
        # stuck workers before any shutdown(wait=True) join. The context
        # manager always ends in wait=True and can hang on a deadlocked
        # pickle reader even after cancel_futures.
        executor = ProcessPoolExecutor()
        force_closed = False
        try:
            futures_by_step = {
                executor.submit(extract_from_file, task): (step, task[0])
                for step, task in zip(steps_needed, tasks)
            }
            results_by_step = _collect_futures_with_timeout(
                futures_by_step,
                read_timeout_sec=read_timeout_sec,
                wall_clock_deadline=wall_clock_deadline,
                run_started_at=run_started_at,
                phase_label=f"reading {ps} for {out_name}",
            )
        except BaseException:
            _force_terminate_executor(executor)
            force_closed = True
            raise
        finally:
            if not force_closed:
                executor.shutdown(wait=True)
        results = [results_by_step[step] for step in steps_needed]

        # Per-parameter skip accounting -- log an explicit summary of *why*
        # any given timestep contributed nothing. Silent skips previously
        # let a missing-on-disk parameter folder (e.g. 4p4) look like a
        # successful high-throughput read.
        n_missing = 0
        n_errors = 0
        n_empty_coord = 0  # coord requested but no matching (y,z) row in file
        first_error = None
        first_missing = None
        for step_idx, step_results in enumerate(results):
            step = steps_needed[step_idx]
            if step_results is None:
                # Legacy path: shouldn't happen after the return-dict change,
                # but treat as unknown skip.
                n_missing += 1
                continue
            if '__missing__' in step_results:
                n_missing += 1
                if first_missing is None:
                    first_missing = step_results['__missing__']
                continue
            if '__error__' in step_results:
                n_errors += 1
                if first_error is None:
                    first_error = step_results['__error__']
                continue
            for (y, z), data in step_results.items():
                if data is None:
                    n_empty_coord += 1
                    continue
                all_extracted_data[(ps, y, z, step)] = data

        total_steps = len(results)
        n_ok = total_steps - n_missing - n_errors
        print(f"  [{ps}] steps ok={n_ok}/{total_steps}  "
              f"missing_files={n_missing}  read_errors={n_errors}  "
              f"coord_not_in_file={n_empty_coord}")
        if n_missing:
            src_dir = os.path.join(SOURCE_ROOT, ps)
            src_exists = os.path.isdir(src_dir)
            print(f"    -> first missing: {first_missing}")
            print(f"    -> source folder '{src_dir}' exists={src_exists}")
            if not src_exists:
                print(f"    -> HINT: entire parameter folder is absent on disk; "
                      f"every step for '{ps}' will be skipped. Unzip "
                      f"'{ps}.zip' or drop '{ps}' from the split lists.")
        if n_errors:
            print(f"    -> first error: {first_error}")

    # 3. Assemble sequences and Count by experiment
    print(f"Assembling sequences for {out_name}...")
    all_plans = wake_plans + random_plans
    random.shuffle(all_plans)
    
    counts = Counter()
    
    # Accounting for assembly-time skips so the console explains any
    # gap between planned sequences and the final saved count.
    skip_reasons = Counter()
    first_missing_seq = {}
    def _iter_valid_sequences():
        for plan_idx, (ps, y_val, z_val, start_step, is_wake) in enumerate(
                tqdm(all_plans, desc="Assembling"), start=1):
            if plan_idx == 1 or plan_idx % 250 == 0:
                _check_wall_clock_deadline(
                    wall_clock_deadline, run_started_at,
                    f"assembly for {out_name}",
                    detail=f" Processed {plan_idx - 1}/{len(all_plans)} plans.")
                print(f"  [assembly-progress] {out_name}: processed "
                      f"{plan_idx - 1}/{len(all_plans)} plans, kept={sum(counts.values())}")
            seq = np.zeros((NUM_TIME_NEW, NUM_X, 52), dtype='float32')
            valid_seq = True
            for t_offset in range(NUM_TIME_NEW):
                step = start_step + t_offset
                data = all_extracted_data.get((ps, y_val, z_val, step))
                if data is None:
                    # Two distinct causes collapse here (either the source .pkl.gz
                    # for this step was missing/errored during extraction, or the
                    # (y,z) simply wasn't present in that file). Both count as
                    # 'assembly_missing_step' but we log the first offender per
                    # param so it's traceable.
                    skip_reasons[f'assembly_missing_step:{ps}'] += 1
                    first_missing_seq.setdefault(
                        ps, f"(y={y_val}, z={z_val}, start_step={start_step}, missing_step={step})")
                    valid_seq = False
                    break

                # Check for non-trivial data (not just zeros in latent dimensions).
                # Latents are columns 0:47.
                #
                # THE ALL-ZERO-BREAK TRAP (see OVERVIEW.md §13 for the full write-up):
                #
                #   A sequence is 80 consecutive frames at a single (param, y, z).
                #   If ANY ONE of those 80 frames has an all-zero latent block
                #   across ALL kept x-samples, `break` discards the entire window
                #   -- not just the offending frame. One decayed frame kills 80.
                #
                #   This is more likely to fire under v3.1's `NUM_X=10` window
                #   (|x|<=20) than under the pre-v3.1 `NUM_X=26` window, because
                #   the all-zero check is over the whole `(NUM_X, 47)` block:
                #   fewer kept x-samples => fewer chances for at least one row
                #   to be non-zero => the check trips for more (param, y, z, step).
                #   Far-downstream wake taps whose signal lives at x >= 21 in some
                #   frames are the population most affected.
                #
                #   Alternatives (not yet applied -- would require a v3.3 bump):
                #     - zero-fill this one frame and `continue` instead of `break`
                #     - threshold-based drop (only discard if >K frames are zero)
                #     - coord-level pre-filter before the plan phase
                #   All three trade different things; see OVERVIEW.md §13.5.
                if np.all(data[:, :47] == 0):
                    print(f"  [skip] {ps} @ (y={y_val}, z={z_val}) step={step}: "
                          f"all-zero latents (encoder produced no signal for this coord/step).")
                    valid_seq = False
                    break

                step_data = data.copy()
                step_data[:, 50] = float(t_offset)
                seq[t_offset] = step_data

            if valid_seq:
                counts[ps] += 1
                yield plan_idx, seq

    total_planned_per_ps = Counter(p[0] for p in all_plans)
    # 4. Save to HDF5 while assembly is still producing sequences, so we never
    # materialize the entire dataset in memory or disappear into one huge
    # `np.array(...)` + compressed create_dataset(...) block.
    _check_wall_clock_deadline(
        wall_clock_deadline, run_started_at, f"HDF5 write for {out_name}",
        detail=f" Starting streamed save for {len(all_plans)} planned sequences.")
    print(f"Saving to {out_name}...")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    num_sequences = _stream_sequences_to_hdf5(
        out_path, _iter_valid_sequences(),
        expected_plan_count=len(all_plans),
        wall_clock_deadline=wall_clock_deadline,
        run_started_at=run_started_at,
        phase_label=f"HDF5 write for {out_name}")

    # 5. Report counts + explicit skip breakdown so the difference between
    # `Planned` and `Final` is never a mystery.
    print(f"\n📊 Sequence counts for {out_name} by experiment:")
    for ps in sorted(param_list):
        planned = total_planned_per_ps.get(ps, 0)
        kept = counts[ps]
        skipped = planned - kept
        line = f"  - {ps}: {kept} sequences (planned={planned}, skipped={skipped})"
        if planned > 0 and kept == 0:
            line += "  <-- ZERO SEQUENCES; check source folder / prior [ps] skip summary above"
        print(line)
    if skip_reasons:
        print(f"  [skip-reasons] {dict(skip_reasons)}")
        for ps, where in first_missing_seq.items():
            print(f"    first missing for {ps}: {where}")

    with h5py.File(out_path, 'a') as f_out:
        # Add metadata
        f_out.attrs['source_root'] = SOURCE_ROOT
        f_out.attrs['sample_percent'] = sample_percent
        f_out.attrs['num_sequences'] = num_sequences
        f_out.attrs['creation_time'] = time.ctime()
        f_out.attrs['param_list'] = [p.encode('utf-8') for p in param_list]
        # Self-documenting layout attributes.
        #
        # The 'data' dataset is written as a 4-D array with shape
        #   (N_sequences, NUM_TIME, NUM_X, INPUT_DIM)
        # i.e. one row per sequence, then time, then spatial index along the
        # x-axis, then per-token features. Downstream readers historically had
        # to branch on `data.ndim` to figure out which layout they were
        # looking at (some older writers flattened (NUM_TIME, NUM_X) into a
        # single 2080-token axis). These attrs remove that ambiguity so any
        # consumer -- the trainer's TransformerDataset, downstream H5
        # readers, plotting scripts -- can read `f.attrs['layout']`
        # instead of inferring from `data.ndim`.
        #
        # Feature column layout (INPUT_DIM=52):
        #   [0:47]  -- 47-dim GEN3 AttentionSE latent (columns 'latent*')
        #   [47]    -- x coordinate (float32)
        #   [48]    -- y coordinate (int32 stored as float32 slot)
        #   [49]    -- z coordinate (int32 stored as float32 slot)
        #   [50]    -- t_idx within the sequence (0..NUM_TIME-1)
        #   [51]    -- param value (float32)
        f_out.attrs['layout'] = 'N_NT_NX_C'
        f_out.attrs['num_time'] = NUM_TIME_NEW
        f_out.attrs['num_x'] = NUM_X
        f_out.attrs['input_dim'] = 52
        f_out.attrs['latent_dim'] = 47
        f_out.attrs['total_timestamps'] = TOTAL_TIMESTAMPS
        f_out.attrs['windows_per_coord'] = WINDOWS_PER_COORD
        # v3 split provenance -- see the "Held-out validation split -- v3
        # convention" comment block in prepare_data(). Consumers that
        # report per-parameter metrics should read `param_list` above and
        # `split_version` to know which val cases are encoder-blinded
        # (`6p4`) vs. transformer-only-held-out (`3p6`).
        # split_version bumps with every documented policy revision to this
        # file, even when the on-disk bytes are unchanged, so downstream
        # consumers can key their audits to a specific OVERVIEW.md section:
        #   'v3'   -> §11: 3p6 replaces the unusable 4p4 in the val split
        #   'v3.1' -> §12: x-window restricted to |x|<=20, 80-frame-only
        #   'v3.2' -> §13: all-zero-break trap documented; dropoff discussion
        #   'v3.3' -> §14: random-plan (y,z) sampled from real data grid
        #                  (fixes the ~50% dropoff traced to random plans
        #                  hitting a regular grid that misses the
        #                  irregular real-data grid). On-disk bytes DIFFER
        #                  from v3.2 for this reason -- regenerate.
        #   'v3.4' -> §15: WAKE_COORDS retired; wake seeds now come from
        #                  the physics-derived atlas built by
        #                  build_wake_atlas.py, and the atlas ALSO drives
        #                  the random-plan exclusion set (probe_yz -
        #                  atlas_yz). Bytes DIFFER from v3.3.
        #   'v3.5' -> OVERVIEW.md §16: 11p4 moves train -> val, giving val
        #                  a high-side bracket alongside 3p6 (low) and 6p4
        #                  (mid). Bytes DIFFER from v3.4 (different
        #                  train/val param membership -> different plans).
        f_out.attrs['split_version'] = 'v3.5'
        # v3.4 wake-atlas provenance -- see OVERVIEW.md §15. These attrs
        # let a downstream consumer audit which atlas was used to build
        # this H5. `wake_atlas_source` is the absolute path (or the
        # sentinel 'legacy:...' if the opt-in fallback fired),
        # `wake_atlas_sha256` is the atlas file's sha (or 'legacy' /
        # 'shard-tree-in-memory-merge'), and `wake_atlas_rows` is a
        # JSON dict {param: unique_(y_grid,z_grid) count}.
        f_out.attrs['wake_atlas_source'] = atlas_meta.get(
            'wake_atlas_source', '')
        f_out.attrs['wake_atlas_sha256'] = atlas_meta.get(
            'wake_atlas_sha256', '')
        f_out.attrs['wake_atlas_rows'] = atlas_meta.get(
            'wake_atlas_rows', '{}')
    print(f"Final {out_name} size: {num_sequences} sequences.")

def prepare_data(sample_percent=100.0, test_mode=False,
                 read_timeout_sec=60.0, wall_clock_sec=7200.0):
    t0 = time.time()
    run_started_at = time.monotonic()
    wall_clock_deadline = _make_wall_clock_deadline(wall_clock_sec)

    # v3.4: wake seeds are loaded per-param from the physics-derived
    # atlas (see OVERVIEW.md §15). Legacy 24-tap literal is only used
    # as a diagnostic fallback via PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK=1.
    print("[wake-atlas] resolving physics-derived wake seed atlas ...")

    # ----------------------------------------------------------------
    # Held-out validation split -- v3 convention.
    #
    # PAPER / ENCODER CONVENTION (context for what changed):
    #
    #   encoder_neurIPS/build_neurIPS_dataset.py trains the GEN3 AttentionSE
    #   autoencoder with `excluded_from_train=["4p4", "6p4"]`:
    #
    #       4p4 -> 6.9 m/s  (low-side extrapolation, encoder-blinded)
    #       6p4 -> 10.0 m/s (mid-range interpolation, encoder-blinded)
    #
    #   These are the only two parameter sets whose 47-dim latents are
    #   *fully* out-of-distribution to the encoder -- i.e. both the encoder
    #   AND the transformer are honestly held-out on them.
    #
    # WHY THIS FILE DEPARTS FROM THAT CONVENTION IN v3:
    #
    #   The raw 4p4 acquisition is a partial/incomplete recording. The
    #   source pickle in /Users/kkreth/PycharmProjects/data/Unmodified_OG_Data/
    #   is deliberately renamed `4p4.notusing.gz` (not `4p4.pkl.gz`) to opt
    #   it out of every downstream `*.pkl.gz` glob; the OG data-prep chain
    #   (Ordered_030 -> ... -> Ordered_200) therefore never emits a
    #   `Final_Cubed_OG_Data_wLatent/4p4/` folder for this trainer to read.
    #   Trying to build a val split that includes `4p4` produces a folder
    #   with zero usable sequences, which our per-parameter skip logging
    #   in process_set() will flag loudly (`ZERO SEQUENCES; check source
    #   folder`).
    #
    #   Rather than ship a broken val set, v3 replaces the missing `4p4`
    #   with `3p6` (5.6 m/s) as the SECOND validation case, keeping `6p4`
    #   as the first. This is an HONEST substitution, not a
    #   convention-preserving one -- see the caveat below.
    #
    # HONESTY CAVEAT (must be preserved when reporting):
    #
    #   - `6p4` is held out from BOTH the encoder and the transformer, so
    #     any metric measured on `6p4` is a genuine encoder+transformer
    #     out-of-distribution result. This matches the §3.5 vortex-reversal
    #     validation in the paper.
    #
    #   - `3p6` was SEEN BY THE ENCODER during its training (it is in the
    #     encoder's inclusion set, not `excluded_from_train`). It is held
    #     out ONLY from the transformer here. Therefore any metric on
    #     `3p6` measures transformer generalisation over latents the
    #     encoder is already comfortable with -- a strictly weaker OOD
    #     statement than `6p4` gives, and NOT comparable to the paper's
    #     4p4 numbers even though it plays the same low-side role
    #     (5.6 m/s < 6.9 m/s < training range 7.2-17.8 m/s).
    #
    # REPORTING GUIDANCE (see OVERVIEW.md v3 section):
    #
    #   Metrics should be reported per-parameter (`3p6` and `6p4`
    #   separately) AND as a simple mean of the two, so a reader can
    #   attribute error to either the low-side transformer-only
    #   extrapolation or the mid-range encoder+transformer OOD case,
    #   rather than reading a single averaged number that hides the
    #   asymmetry. Averaging is unweighted because the two val sets are
    #   the same size after this file runs (identical wake+random plan
    #   counts per param).
    #
    #   ADDING 11p4 (v3.5): `11p4` (17.8 m/s) is the highest-Reynolds case
    #   in the corpus. Moving it from train to val gives a HIGH-side
    #   companion to `3p6`'s low-side and `6p4`'s mid-range, so the val
    #   split now brackets the extremes of the speed sweep. Same honesty
    #   caveat as `3p6`: the encoder SAW `11p4` during its training (it is
    #   NOT in `excluded_from_train`), so any `11p4` metric is a
    #   transformer-only OOD statement, not an encoder+transformer one.
    #
    # LINEAGE:
    #   v1   -> train had 4p4;  val = ["6p4"]                       (encoder-inconsistent)
    #   v2   -> train dropped 4p4; val = ["4p4", "6p4"]            (encoder-consistent but 4p4 raw file is partial and unusable)
    #   v3   -> train also drops 3p6; val = ["3p6", "6p4"]        (honest substitute for the unusable 4p4)
    #   v3.5 -> train also drops 11p4; val = ["3p6", "6p4", "11p4"] (current -- adds high-side bracket)
    # ----------------------------------------------------------------
    train_params = ["4p6", "5p2", "6p6", "7p2", "7p8", "8p4", "10p4"]
    val_params = ["3p6", "6p4", "11p4"]

    # Resolve the atlas once per split so the source/sha/rows are
    # honestly per-split; train and val use disjoint param lists.
    train_atlas, train_meta = resolve_wake_atlas_for_params(train_params)
    val_atlas, val_meta = resolve_wake_atlas_for_params(val_params)

    train_out = f"train_{NUM_TIME_NEW}.h5"
    val_out = f"val_{NUM_TIME_NEW}.h5"
    process_set(train_params, train_out, train_atlas, train_meta,
                sample_percent, test_mode,
                read_timeout_sec=read_timeout_sec,
                wall_clock_deadline=wall_clock_deadline,
                run_started_at=run_started_at)
    process_set(val_params, val_out, val_atlas, val_meta,
                sample_percent, test_mode,
                read_timeout_sec=read_timeout_sec,
                wall_clock_deadline=wall_clock_deadline,
                run_started_at=run_started_at)
    
    print(f"\n✅ Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=100.0, help="Percentage of data to sample (0-100)")
    parser.add_argument("--test", action="store_true", help="Test mode: only first sequence for each experiment")
    # v3 policy: 80-frame ONLY. The --num-time flag is retained for backward
    # compatibility with any wrapper script that still passes it, but the
    # only accepted value is 80; anything else is rejected loudly so a stale
    # `--num-time 40` invocation cannot silently regenerate legacy data.
    parser.add_argument("--num-time", type=int, default=80,
                        help="Sequence length in frames. v3 accepts ONLY 80 "
                             "(12 context + 68 forecast @ 120 Hz). The v1.0 "
                             "40-frame path is retired; passing any other "
                             "value is a hard error.")
    parser.add_argument("--read-timeout-sec", type=float, default=60.0,
                        help="Fail a per-param extraction batch if no worker "
                             "finishes within this many seconds. Prevents a "
                             "single stuck pickle read from blocking forever.")
    parser.add_argument("--wall-clock-sec", type=float, default=7200.0,
                        help="Overall wall-clock limit for the full train+val "
                             "regeneration run. Pass 0 to disable.")
    args = parser.parse_args()

    if args.num_time != 80:
        raise SystemExit(
            f"--num-time={args.num_time} is not supported: v3 produces ONLY "
            f"80-frame data (train_80.h5 / val_80.h5). Rerun without the flag "
            f"or with `--num-time 80`."
        )
    NUM_TIME_NEW = 80
    WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW
    if args.read_timeout_sec <= 0:
        raise SystemExit("--read-timeout-sec must be > 0 so extraction cannot hang forever.")
    if args.wall_clock_sec < 0:
        raise SystemExit("--wall-clock-sec must be >= 0 (use 0 to disable the overall limit).")
    print(f"NUM_TIME_NEW={NUM_TIME_NEW}, NUM_X={NUM_X}, "
          f"WINDOWS_PER_COORD={WINDOWS_PER_COORD}, "
          f"X_COORDS=[{X_COORDS.min():.0f}..{X_COORDS.max():.0f}] ({len(X_COORDS)} pts)")
    print(f"Runtime guards: read_timeout_sec={args.read_timeout_sec}, "
          f"wall_clock_sec={args.wall_clock_sec}")
    prepare_data(args.sample, args.test,
                 read_timeout_sec=args.read_timeout_sec,
                 wall_clock_sec=args.wall_clock_sec)
