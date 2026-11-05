"""
build_wake_atlas.py -- restart-safe, multi-process wake atlas builder.

Sweeps every (param, step) in the OG data tree, computes a 3-D
vorticity field, extracts the >=90%-of-peak core region inside the
interaction zone |x|<=30, and emits one row per core point PLUS
spatial sliding-window neighbours around each core (dy, dz offsets
snapped to the probe pickle's real (y, z) grid).

Each (param, step) writes to its own shard file under
    transformer_neurIPS/data/wake_atlas_shards/<param>/step_<NNNN>.parquet
    (or .csv.gz fallback if pyarrow is unavailable)
so a mid-run kill loses at most one shard. Re-launching the script is
a no-op for any (param, step) whose shard already exists and matches
the current SCHEMA_VER.

The physics kernel (`find_vortex_core_region`) is a self-contained copy
of `vorticity_search.find_vortex_core_region` so this module stays
inside `transformer_neurIPS/` per the scope invariant enforced by
tests/test_changes_scoped_to_transformer_neurips.py.

CLI reference:

    python transformer_neurIPS/build_wake_atlas.py                 # default sweep, all params, all steps
    python transformer_neurIPS/build_wake_atlas.py --params 6p4    # single param
    python transformer_neurIPS/build_wake_atlas.py --steps 1:101   # steps 1..100
    python transformer_neurIPS/build_wake_atlas.py --workers 8
    python transformer_neurIPS/build_wake_atlas.py --force         # recompute all shards
    python transformer_neurIPS/build_wake_atlas.py --merge         # merge shards into wake_atlas.csv.gz

See OVERVIEW.md §15 for the full provenance chain.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------
SCHEMA_VER = 1

DEFAULT_SOURCE_ROOT = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data"
DEFAULT_SHARDS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wake_atlas_shards")
DEFAULT_OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wake_atlas.csv.gz")
DEFAULT_SHA256 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wake_atlas.sha256")
DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "wake_atlas.manifest.json")

ALL_PARAMS = ["3p6", "4p4", "4p6", "5p2", "6p4", "6p6",
              "7p2", "7p8", "8p4", "10p4", "11p4"]

# Column order for the shard schema (also used by the merged atlas).
SHARD_COLUMNS = [
    "schema_ver", "param", "step", "core_id", "is_neighbour",
    "dy", "dz", "x", "y", "z", "cx", "cy", "cz",
    "y_grid", "z_grid", "vort_mag",
    "omega_x", "omega_y", "omega_z",
    "peak_val", "n_core_points",
]


# --------------------------------------------------------------------------
# Parquet availability check
# --------------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
    _SHARD_EXT = ".parquet"
except Exception:
    _HAS_PARQUET = False
    _SHARD_EXT = ".csv.gz"


def _read_shard(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_shard(df: pd.DataFrame, path: str) -> None:
    """Atomic write: <path>.tmp -> os.replace(<path>)."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Physics kernel (copy of vorticity_search.find_vortex_core_region,
# trimmed for atlas needs: we do NOT need the interpolated omega at the
# centroid, and we do NOT round(cx, cy, cz) so consumers see the raw
# weighted centroid).
# --------------------------------------------------------------------------
def find_vortex_core_region(df: pd.DataFrame,
                            x_range: Tuple[float, float] = (-30.0, 30.0),
                            threshold: float = 0.90) -> Optional[dict]:
    """Return dict with peak_val, n_core_points, cx/cy/cz, and a list of
    per-core-point records `{x, y, z, vort_mag, omega_x, omega_y, omega_z}`
    for every grid point at >=threshold*peak inside the interaction zone.
    Returns None if the dataframe cannot be pivoted (missing velocity cols).
    """
    if not {"vx", "vy", "vz", "x", "y", "z"}.issubset(df.columns):
        return None

    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())
    zs = np.sort(df["z"].unique())
    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx < 2 or ny < 2 or nz < 2:
        return None

    try:
        pivot_vx = df.pivot(index="x", columns=["y", "z"], values="vx").values.reshape(nx, ny, nz)
        pivot_vy = df.pivot(index="x", columns=["y", "z"], values="vy").values.reshape(nx, ny, nz)
        pivot_vz = df.pivot(index="x", columns=["y", "z"], values="vz").values.reshape(nx, ny, nz)
    except Exception:
        return None

    dvy_dx = np.gradient(pivot_vy, xs, axis=0)
    dvx_dy = np.gradient(pivot_vx, ys, axis=1)
    omega_z = dvy_dx - dvx_dy

    dvz_dy = np.gradient(pivot_vz, ys, axis=1)
    dvy_dz = np.gradient(pivot_vy, zs, axis=2)
    omega_x = dvz_dy - dvy_dz

    dvx_dz = np.gradient(pivot_vx, zs, axis=2)
    dvz_dx = np.gradient(pivot_vz, xs, axis=0)
    omega_y = dvx_dz - dvz_dx

    mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    x_mask = (xs >= x_range[0]) & (xs <= x_range[1])
    xs_sub = xs[x_mask]
    if xs_sub.size == 0:
        return None
    mag_sub = mag[x_mask, :, :]
    ox_sub = omega_x[x_mask, :, :]
    oy_sub = omega_y[x_mask, :, :]
    oz_sub = omega_z[x_mask, :, :]

    peak_val = float(mag_sub.max())
    if not np.isfinite(peak_val) or peak_val <= 0.0:
        return None

    core_mask = mag_sub >= threshold * peak_val
    core_idxs = np.argwhere(core_mask)
    n_points = int(len(core_idxs))
    if n_points == 0:
        return None

    core_xs = xs_sub[core_idxs[:, 0]]
    core_ys = ys[core_idxs[:, 1]]
    core_zs = zs[core_idxs[:, 2]]
    weights = mag_sub[core_mask]
    w_sum = float(weights.sum())
    cx = float(np.sum(core_xs * weights) / w_sum)
    cy = float(np.sum(core_ys * weights) / w_sum)
    cz = float(np.sum(core_zs * weights) / w_sum)

    core_points = []
    for i in range(n_points):
        ix, iy, iz = core_idxs[i]
        core_points.append({
            "x": float(core_xs[i]),
            "y": float(core_ys[i]),
            "z": float(core_zs[i]),
            "vort_mag": float(weights[i]),
            "omega_x": float(ox_sub[ix, iy, iz]),
            "omega_y": float(oy_sub[ix, iy, iz]),
            "omega_z": float(oz_sub[ix, iy, iz]),
        })

    return {
        "peak_val": peak_val,
        "n_core_points": n_points,
        "cx": cx, "cy": cy, "cz": cz,
        "core_points": core_points,
        "y_grid": [int(v) for v in ys.tolist()],
        "z_grid": [int(v) for v in zs.tolist()],
    }


# --------------------------------------------------------------------------
# Neighbour math
# --------------------------------------------------------------------------
def compute_neighbour_offsets(radius_yz: int, stride_yz: int) -> List[Tuple[int, int]]:
    """Return the (dy, dz) offsets on a `stride_yz`-spaced grid inside the
    L-inf ball of radius `radius_yz`. (0, 0) is included as the core row.
    Deterministic order: sorted by (dy, dz).
    """
    steps = list(range(-radius_yz, radius_yz + 1, stride_yz))
    if 0 not in steps:
        # ensure the core (0,0) sits on the grid
        steps = sorted(set(steps + [0]))
    offsets = [(dy, dz) for dy in steps for dz in steps
               if abs(dy) <= radius_yz and abs(dz) <= radius_yz]
    offsets.sort()
    return offsets


def snap_to_grid(val: float, grid_sorted: List[int]) -> int:
    """Nearest-tap snap; on ties, prefer the smaller-|value| tap."""
    arr = np.asarray(grid_sorted, dtype=np.int64)
    diffs = np.abs(arr - val)
    best = int(diffs.min())
    cands = arr[diffs == best]
    # tie-break: smaller magnitude wins
    return int(cands[np.argmin(np.abs(cands))])


# --------------------------------------------------------------------------
# Per-(param, step) worker
# --------------------------------------------------------------------------
def _load_pickle(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _rows_for_step(param: str, step: int, df: pd.DataFrame,
                   neighbour_offsets: List[Tuple[int, int]]) -> Optional[pd.DataFrame]:
    """Compute a shard DataFrame for one (param, step). Returns None on
    unrecoverable failure (missing velocity columns, degenerate grid, no
    core points, etc.). Callers should still write an EMPTY shard in that
    case so the restart-protocol treats the step as done.
    """
    result = find_vortex_core_region(df)
    if result is None:
        return None

    y_grid_all = result["y_grid"]
    z_grid_all = result["z_grid"]
    y_grid_set = set(y_grid_all)
    z_grid_set = set(z_grid_all)

    rows: List[dict] = []
    peak_val = result["peak_val"]
    n_core = result["n_core_points"]
    cx, cy, cz = result["cx"], result["cy"], result["cz"]

    for core_id, cp in enumerate(result["core_points"]):
        core_y_snap = snap_to_grid(cp["y"], y_grid_all)
        core_z_snap = snap_to_grid(cp["z"], z_grid_all)
        for (dy, dz) in neighbour_offsets:
            yg = core_y_snap + dy
            zg = core_z_snap + dz
            # Skip neighbours that fall off the real (y,z) grid.
            if yg not in y_grid_set or zg not in z_grid_set:
                continue
            is_neighbour = not (dy == 0 and dz == 0)
            rows.append({
                "schema_ver": SCHEMA_VER,
                "param": param,
                "step": int(step),
                "core_id": int(core_id),
                "is_neighbour": bool(is_neighbour),
                "dy": int(dy),
                "dz": int(dz),
                "x": float(cp["x"]),
                "y": float(cp["y"]),
                "z": float(cp["z"]),
                "cx": float(cx),
                "cy": float(cy),
                "cz": float(cz),
                "y_grid": int(yg),
                "z_grid": int(zg),
                "vort_mag": float(cp["vort_mag"]) if not is_neighbour else float("nan"),
                "omega_x": float(cp["omega_x"]) if not is_neighbour else float("nan"),
                "omega_y": float(cp["omega_y"]) if not is_neighbour else float("nan"),
                "omega_z": float(cp["omega_z"]) if not is_neighbour else float("nan"),
                "peak_val": float(peak_val),
                "n_core_points": int(n_core),
            })

    if not rows:
        return None
    return pd.DataFrame(rows, columns=SHARD_COLUMNS)


def _process_one(task: Tuple[str, int, str, str, List[Tuple[int, int]]]) -> dict:
    """Worker entry-point. task=(param, step, source_root, shard_path, offsets).

    Returns a dict describing what happened:
        {'param': ..., 'step': ..., 'status': 'written' | 'skipped_existing'
         | 'empty' | 'missing_file' | 'error', 'n_rows': int, 'msg': ...}
    """
    param, step, source_root, shard_path, offsets = task

    # Restart check happens in the parent too; re-check here to avoid a
    # tiny TOCTOU race across worker restarts.
    if os.path.exists(shard_path):
        try:
            head = _read_shard(shard_path).head(1)
            if len(head) == 0 or int(head.iloc[0]["schema_ver"]) == SCHEMA_VER:
                return {"param": param, "step": step,
                        "status": "skipped_existing", "n_rows": -1, "msg": ""}
        except Exception:
            # Corrupt shard -- recompute.
            pass

    pkl_path = os.path.join(source_root, param, f"{step:04d}.pkl.gz")
    df = _load_pickle(pkl_path)
    if df is None:
        # write an empty marker shard so restart skips this step next time
        empty = pd.DataFrame(columns=SHARD_COLUMNS)
        # ensure schema_ver row exists so restart check doesn't recompute
        empty = pd.concat([
            empty,
            pd.DataFrame([{c: (SCHEMA_VER if c == "schema_ver"
                               else (param if c == "param"
                                     else (int(step) if c == "step" else np.nan)))
                            for c in SHARD_COLUMNS}])
        ], ignore_index=True).iloc[0:0]
        # Trick above kept the schema; simpler: write a 1-row sentinel with
        # is_neighbour=True and n_core_points=0 as an empty marker.
        sentinel = pd.DataFrame([{
            "schema_ver": SCHEMA_VER, "param": param, "step": int(step),
            "core_id": -1, "is_neighbour": True, "dy": 0, "dz": 0,
            "x": float("nan"), "y": float("nan"), "z": float("nan"),
            "cx": float("nan"), "cy": float("nan"), "cz": float("nan"),
            "y_grid": -999999, "z_grid": -999999,
            "vort_mag": float("nan"), "omega_x": float("nan"),
            "omega_y": float("nan"), "omega_z": float("nan"),
            "peak_val": float("nan"), "n_core_points": 0,
        }], columns=SHARD_COLUMNS)
        _write_shard(sentinel, shard_path)
        return {"param": param, "step": step,
                "status": "missing_file", "n_rows": 0, "msg": pkl_path}

    shard_df = _rows_for_step(param, step, df, offsets)
    if shard_df is None or len(shard_df) == 0:
        # empty sentinel (see above)
        sentinel = pd.DataFrame([{
            "schema_ver": SCHEMA_VER, "param": param, "step": int(step),
            "core_id": -1, "is_neighbour": True, "dy": 0, "dz": 0,
            "x": float("nan"), "y": float("nan"), "z": float("nan"),
            "cx": float("nan"), "cy": float("nan"), "cz": float("nan"),
            "y_grid": -999999, "z_grid": -999999,
            "vort_mag": float("nan"), "omega_x": float("nan"),
            "omega_y": float("nan"), "omega_z": float("nan"),
            "peak_val": float("nan"), "n_core_points": 0,
        }], columns=SHARD_COLUMNS)
        _write_shard(sentinel, shard_path)
        return {"param": param, "step": step,
                "status": "empty", "n_rows": 0, "msg": ""}

    _write_shard(shard_df, shard_path)
    return {"param": param, "step": step, "status": "written",
            "n_rows": int(len(shard_df)), "msg": ""}


# --------------------------------------------------------------------------
# Sweep orchestration
# --------------------------------------------------------------------------
def shard_path_for(shards_dir: str, param: str, step: int) -> str:
    return os.path.join(shards_dir, param, f"step_{step:04d}{_SHARD_EXT}")


def build_sweep(params: List[str],
                steps: List[int],
                source_root: str,
                shards_dir: str,
                radius_yz: int,
                stride_yz: int,
                workers: int,
                force: bool,
                log=print) -> Dict[str, dict]:
    """Run the sweep. Returns a summary dict {param: {counters}}."""
    offsets = compute_neighbour_offsets(radius_yz, stride_yz)
    log(f"[atlas] shard codec: {'parquet' if _HAS_PARQUET else 'csv.gz'} "
        f"(pyarrow_available={_HAS_PARQUET})")
    log(f"[atlas] neighbour offsets: {len(offsets)} points "
        f"(radius={radius_yz}, stride={stride_yz})")

    # Filter missing param folders (loud warning, write _MISSING marker)
    live_params: List[str] = []
    for p in params:
        pdir = os.path.join(source_root, p)
        if not os.path.isdir(pdir):
            log(f"[atlas] WARNING: param '{p}' source dir does not exist: "
                f"{pdir}; skipping. (Marker: _MISSING under shards dir.)")
            os.makedirs(os.path.join(shards_dir, p), exist_ok=True)
            with open(os.path.join(shards_dir, p, "_MISSING"), "w") as f:
                f.write(f"source folder absent as of {time.ctime()}: {pdir}\n")
            continue
        live_params.append(p)

    # Build task list (skip already-done shards unless --force)
    tasks: List[Tuple[str, int, str, str, List[Tuple[int, int]]]] = []
    skipped_existing = 0
    for p in live_params:
        os.makedirs(os.path.join(shards_dir, p), exist_ok=True)
        for s in steps:
            sp = shard_path_for(shards_dir, p, s)
            if os.path.exists(sp) and not force:
                skipped_existing += 1
                continue
            tasks.append((p, s, source_root, sp, offsets))
    log(f"[atlas] planned tasks: {len(tasks)} "
        f"(already-done skipped: {skipped_existing})")

    summary: Dict[str, dict] = {p: {"written": 0, "skipped_existing": 0,
                                     "empty": 0, "missing_file": 0,
                                     "rows": 0}
                                 for p in live_params}
    if not tasks:
        return summary

    n_done = 0
    log_every = max(1, len(tasks) // 40)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            p = res["param"]
            if p not in summary:
                summary[p] = {"written": 0, "skipped_existing": 0,
                              "empty": 0, "missing_file": 0, "rows": 0}
            summary[p][res["status"]] = summary[p].get(res["status"], 0) + 1
            if res["n_rows"] > 0:
                summary[p]["rows"] += res["n_rows"]
            n_done += 1
            if n_done % log_every == 0 or n_done == len(tasks):
                log(f"[atlas] progress: {n_done}/{len(tasks)} "
                    f"({100.0 * n_done / len(tasks):.1f}%)")

    for p in sorted(summary.keys()):
        s = summary[p]
        log(f"[atlas] {p}: written={s.get('written', 0)} "
            f"empty={s.get('empty', 0)} missing_file={s.get('missing_file', 0)} "
            f"rows={s.get('rows', 0)}")
    return summary


# --------------------------------------------------------------------------
# Merge mode
# --------------------------------------------------------------------------
def iter_shard_files(shards_dir: str) -> List[str]:
    out = []
    if not os.path.isdir(shards_dir):
        return out
    for p in sorted(os.listdir(shards_dir)):
        pdir = os.path.join(shards_dir, p)
        if not os.path.isdir(pdir):
            continue
        for f in sorted(os.listdir(pdir)):
            if f.startswith("step_") and (f.endswith(".parquet") or f.endswith(".csv.gz")):
                out.append(os.path.join(pdir, f))
    return out


def load_shard_tree(shards_dir: str, include_empty: bool = False) -> pd.DataFrame:
    """Load every shard file under shards_dir and concat. Empty-sentinel
    rows (n_core_points == 0) are dropped unless include_empty=True.
    """
    files = iter_shard_files(shards_dir)
    if not files:
        return pd.DataFrame(columns=SHARD_COLUMNS)
    frames = []
    for fp in files:
        try:
            df = _read_shard(fp)
        except Exception as e:
            print(f"[atlas] WARNING: failed to read shard {fp}: {e}")
            continue
        if not include_empty:
            df = df[df["n_core_points"] > 0]
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=SHARD_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out


def merge_and_write(shards_dir: str,
                    output_csv: str,
                    sha256_path: str,
                    manifest_path: str,
                    log=print) -> None:
    log(f"[atlas] merging shards under {shards_dir} ...")
    df = load_shard_tree(shards_dir, include_empty=False)
    if len(df) == 0:
        log(f"[atlas] merge: NO shards with rows found. "
            f"Refusing to overwrite {output_csv}.")
        return

    # Dedup by (param, step, y_grid, z_grid, is_neighbour)
    dedup_cols = ["param", "step", "y_grid", "z_grid", "is_neighbour"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    after = len(df)
    log(f"[atlas] merged rows: {before} -> {after} after dedup on {dedup_cols}")

    # Stable sort
    df = df.sort_values(
        ["param", "step", "core_id", "dy", "dz"],
        kind="mergesort"
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    tmp = output_csv + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, output_csv)
    log(f"[atlas] wrote {output_csv} ({os.path.getsize(output_csv) / 1e6:.2f} MB)")

    # sha256
    h = hashlib.sha256()
    with open(output_csv, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with open(sha256_path, "w") as f:
        f.write(f"{sha}  {os.path.basename(output_csv)}\n")
    log(f"[atlas] wrote {sha256_path}: {sha}")

    # manifest
    manifest: Dict[str, dict] = {}
    for p in sorted(df["param"].unique()):
        sub = df[df["param"] == p]
        manifest[p] = {
            "rows": int(len(sub)),
            "unique_yz": int(sub[["y_grid", "z_grid"]].drop_duplicates().shape[0]),
            "steps_with_cores": int(sub["step"].nunique()),
        }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    log(f"[atlas] wrote {manifest_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_steps(spec: str) -> List[int]:
    """`--steps 1:1201` -> range(1, 1201); `--steps 1,5,10` -> [1,5,10]."""
    if ":" in spec:
        a, b = spec.split(":", 1)
        return list(range(int(a), int(b)))
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_params(spec: str) -> List[str]:
    if spec.strip().lower() == "all":
        return list(ALL_PARAMS)
    return [x.strip() for x in spec.split(",") if x.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    p.add_argument("--params", type=str, default="all",
                   help="Comma-separated param list or 'all' (default: all)")
    p.add_argument("--steps", type=str, default="1:1201",
                   help="Step range 'a:b' (exclusive) or comma list (default: 1:1201)")
    p.add_argument("--workers", type=int,
                   default=min(os.cpu_count() or 4, 8),
                   help="ProcessPool workers for the vorticity kernel")
    p.add_argument("--io-workers", type=int, default=4,
                   help="ThreadPool workers for shard I/O (parent-side)")
    p.add_argument("--radius-yz", type=int, default=8,
                   help="Sliding-window radius (L-inf) in the (y, z) plane")
    p.add_argument("--stride-yz", type=int, default=4,
                   help="Sliding-window stride in the (y, z) plane")
    p.add_argument("--source-root", type=str, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--shards-dir", type=str, default=DEFAULT_SHARDS_DIR)
    p.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--sha256-path", type=str, default=DEFAULT_SHA256)
    p.add_argument("--manifest-path", type=str, default=DEFAULT_MANIFEST)
    p.add_argument("--merge", action="store_true",
                   help="After sweep (or standalone), merge shards to wake_atlas.csv.gz")
    p.add_argument("--merge-only", action="store_true",
                   help="Skip the sweep and only run the merge step")
    p.add_argument("--force", action="store_true",
                   help="Recompute every shard, ignoring existing files")
    args = p.parse_args(argv)

    params = _parse_params(args.params)
    steps = _parse_steps(args.steps)
    print(f"[atlas] params={params} steps={steps[0]}..{steps[-1]} "
          f"({len(steps)} total)  workers={args.workers}")

    if not args.merge_only:
        build_sweep(
            params=params, steps=steps,
            source_root=args.source_root,
            shards_dir=args.shards_dir,
            radius_yz=args.radius_yz,
            stride_yz=args.stride_yz,
            workers=args.workers,
            force=args.force,
        )

    if args.merge or args.merge_only:
        merge_and_write(
            shards_dir=args.shards_dir,
            output_csv=args.output_csv,
            sha256_path=args.sha256_path,
            manifest_path=args.manifest_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
