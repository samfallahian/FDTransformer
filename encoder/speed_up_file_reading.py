import os
import time
import argparse
import json
from typing import Dict, Any

# Ensure we can import from project root when running this script directly
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from EfficientDataLoader import EfficientDataLoader  # noqa: E402
from Ordered_001_Initialize import HostPreferences  # noqa: E402


def human_seconds(secs: float) -> str:
    return f"{secs:.3f}s"


def summarize_timings(title: str, profiling: Dict[str, Any]) -> str:
    timings = profiling.get("timings", {})
    # Order keys for readability
    order = [
        "cache_check_seconds",
        "list_files_seconds",
        "metadata_seconds",
        "_load_file_seconds_total",
        "_load_file_calls",
        "_sample_rows_seconds_total",
        "_sample_rows_calls",
        "get_batch_select_seconds_total",
        "get_batch_allocation_seconds_total",
        "get_batch_sampling_seconds_total",
        "get_batch_concat_shuffle_seconds_total",
        "get_batch_seconds_total",
    ]
    lines = [f"\n=== {title} ==="]
    for k in order:
        if k in timings:
            v = timings[k]
            if isinstance(v, (int, float)):
                if k.endswith("_calls"):
                    lines.append(f"{k}: {int(v)}")
                else:
                    lines.append(f"{k}: {human_seconds(float(v))}")
            else:
                lines.append(f"{k}: {v}")
    notes = profiling.get("notes", {})
    if notes:
        lines.append("notes:")
        for nk, nv in notes.items():
            lines.append(f"  - {nk}: {nv}")
    return "\n".join(lines)


def run_once(root: str, rows: int, row_floor: int, batches: int, 
             use_cache: bool, cache_filename: str) -> Dict[str, Any]:
    # Optionally remove cache for cold start
    cache_path = os.path.join(root, cache_filename)
    if not use_cache and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except OSError:
            pass

    t0 = time.perf_counter()
    loader = EfficientDataLoader(
        root_directory=root,
        batch_size=rows,
        enable_manifest_cache=True,  # always build and possibly use cache
        cache_filename=cache_filename,
        enable_profiling=True,
    )
    t1 = time.perf_counter()

    # pull some batches to populate cache and measure sampling
    batch_times = []
    for _ in range(batches):
        b0 = time.perf_counter()
        _ = loader.get_batch(rows, ROW_FLOOR=row_floor)
        b1 = time.perf_counter()
        batch_times.append(b1 - b0)

    result = {
        "init_seconds": t1 - t0,
        "avg_batch_seconds": sum(batch_times) / len(batch_times) if batch_times else 0.0,
        "profiling": loader.profiling,
        "file_count": len(loader.all_files),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark EfficientDataLoader read/search speed and caching. "
            "By default, uses HostPreferences.training_data_path from Ordered_001_Initialize."
        )
    )
    parser.add_argument("--root", help="Override root directory containing .pkl files (recursively). If omitted, use HostPreferences.training_data_path.")
    parser.add_argument("--rows", type=int, default=512, help="Rows per batch")
    parser.add_argument("--row-floor", type=int, default=20, help="Minimum rows per file in a batch")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches to sample")
    parser.add_argument("--cache-filename", default=".efficient_dataloader_cache.pkl", help="Cache filename inside root directory")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output in JSON for machine parsing")
    args = parser.parse_args()

    # Resolve root from preferences if not provided
    resolved_root = None
    if args.root:
        resolved_root = os.path.abspath(args.root)
    else:
        try:
            # Emulate other modules: resolve preferences file relative to project root
            preferences_path = os.path.join(PROJECT_ROOT, "experiment.preferences")
            if not os.path.isfile(preferences_path):
                raise FileNotFoundError(f"Preferences file not found at {preferences_path}")
            prefs = HostPreferences(filename=preferences_path)
            if not prefs.training_data_path:
                raise ValueError("'training_data_path' is not set in preferences")
            resolved_root = os.path.abspath(prefs.training_data_path)
        except Exception as e:
            parser.error(
                f"Failed to resolve root from HostPreferences: {e}. "
                f"Provide --root explicitly or fix experiment.preferences at {os.path.join(PROJECT_ROOT, 'experiment.preferences')}."
            )
            return

    if not os.path.isdir(resolved_root):
        parser.error(f"Resolved root does not exist or is not a directory: {resolved_root}")
        return

    # Cold run: delete cache if exists, then instantiate and sample
    cold = run_once(resolved_root, args.rows, args.row_floor, args.batches, use_cache=False, cache_filename=args.cache_filename)

    # Warm run: instantiate again (should load from cache), then sample
    warm = run_once(resolved_root, args.rows, args.row_floor, args.batches, use_cache=True, cache_filename=args.cache_filename)

    if args.as_json:
        print(json.dumps({"cold": cold, "warm": warm}, indent=2))
    else:
        print(f"Root: {resolved_root}")
        print(f"Files discovered: {cold['file_count']}")
        print(f"Cold init (no cache manifest/metadata): {human_seconds(cold['init_seconds'])}")
        print(f"Warm init (with cache): {human_seconds(warm['init_seconds'])}")
        print(f"Cold avg batch time over {args.batches}: {human_seconds(cold['avg_batch_seconds'])}")
        print(f"Warm avg batch time over {args.batches}: {human_seconds(warm['avg_batch_seconds'])}")
        print(summarize_timings("Cold profiling", cold["profiling"]))
        print(summarize_timings("Warm profiling", warm["profiling"]))


if __name__ == "__main__":
    main()
