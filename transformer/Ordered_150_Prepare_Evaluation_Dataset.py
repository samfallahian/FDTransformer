"""
This script prepares an evaluation dataset for the Transformer model by creating an HDF5 file 
containing random samples of 80-timestep windows. Each sample consists of a (y, z) coordinate line 
across all 26 X coordinates.

For each point in the window, it stores 52 features:
- 47 latent variables
- x, y, z coordinates
- Relative time (0-79)
- Parameter value

Additionally, it saves:
- Original velocities (vx, vy, vz) for the 80th timestep (t_idx=79)
- Sample start frame metadata (start_t) and source time metadata (start_time)

The sampling process is designed to align with the validation data from
Ordered_010_Prepare_Dataset.py.
"""
import os
import pandas as pd
import numpy as np
import h5py
import random
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformer_config import add_config_arg, load_config, optional_int, resolve_path

# Constants - same as Ordered_010_Prepare_Dataset.py
X_COORDS = [ -49, -45, -41, -37, -33, -29, -26, -22, -18, -14, -10, -6, -2, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
Z_COORDS = [-21, -17, -13, -9, -5, -1, 2, 6, 10, 14, 18, 22]
Y_COORDS = [-71, -67, -63, -59, -55, -51, -47, -43, -39, -35, -31, -28, -24, -20, -16, -12, -8, -4, 0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75]

LATENT_COLS = [f"latent_{i}" for i in range(1, 48)]
NUM_X = len(X_COORDS)
NUM_TIME = 80
# Features: 47 (latent) + 3 (x,y,z) + 1 (rel_time) + 1 (param_val) = 52
NUM_FEATURES = 47 + 3 + 1 + 1 
LATENT_DIM = len(LATENT_COLS)
X_COORDS_ARRAY = np.asarray(X_COORDS, dtype='float32')
ORIG_V_COLS = ['original_vx', 'original_vy', 'original_vz']
DEFAULT_WORKERS = min(8, (os.cpu_count() or 1))

INPUT_ROOT = None
OUTPUT_H5 = None

def parse_param(p_str):
    """Convert directory name like '7p8' to float 7.8."""
    return float(p_str.replace('p', '.'))

def get_available_yz_and_t(param_set):
    """Identify which (y, z) pairs and how many time steps are present in the data."""
    p_dir = os.path.join(INPUT_ROOT, param_set)
    files = sorted([f for f in os.listdir(p_dir) if f.endswith('.pkl.gz')])
    if not files:
        return [], 0
    
    # Extract maximum time index from filenames (e.g., '1200.pkl.gz' -> 1200)
    try:
        max_t = int(files[-1].split('.')[0])
    except:
        max_t = len(files)
        
    # Check a file in the middle to ensure good coverage
    sample_file = os.path.join(p_dir, files[len(files)//2])
    try:
        df = pd.read_pickle(sample_file, compression='gzip')
        yz = df[['y', 'z']].drop_duplicates()
        valid = yz[yz['y'].isin(Y_COORDS) & yz['z'].isin(Z_COORDS)]
        return valid.values.tolist(), max_t
    except Exception as e:
        print(f"Error reading {sample_file}: {e}")
        return [], 0

def generate_sample_definitions(param_sets, n_total, fixed_start_t=None, max_workers=DEFAULT_WORKERS):
    """
    Generate (param_set, y, z, start_t) combinations.

    If fixed_start_t is provided, all samples use that start frame.
    Otherwise, start_t is sampled randomly per sample.
    """
    if fixed_start_t is not None and fixed_start_t < 1:
        raise ValueError("fixed_start_t must be >= 1.")

    ps_yz_map = {}
    ps_max_t_map = {}
    
    def get_info(ps):
        yz, max_t = get_available_yz_and_t(ps)
        return ps, yz, max_t

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(get_info, param_sets), 
                            total=len(param_sets), 
                            desc=f"Gathering metadata for {len(param_sets)} parameter sets"))
    
    required_min_max_t = NUM_TIME if fixed_start_t is None else (fixed_start_t + NUM_TIME - 1)
    for ps, valid_yz, max_t in results:
        if valid_yz and max_t >= required_min_max_t:
            ps_yz_map[ps] = valid_yz
            ps_max_t_map[ps] = max_t
    
    if not ps_yz_map:
        if fixed_start_t is None:
            raise ValueError(f"No valid data found or all parameter sets have fewer than {NUM_TIME} time steps.")
        raise ValueError(
            f"No valid data found for fixed_start_t={fixed_start_t}. "
            f"Need at least {required_min_max_t} frames per parameter set."
        )

    print(f"Found {len(ps_yz_map)} parameter sets with valid data.")
    samples = []
    param_list = list(ps_yz_map.keys())
    
    for _ in range(n_total):
        ps = random.choice(param_list)
        y, z = random.choice(ps_yz_map[ps])
        if fixed_start_t is None:
            max_t = ps_max_t_map[ps]
            # Ensure we can fit a sequence of length NUM_TIME
            start_t = random.randint(1, max_t - NUM_TIME + 1)
        else:
            start_t = fixed_start_t
        samples.append({'param_set': ps, 'y': y, 'z': z, 'start_t': start_t})
    
    return samples

def process_and_save(samples, output_path, max_workers=DEFAULT_WORKERS):
    """Extract data for samples and save to HDF5, including original velocities and start-time metadata."""
    n = len(samples)
    samples_by_ps = {}
    for i, s in enumerate(samples):
        ps = s['param_set']
        if ps not in samples_by_ps:
            samples_by_ps[ps] = []
        samples_by_ps[ps].append((i, s))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Standard dataset: (N, 80, 26, 52)
        ds = f.create_dataset('data', (n, NUM_TIME, NUM_X, NUM_FEATURES), dtype='float32', chunks=(1, NUM_TIME, NUM_X, NUM_FEATURES))
        
        # New dataset for original velocities at 80th timestep: (N, 26, 3)
        ds_orig = f.create_dataset('originals', (n, NUM_X, 3), dtype='float32', chunks=(1, NUM_X, 3))

        # Metadata datasets for sample start frames/times
        ds_start_t = f.create_dataset('start_t', (n,), dtype='int32')
        ds_start_time = f.create_dataset('start_time', (n,), dtype='float32')
        
        # Metadata attributes
        f.attrs['x_coords'] = X_COORDS
        f.attrs['feature_description'] = f"0-46: latent, 47: x, 48: y, 49: z, 50: relative_time (0-{NUM_TIME-1}), 51: parameter_value"
        f.attrs['originals_description'] = f"Original vx, vy, vz for the {NUM_TIME}th timestep (t_idx={NUM_TIME-1})"
        f.attrs['start_t_description'] = "Absolute start frame index used when sampling each record."
        f.attrs['start_time_description'] = "Source dataframe 'time' value at sample start (fallback: start_t)."

        with tqdm(total=n, desc=f"Processing {os.path.basename(output_path)}") as pbar:
            def process_ps(ps):
                ps_val = parse_param(ps)
                ps_samples = samples_by_ps[ps]
                # Sort by start_t to optimize file loading
                ps_samples.sort(key=lambda x: x[1]['start_t'])
                
                # t -> {'df': dataframe, 'yz_cache': {(y,z): (latents, orig_v, start_time)}} or None
                current_window = {}

                def get_cached_line(frame_entry, y_val, z_val):
                    """Cache per-(time, y, z) extraction to avoid repeated DataFrame filtering/reindexing."""
                    yz_key = (y_val, z_val)
                    yz_cache = frame_entry['yz_cache']
                    if yz_key in yz_cache:
                        return yz_cache[yz_key]

                    df = frame_entry['df']
                    rows = df[(df['y'] == y_val) & (df['z'] == z_val)]
                    if rows.empty:
                        yz_cache[yz_key] = None
                        return None

                    rows = rows.set_index('x').reindex(X_COORDS)
                    latents = np.nan_to_num(rows[LATENT_COLS].to_numpy(dtype='float32'))
                    orig_v = np.nan_to_num(rows[ORIG_V_COLS].to_numpy(dtype='float32'))

                    start_time_val = None
                    if 'time' in rows.columns:
                        valid_times = rows['time'].dropna()
                        if not valid_times.empty:
                            start_time_val = float(valid_times.iloc[0])

                    yz_cache[yz_key] = (latents, orig_v, start_time_val)
                    return yz_cache[yz_key]
                
                for idx, s in ps_samples:
                    start_t = s['start_t']
                    y_val = s['y']
                    z_val = s['z']
                    sample_start_time = float(start_t)  # fallback when 'time' is unavailable
                    
                    needed_times = range(start_t, start_t + NUM_TIME)
                    
                    # Cleanup window
                    for t in list(current_window.keys()):
                        if t < start_t:
                            del current_window[t]
                    
                    # Load new files
                    for t in needed_times:
                        if t not in current_window:
                            file_path = os.path.join(INPUT_ROOT, ps, f"{t:04d}.pkl.gz")
                            if os.path.exists(file_path):
                                try:
                                    df = pd.read_pickle(file_path, compression='gzip')
                                    # Pre-filter for X_COORDS to save memory and speed up grouping
                                    df = df[df['x'].isin(X_COORDS)]
                                    current_window[t] = {'df': df, 'yz_cache': {}}
                                except:
                                    current_window[t] = None
                            else:
                                current_window[t] = None
                    
                    # Extract tensor for this sample
                    sample_tensor = np.zeros((NUM_TIME, NUM_X, NUM_FEATURES), dtype='float32')
                    original_velocities = np.zeros((NUM_X, 3), dtype='float32')

                    for t_idx, t in enumerate(needed_times):
                        frame_entry = current_window.get(t)
                        if frame_entry is None:
                            continue

                        cached_line = get_cached_line(frame_entry, y_val, z_val)
                        if cached_line is None:
                            continue

                        latents, orig_v, start_time_val = cached_line
                        sample_tensor[t_idx, :, :LATENT_DIM] = latents
                        sample_tensor[t_idx, :, LATENT_DIM] = X_COORDS_ARRAY
                        sample_tensor[t_idx, :, LATENT_DIM + 1] = y_val
                        sample_tensor[t_idx, :, LATENT_DIM + 2] = z_val
                        sample_tensor[t_idx, :, LATENT_DIM + 3] = t_idx  # Relative time
                        sample_tensor[t_idx, :, LATENT_DIM + 4] = ps_val

                        # Capture actual source start time from dataframe, if available
                        if t_idx == 0 and start_time_val is not None:
                            sample_start_time = start_time_val

                        # Capture original velocities for the 80th timestep
                        if t_idx == NUM_TIME - 1:
                            original_velocities[:] = orig_v
                    
                    ds[idx] = sample_tensor
                    ds_orig[idx] = original_velocities
                    ds_start_t[idx] = start_t
                    ds_start_time[idx] = sample_start_time
                    pbar.update(1)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(process_ps, samples_by_ps.keys()))

def main():
    parser = argparse.ArgumentParser(description="Prepare Evaluation dataset with original velocity metadata.")
    add_config_arg(parser)
    parser.add_argument("--input_root", "--input-root", dest="input_root", default=None, help="Directory containing parameter-set subdirectories of source .pkl.gz files.")
    parser.add_argument("--output_h5", "--output-h5", dest="output_h5", default=None, help="Evaluation HDF5 output path.")
    parser.add_argument("--num_samples", "--num-samples", dest="num_samples", type=int, default=None, help="Number of samples.")
    parser.add_argument("--test_run", "--test-run", dest="test_run", action="store_true", help="Run with very few samples for testing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num_time", "--num-time", dest="num_time", type=int, default=None, help="Number of consecutive time steps per sample.")
    parser.add_argument(
        "--start_t",
        "--start-t",
        dest="start_t",
        default=None,
        help="Fixed start frame index for all samples. Use none/null/all/0 for random sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Thread workers for metadata scan/extraction (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    path_cfg = cfg["paths"]

    global INPUT_ROOT, OUTPUT_H5, NUM_TIME
    INPUT_ROOT = resolve_path(args.input_root or path_cfg["evaluation_input_root"])
    OUTPUT_H5 = resolve_path(args.output_h5 or path_cfg["evaluation_h5"])
    NUM_TIME = args.num_time if args.num_time is not None else data_cfg.get("num_time", NUM_TIME)
    workers = args.workers if args.workers is not None else data_cfg.get("workers", DEFAULT_WORKERS)
    start_t = optional_int(args.start_t) if args.start_t is not None else optional_int(data_cfg.get("evaluation_start_t"))

    if workers < 1:
        raise ValueError("--workers must be >= 1.")

    seed = args.seed if args.seed is not None else data_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    configured_samples = args.num_samples if args.num_samples is not None else data_cfg["num_samples"]
    num_samples = data_cfg["test_num_samples"] if args.test_run else configured_samples
    
    param_sets = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])
    if not param_sets:
        print(f"No parameter directories found in {INPUT_ROOT}")
        return

    print(f"Total samples requested for evaluation file: {num_samples}")
    print(f"Using workers: {workers}")
    if start_t is None:
        print("Sampling mode: random start_t")
    else:
        print(f"Sampling mode: fixed start_t={start_t}")
    
    # To match validation dataset from Ordered_010_Prepare_Dataset.py, we call it twice
    print("Generating training definitions (to align random state)...")
    _ = generate_sample_definitions(
        param_sets,
        num_samples,
        fixed_start_t=start_t,
        max_workers=workers,
    )
    
    print("Generating evaluation (validation-aligned) sample definitions...")
    eval_samples = generate_sample_definitions(
        param_sets,
        num_samples,
        fixed_start_t=start_t,
        max_workers=workers,
    )
    
    # Process and save
    print(f"\n=== Processing Evaluation Data ===")
    process_and_save(eval_samples, OUTPUT_H5, max_workers=workers)

    print("\nDone! File created at:", OUTPUT_H5)

if __name__ == "__main__":
    main()
