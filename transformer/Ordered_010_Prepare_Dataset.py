"""
Ordered_010_Prepare_Dataset.py

This script prepares the dataset for the Transformer model by sampling sequences from precomputed latent space cubes.

Overview:
The script processes compressed pickle files containing latent representations of physical data. 
It generates a specified number of samples (default 250,000) for both training and validation sets. 
Each sample is a spatio-temporal window: a sequence of 80 time steps at a fixed (y, z) coordinate, 
covering all 26 points along the X-axis.

File Structure:
- Constants: Coordinate lists for X, Y, Z, latent feature names, and input/output paths.
- parse_param: Utility to convert directory strings (e.g., '7p8') to float parameter values (7.8).
- get_available_yz: Scans files to discover valid (y, z) coordinate pairs in the source data.
- generate_sample_definitions: Creates random metadata (param, y, z, start_t) for dataset samples.
- process_and_save: 
    - Core logic that constructs 4D tensors and writes them to HDF5.
    - Uses ThreadPoolExecutor to process parameter sets in parallel.
    - Sorts samples by time to optimize file loading with a sliding window mechanism.
- main: Handles CLI arguments and triggers processing for training and validation splits.

Data Structure:
The output HDF5 files contain a 'data' dataset with shape (N, 80, 26, 52):
    - N: Number of samples.
    - 80: Time steps (Temporal dimension).
    - 26: X-coordinates (Spatial dimension).
    - 52: Features per point:
        - [0:47]: Latent features (47 dimensions).
        - [47]: X-coordinate.
        - [48]: Y-coordinate.
        - [49]: Z-coordinate.
        - [50]: Relative time (0-79).
        - [51]: Parameter value (extracted from directory name).

Usage:
    python Ordered_010_Prepare_Dataset.py --num_samples 250000 --test_run
"""
import os
import pandas as pd
import numpy as np
import h5py
import random
import argparse
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor
from transformer_config import add_config_arg, load_config, resolve_path

# Constants from issue description
X_COORDS = [ -49, -45, -41, -37, -33, -29, -26, -22, -18, -14, -10, -6, -2, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
Z_COORDS = [-21, -17, -13, -9, -5, -1, 2, 6, 10, 14, 18, 22]
Y_COORDS = [-71, -67, -63, -59, -55, -51, -47, -43, -39, -35, -31, -28, -24, -20, -16, -12, -8, -4, 0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75]
LATENT_COLS = [f"latent_{i}" for i in range(1, 48)]
NUM_X = len(X_COORDS)
NUM_TIME = 80
# Features: 47 (latent) + 3 (x,y,z) + 1 (rel_time) + 1 (param_val) = 52
NUM_FEATURES = 47 + 3 + 1 + 1 

INPUT_ROOT = None
OUTPUT_DIR = None

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

def generate_sample_definitions(param_sets, n_total):
    """Generate random (param_set, y, z, start_t) combinations."""
    ps_yz_map = {}
    ps_max_t_map = {}
    
    def get_info(ps):
        yz, max_t = get_available_yz_and_t(ps)
        return ps, yz, max_t

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(get_info, param_sets), 
                            total=len(param_sets), 
                            desc=f"Gathering metadata for {len(param_sets)} parameter sets"))
    
    for ps, valid_yz, max_t in results:
        if valid_yz and max_t >= NUM_TIME:
            ps_yz_map[ps] = valid_yz
            ps_max_t_map[ps] = max_t
    
    if not ps_yz_map:
        raise ValueError(f"No valid data found or all parameter sets have fewer than {NUM_TIME} time steps.")

    print(f"Found {len(ps_yz_map)} parameter sets with valid data.")
    samples = []
    param_list = list(ps_yz_map.keys())
    
    for _ in range(n_total):
        ps = random.choice(param_list)
        y, z = random.choice(ps_yz_map[ps])
        max_t = ps_max_t_map[ps]
        
        # Ensure we can fit a sequence of length NUM_TIME
        # If files are numbered 1 to 1200, and NUM_TIME=80,
        # the last possible start_t is 1200 - 80 + 1 = 1121.
        # Then sequence is 1121, 1122, ..., 1200 (exactly 80 steps)
        start_t = random.randint(1, max_t - NUM_TIME + 1)
        samples.append({'param_set': ps, 'y': y, 'z': z, 'start_t': start_t})
    
    return samples

def process_and_save(samples, output_path):
    """Extract data for samples and save to HDF5."""
    n = len(samples)
    samples_by_ps = {}
    for i, s in enumerate(samples):
        ps = s['param_set']
        if ps not in samples_by_ps:
            samples_by_ps[ps] = []
        samples_by_ps[ps].append((i, s))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create dataset: (N, 80, 26, 52)
        # Using gzip compression and shuffle for better storage efficiency
        ds = f.create_dataset('data', 
                               (n, NUM_TIME, NUM_X, NUM_FEATURES), 
                               dtype='float32', 
                               chunks=(1, NUM_TIME, NUM_X, NUM_FEATURES),
                               compression='gzip',
                               compression_opts=4,
                               shuffle=True)
        
        # Also store some metadata attributes
        f.attrs['x_coords'] = X_COORDS
        f.attrs['feature_description'] = f"0-46: latent, 47: x, 48: y, 49: z, 50: relative_time (0-{NUM_TIME-1}), 51: parameter_value"

        with tqdm(total=n, desc=f"Processing {os.path.basename(output_path)}") as pbar:
            def process_ps(ps):
                ps_val = parse_param(ps)
                ps_samples = samples_by_ps[ps]
                # Sort by start_t to optimize file loading
                ps_samples.sort(key=lambda x: x[1]['start_t'])
                
                current_window = {} # t -> dataframe
                
                for idx, s in ps_samples:
                    start_t = s['start_t']
                    y_val = s['y']
                    z_val = s['z']
                    
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
                                    current_window[t] = df
                                except:
                                    current_window[t] = None
                            else:
                                current_window[t] = None
                    
                    # Extract tensor for this sample
                    sample_tensor = np.zeros((NUM_TIME, NUM_X, NUM_FEATURES), dtype='float32')
                    for t_idx, t in enumerate(needed_times):
                        df = current_window.get(t)
                        if df is not None:
                            # Filter rows
                            mask = (df['y'] == y_val) & (df['z'] == z_val)
                            rows = df[mask]
                            
                            if not rows.empty:
                                # Reindex to ensure order and presence of all X
                                rows = rows.set_index('x').reindex(X_COORDS).reset_index()
                                
                                latents = rows[LATENT_COLS].values
                                # Handle potential NaNs from reindex
                                latents = np.nan_to_num(latents)
                                
                                xs = rows['x'].values.astype('float32')
                                ys = np.full(NUM_X, y_val, dtype='float32')
                                zs = np.full(NUM_X, z_val, dtype='float32')
                                ts = np.full(NUM_X, t_idx, dtype='float32') # Relative time
                                pv = np.full(NUM_X, ps_val, dtype='float32')
                                
                                combined = np.column_stack([latents, xs, ys, zs, ts, pv])
                                sample_tensor[t_idx] = combined
                    
                    ds[idx] = sample_tensor
                    pbar.update(1)

            with ThreadPoolExecutor() as executor:
                list(executor.map(process_ps, samples_by_ps.keys()))

def main():
    parser = argparse.ArgumentParser(description="Prepare Transformer dataset from latent space cubes.")
    add_config_arg(parser)
    parser.add_argument("--input_root", "--input-root", dest="input_root", default=None, help="Directory containing parameter-set subdirectories of .pkl.gz latent cubes.")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", default=None, help="Directory where training_data.h5 and validation_data.h5 are written.")
    parser.add_argument("--num_samples", "--num-samples", dest="num_samples", type=int, default=None, help="Number of samples per file.")
    parser.add_argument("--test_run", "--test-run", dest="test_run", action="store_true", help="Run with very few samples for testing.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num_time", "--num-time", dest="num_time", type=int, default=None, help="Number of consecutive time steps per sample.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    path_cfg = cfg["paths"]

    global INPUT_ROOT, OUTPUT_DIR, NUM_TIME
    INPUT_ROOT = resolve_path(args.input_root or path_cfg["latent_input_root"])
    OUTPUT_DIR = resolve_path(args.output_dir or path_cfg["transformer_input_dir"])
    NUM_TIME = args.num_time if args.num_time is not None else data_cfg.get("num_time", NUM_TIME)

    seed = args.seed if args.seed is not None else data_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    configured_samples = args.num_samples if args.num_samples is not None else data_cfg["num_samples"]
    num_samples = data_cfg["test_num_samples"] if args.test_run else configured_samples
    
    param_sets = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])
    if not param_sets:
        print(f"No parameter directories found in {INPUT_ROOT}")
        return

    print(f"Total samples requested per file: {num_samples}")
    
    # Generate definitions for Training and Validation
    print("Generating training sample definitions...")
    train_samples = generate_sample_definitions(param_sets, num_samples)
    print("Generating validation sample definitions...")
    val_samples = generate_sample_definitions(param_sets, num_samples)
    
    # Process Training
    print("\n=== Processing Training Data ===")
    process_and_save(train_samples, os.path.join(OUTPUT_DIR, "training_data.h5"))
    
    # Process Validation
    print("\n=== Processing Validation Data ===")
    process_and_save(val_samples, os.path.join(OUTPUT_DIR, "validation_data.h5"))

    print("\nDone! Files created in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
