"""
This script prepares an evaluation dataset for the Transformer model by creating an HDF5 file 
containing random samples of 80-timestep windows. Each sample consists of a (y, z) coordinate line 
across all 26 X coordinates.

For each point in the window, it stores 52 features:
- 47 latent variables
- x, y, z coordinates
- Relative time (0-79)
- Parameter value

Additionally, it saves the original velocities (vx, vy, vz) for the 80th timestep (t_idx=79) 
as ground truth for evaluation. The sampling process is designed to align with the 
validation data from Ordered_010_Prepare_Dataset.py.
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

# Constants - same as Ordered_010_Prepare_Dataset.py
X_COORDS = [ -49, -45, -41, -37, -33, -29, -26, -22, -18, -14, -10, -6, -2, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
Z_COORDS = [-21, -17, -13, -9, -5, -1, 2, 6, 10, 14, 18, 22]
Y_COORDS = [-71, -67, -63, -59, -55, -51, -47, -43, -39, -35, -31, -28, -24, -20, -16, -12, -8, -4, 0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75]

LATENT_COLS = [f"latent_{i}" for i in range(1, 48)]
NUM_X = len(X_COORDS)
NUM_TIME = 80
# Features: 47 (latent) + 3 (x,y,z) + 1 (rel_time) + 1 (param_val) = 52
NUM_FEATURES = 47 + 3 + 1 + 1 

INPUT_ROOT = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent"
OUTPUT_H5 = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"

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
        start_t = random.randint(1, max_t - NUM_TIME + 1)
        samples.append({'param_set': ps, 'y': y, 'z': z, 'start_t': start_t})
    
    return samples

def process_and_save(samples, output_path):
    """Extract data for samples and save to HDF5, including original velocities for T80."""
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
        
        # Metadata attributes
        f.attrs['x_coords'] = X_COORDS
        f.attrs['feature_description'] = f"0-46: latent, 47: x, 48: y, 49: z, 50: relative_time (0-{NUM_TIME-1}), 51: parameter_value"
        f.attrs['originals_description'] = f"Original vx, vy, vz for the {NUM_TIME}th timestep (t_idx={NUM_TIME-1})"

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
                    original_velocities = np.zeros((NUM_X, 3), dtype='float32')

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

                                # Capture original velocities for the 80th timestep
                                if t_idx == NUM_TIME - 1:
                                    orig_v = rows[['original_vx', 'original_vy', 'original_vz']].values.astype('float32')
                                    original_velocities = np.nan_to_num(orig_v)
                    
                    ds[idx] = sample_tensor
                    ds_orig[idx] = original_velocities
                    pbar.update(1)

            with ThreadPoolExecutor() as executor:
                list(executor.map(process_ps, samples_by_ps.keys()))

def main():
    parser = argparse.ArgumentParser(description="Prepare Evaluation dataset with original velocity metadata.")
    parser.add_argument("--num_samples", type=int, default=250000, help="Number of samples.")
    parser.add_argument("--test_run", action="store_true", help="Run with very few samples for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_samples = 1000 if args.test_run else args.num_samples
    
    param_sets = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])
    if not param_sets:
        print(f"No parameter directories found in {INPUT_ROOT}")
        return

    print(f"Total samples requested for evaluation file: {num_samples}")
    
    # To match validation dataset from Ordered_010_Prepare_Dataset.py, we call it twice
    print("Generating training definitions (to align random state)...")
    _ = generate_sample_definitions(param_sets, num_samples)
    
    print("Generating evaluation (validation-aligned) sample definitions...")
    eval_samples = generate_sample_definitions(param_sets, num_samples)
    
    # Process and save
    print(f"\n=== Processing Evaluation Data ===")
    process_and_save(eval_samples, OUTPUT_H5)

    print("\nDone! File created at:", OUTPUT_H5)

if __name__ == "__main__":
    main()
