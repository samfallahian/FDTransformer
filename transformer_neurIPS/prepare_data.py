import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import random
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

# Paths
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
SOURCE_ROOT = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "transformer_neurIPS/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# NUM_TIME_NEW is the sequence length (context + forecast) in frames.
# Default 40 (12 context + 28 forecast, 233.3 ms forecast @ 120 Hz -- v1.0).
# For the v2.0 migration this is overridden to 80 (12 context + 68 forecast,
# 566.7 ms forecast @ 120 Hz) via the --num-time CLI flag, which also switches
# the output filenames to train_<N>.h5 / val_<N>.h5. WINDOWS_PER_COORD is
# derived so that every coordinate contributes disjoint, non-overlapping
# windows tiling all TOTAL_TIMESTAMPS frames.
NUM_TIME_NEW = 40
TOTAL_TIMESTAMPS = 1200
WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW # 30 for N=40, 15 for N=80
NUM_X = 26
X_COORDS = np.array([-29, -26, -22, -18, -14, -10, -6, -2, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69], dtype='float32')

# 24 unique wake coordinates identified from documentation
WAKE_COORDS = [
    (-71, -1), (-67, -1), (-63, -21), (-59, -17), (-55, 2), (-47, -21), 
    (-43, 22), (-31, -21), (-16, 22), (-12, 22), (-8, 18), (0, -1), 
    (3, 10), (11, 22), (15, 22), (23, 22), (27, 22), (39, 10), 
    (47, -21), (55, 2), (59, 2), (67, 10), (71, -13), (75, -5)
]

def parse_param(p_str):
    mapping = {"3p6": 5.6, "4p4": 6.9, "4p6": 7.2, "5p2": 8.1, "6p4": 10.0, 
               "6p6": 10.3, "7p2": 11.3, "7p8": 12.2, "8p4": 13.1, "10p4": 16.3, "11p4": 17.8}
    return mapping.get(p_str, 0.0)

def get_file_path(param_set, step):
    return os.path.join(SOURCE_ROOT, param_set, f"{step:04d}.pkl.gz")

def extract_from_file(args):
    """Worker function to extract multiple coordinates from a single file."""
    f_path, coords_to_extract, param_val = args
    if not os.path.exists(f_path):
        return None
    
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
        # print(f"Error processing {f_path}: {e}")
        return None

def process_set(param_list, out_name, selected_wake_coords, sample_percent=100.0, test_mode=False):
    print(f"\n🚀 Building {out_name}...")
    
    # 1. Determine all unique coordinates needed per parameter set
    param_to_coords = {ps: set() for ps in param_list}
    wake_plans = []
    
    if test_mode:
        print("Running in TEST MODE: only first sequence for each experiment.")
        for ps in param_list:
            y, z = selected_wake_coords[0]
            start_step = 1
            wake_plans.append((ps, y, z, start_step, True))
            param_to_coords[ps].add((y, z))
    else:
        for ps in param_list:
            for y, z in selected_wake_coords:
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
    random_plans = []
    if not test_mode:
        random.seed(42)
        
        # Get range of y and z from a sample file if possible, else use default
        sample_ps = param_list[0]
        sample_file = get_file_path(sample_ps, 1)
        if os.path.exists(sample_file):
            try:
                sdf = pd.read_pickle(sample_file, compression='gzip')
                y_min, y_max = int(sdf['y'].min()), int(sdf['y'].max())
                z_min, z_max = int(sdf['z'].min()), int(sdf['z'].max())
                y_range = np.arange(y_min, y_max + 1, 4)
                z_range = np.arange(z_min, z_max + 1, 4)
                print(f"Random sampling from y:[{y_min}, {y_max}], z:[{z_min}, {z_max}]")
            except:
                y_range = np.arange(-80, 81, 4)
                z_range = np.arange(-80, 81, 4)
        else:
            y_range = np.arange(-80, 81, 4)
            z_range = np.arange(-80, 81, 4)

        while len(random_plans) < num_wake_plans:
            ps = random.choice(param_list)
            y = random.choice(y_range)
            z = random.choice(z_range)
            if (y, z) in WAKE_COORDS: continue
            w_idx = random.randint(0, WINDOWS_PER_COORD - 1)
            start_step = w_idx * NUM_TIME_NEW + 1
            random_plans.append((ps, y, z, start_step, False))
            param_to_coords[ps].add((y, z))

    print(f"Planned {num_wake_plans} wake sequences and {len(random_plans)} random sequences.")
    
    # 2. Extract data for each parameter set
    all_extracted_data = {} # (ps, y, z, step) -> features
    
    for ps in param_list:
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
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(extract_from_file, tasks), total=len(tasks), desc=f"Reading {ps} once"))
        
        for step_idx, step_results in enumerate(results):
            step = steps_needed[step_idx]
            if step_results is None:
                continue
            for (y, z), data in step_results.items():
                if data is not None:
                    all_extracted_data[(ps, y, z, step)] = data

    # 3. Assemble sequences and Count by experiment
    print(f"Assembling sequences for {out_name}...")
    all_plans = wake_plans + random_plans
    random.shuffle(all_plans)
    
    final_data = []
    counts = Counter()
    
    for ps, y_val, z_val, start_step, is_wake in tqdm(all_plans, desc="Assembling"):
        seq = np.zeros((NUM_TIME_NEW, NUM_X, 52), dtype='float32')
        valid_seq = True
        for t_offset in range(NUM_TIME_NEW):
            step = start_step + t_offset
            data = all_extracted_data.get((ps, y_val, z_val, step))
            if data is None:
                # print(f"Skipping sequence {ps} at ({y_val}, {z_val}) step {step}: Missing data.")
                valid_seq = False
                break
            
            # Check for non-trivial data (not just zeros in latent dimensions)
            # Latents are columns 0:47
            if np.all(data[:, :47] == 0):
                print(f"Skipping sequence {ps} at ({y_val}, {z_val}) step {step}: All-zero latents found.")
                valid_seq = False
                break

            step_data = data.copy()
            step_data[:, 50] = float(t_offset)
            seq[t_offset] = step_data
        
        if valid_seq:
            final_data.append(seq)
            counts[ps] += 1

    # 4. Report counts
    print(f"\n📊 Sequence counts for {out_name} by experiment:")
    for ps in sorted(param_list):
        print(f"  - {ps}: {counts[ps]} sequences")

    # 5. Save to HDF5
    print(f"Saving to {out_name}...")
    with h5py.File(os.path.join(OUTPUT_DIR, out_name), 'w') as f_out:
        f_out.create_dataset('data', data=np.array(final_data), compression='gzip')
        # Add metadata
        f_out.attrs['source_root'] = SOURCE_ROOT
        f_out.attrs['sample_percent'] = sample_percent
        f_out.attrs['num_sequences'] = len(final_data)
        f_out.attrs['creation_time'] = time.ctime()
        f_out.attrs['param_list'] = [p.encode('utf-8') for p in param_list]
    print(f"Final {out_name} size: {len(final_data)} sequences.")

def prepare_data(sample_percent=100.0, test_mode=False):
    t0 = time.time()
    
    # Use all 24 wake coordinates for 100% coverage as requested
    selected_wake_coords = WAKE_COORDS
    print(f"Using all {len(selected_wake_coords)} wake coordinates for 100% coverage.")

    train_params = ["3p6", "4p4", "4p6", "5p2", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"]
    val_params = ["6p4"]

    train_out = f"train_{NUM_TIME_NEW}.h5"
    val_out = f"val_{NUM_TIME_NEW}.h5"
    process_set(train_params, train_out, selected_wake_coords, sample_percent, test_mode)
    process_set(val_params, val_out, selected_wake_coords, sample_percent, test_mode)
    
    print(f"\n✅ Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=100.0, help="Percentage of data to sample (0-100)")
    parser.add_argument("--test", action="store_true", help="Test mode: only first sequence for each experiment")
    parser.add_argument("--num-time", type=int, default=NUM_TIME_NEW,
                        help="Sequence length in frames (40 for v1.0, 80 for the v2.0 migration). "
                             "Must divide TOTAL_TIMESTAMPS (1200). Output filenames switch to "
                             "train_<N>.h5 / val_<N>.h5 accordingly.")
    args = parser.parse_args()

    if TOTAL_TIMESTAMPS % args.num_time != 0:
        raise SystemExit(
            f"--num-time={args.num_time} does not divide TOTAL_TIMESTAMPS={TOTAL_TIMESTAMPS}; "
            f"pick a value from {[n for n in (10, 20, 24, 30, 40, 48, 50, 60, 75, 80, 100, 120, 150, 200, 240, 300, 400, 600, 1200) if TOTAL_TIMESTAMPS % n == 0]}"
        )
    NUM_TIME_NEW = args.num_time
    WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW
    print(f"NUM_TIME_NEW={NUM_TIME_NEW}, WINDOWS_PER_COORD={WINDOWS_PER_COORD}")
    prepare_data(args.sample, args.test)
