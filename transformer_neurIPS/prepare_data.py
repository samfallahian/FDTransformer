import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

# Paths
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
SOURCE_ROOT = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "transformer_neurIPS/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
NUM_TIME_NEW = 40
TOTAL_TIMESTAMPS = 1200
WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW # 30
NUM_X = 26
X_COORDS = np.linspace(-30, 30, NUM_X)

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
        latent_cols = df.columns.tolist()[10:57]
        
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
            extracted = np.column_stack([
                lats, 
                rows['x'].values.astype('float32'),
                np.full(NUM_X, y_val, dtype='float32'),
                np.full(NUM_X, z_val, dtype='float32'),
                np.zeros(NUM_X, dtype='float32'), # t_idx placeholder
                np.full(NUM_X, param_val, dtype='float32')
            ])
            results[(y_val, z_val)] = extracted
        return results
    except Exception as e:
        # print(f"Error processing {f_path}: {e}")
        return None

def process_set(param_list, out_name, selected_wake_coords):
    print(f"\n🚀 Building {out_name}...")
    
    # 1. Determine all unique coordinates needed per parameter set
    param_to_coords = {ps: set() for ps in param_list}
    wake_plans = []
    for ps in param_list:
        for y, z in selected_wake_coords:
            param_to_coords[ps].add((y, z))
            for w_idx in range(WINDOWS_PER_COORD):
                start_step = w_idx * NUM_TIME_NEW + 1
                wake_plans.append((ps, y, z, start_step, True))

    num_wake_plans = len(wake_plans)
    
    # Generate random plans 1-for-1
    random_plans = []
    random.seed(42)
    while len(random_plans) < num_wake_plans:
        ps = random.choice(param_list)
        y = random.choice(np.arange(-80, 81, 4))
        z = random.choice(np.arange(-80, 81, 4))
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
        for step in range(1, TOTAL_TIMESTAMPS + 1):
            f_path = get_file_path(ps, step)
            tasks.append((f_path, coords, param_val))
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(extract_from_file, tasks), total=len(tasks), desc=f"Reading {ps} once"))
        
        for step_idx, step_results in enumerate(results):
            step = step_idx + 1
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
    print(f"Final {out_name} size: {len(final_data)} sequences.")

def prepare_data():
    t0 = time.time()
    
    random.seed(42)
    selected_wake_coords = random.sample(WAKE_COORDS, 12)
    print(f"Selected 12 wake coordinates for 50% coverage.")

    train_params = ["3p6", "4p4", "4p6", "5p2", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"]
    val_params = ["6p4"]

    process_set(train_params, "train_40.h5", selected_wake_coords)
    process_set(val_params, "val_40.h5", selected_wake_coords)
    
    print(f"\n✅ Total time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    prepare_data()
