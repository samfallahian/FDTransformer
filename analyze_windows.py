import h5py
import pandas as pd
import numpy as np
import os
import sys

def analyze_windows():
    h5_path = '/Users/kkreth/PycharmProjects/data/transformer_input/training_data.h5'
    core_csv = 'vorticity_core_points.csv'
    
    if not os.path.exists(h5_path):
        print(f"H5 not found: {h5_path}")
        return

    core = pd.read_csv(core_csv)
    core_yz = set(zip(core['y'], core['z']))
    
    f = h5py.File(h5_path, 'r')
    total_rows = f['data'].shape[0]
    chunk_size = 200000
    
    # wake_map: (u, y, z) -> [min_idx, max_idx]
    wake_map = {}
    
    print(f"Scanning {total_rows} indices for wake window identification...")
    
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        # Just grab the metadata features
        data = f['data'][start:end, 0, 0, 48:52] 
        
        for i in range(len(data)):
            y, z = int(data[i, 0]), int(data[i, 1])
            if (y, z) in core_yz:
                u = data[i, 3] # ps_idx
                key = (u, y, z)
                if key not in wake_map:
                    wake_map[key] = [start + i, start + i]
                else:
                    wake_map[key][1] = start + i
        
        print(f"  Processed {end}/{total_rows}. Found {len(wake_map)} unique areas.")

    print("\n" + "="*50)
    print("WAKE AREA WINDOW ANALYSIS (FIRST 15 AREAS)")
    print("="*50)
    print(f"{'Experiment':<10} | {'Y':<5} | {'Z':<5} | {'Span':<8} | {'40-step Windows'}")
    print("-" * 50)
    
    total_windows = 0
    items = sorted(list(wake_map.items()))
    for k, v in items[:15]:
        span = v[1] - v[0]
        windows = span // 40
        total_windows += windows
        print(f"{k[0]:<10.1f} | {k[1]:<5} | {k[2]:<5} | {span:<8} | {windows}")

    # Estimate total potential
    grand_total_windows = sum((v[1]-v[0])//40 for v in wake_map.values())
    print("-" * 50)
    print(f"Total Unique (U, y, z) Areas: {len(wake_map)}")
    print(f"Total Non-Overlapping 40-step Windows: {grand_total_windows}")
    print(f"Target (50% of Wake Areas): {len(wake_map)//2} areas")
    print(f"Resulting Dataset Size: ~{grand_total_windows} sequences (balanced 50/50)")
    
    f.close()

if __name__ == "__main__":
    analyze_windows()
