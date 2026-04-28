"""
Ordered_060_Create_Cubes_Per_Line.py

This script constructs "velocity cubes" for each centroid row in the filtered simulation data.
It maps each centroid to its 124 neighbors (pre-computed in a mapping CSV) and retrieves
their corresponding velocity components (vx, vy, vz) from the same time step.

Flow:
    [ Filtered Time-Step Data (.pkl.gz) ]      [ Neighbor Mapping (CSV) ]
                   |                                     |
                   |                                     v
                   |                         1. Load centroid-to-neighbor 
                   |                            coordinate mapping.
                   |                                     |
                   v                                     |
    2. Parallel Process Files <--------------------------+
       For each file (one time-step):
         a. Build fast lookup dictionaries for vx, vy, vz.
         b. For each centroid row:
            - Identify its 124 neighbors' (x,y,z).
            - Retrieve their velocities (default 0.0 if missing).
         c. Flatten these into 375 velocity columns (125 points * 3 components).
         d. Concatenate with original metadata.
                   |
                   v
    [ Output: Final_Cubed_OG_Data/{input_file_path}.pkl.gz ]

Main Components:
- init_worker(): Loads the large mapping file once per worker process.
- process_file_task(): Performs the coordinate-to-velocity lookup and cube formation.
- main(): Manages the parallel execution using ProcessPoolExecutor.
"""

import os
import pandas as pd
import numpy as np
import time
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Globals for workers to avoid re-loading/serializing large data for every file
G_MAP_DF = None
G_NEIGHBOR_PREFIXES = None

def init_worker(map_path):
    """
    Load the mapping file once per worker process.
    """
    global G_MAP_DF, G_NEIGHBOR_PREFIXES
    G_MAP_DF = pd.read_csv(map_path)
    
    # Cast neighbor coordinate columns to int for consistency and faster lookup
    nbr_cols = [c for c in G_MAP_DF.columns if c.startswith('nbr_')]
    for col in nbr_cols:
        G_MAP_DF[col] = G_MAP_DF[col].astype(int)
    
    # Identify neighbor prefixes
    nbr_x_cols = [c for c in G_MAP_DF.columns if c.startswith('nbr_') and c.endswith('_x')]
    G_NEIGHBOR_PREFIXES = [c[4:-2] for c in nbr_x_cols]

def process_file_task(input_file, output_file):
    """
    Worker task: Process a single file, adding velocity information for neighbors.
    Uses G_MAP_DF and G_NEIGHBOR_PREFIXES from the worker's global scope.
    """
    try:
        df_input = pd.read_pickle(input_file, compression='gzip')
        if df_input.empty:
            return True
        
        # Create lookups for each velocity component for speed
        # Using integer keys for faster dictionary lookups
        df_lookup_src = df_input.copy()
        for col in ['x', 'y', 'z']:
            df_lookup_src[col] = df_lookup_src[col].astype(int)
        df_lookup_src = df_lookup_src.set_index(['x', 'y', 'z'])
        
        lookup_vx = df_lookup_src['vx'].to_dict()
        lookup_vy = df_lookup_src['vy'].to_dict()
        lookup_vz = df_lookup_src['vz'].to_dict()
        
        # Merge input with map to get neighbor coordinates for each centroid
        # The join filters the input to only include rows that are centroids
        merged_df = pd.merge(df_input, G_MAP_DF, 
                             left_on=['x', 'y', 'z'], 
                             right_on=['centroid_x', 'centroid_y', 'centroid_z'])
        
        if merged_df.empty:
            return True

        new_cols = {}
        for prefix in G_NEIGHBOR_PREFIXES:
            # Extract coordinates and convert to tuples for lookup
            # Coordinates in G_MAP_DF were already cast to int in init_worker
            nbr_x = merged_df[f"nbr_{prefix}_x"].values
            nbr_y = merged_df[f"nbr_{prefix}_y"].values
            nbr_z = merged_df[f"nbr_{prefix}_z"].values
            
            coords = list(zip(nbr_x, nbr_y, nbr_z))
            
            # Map coordinates to velocities. Default to 0.0 if neighbor is missing
            new_cols[f"velocity_{prefix}_x"] = [lookup_vx.get(c, 0.0) for c in coords]
            new_cols[f"velocity_{prefix}_y"] = [lookup_vy.get(c, 0.0) for c in coords]
            new_cols[f"velocity_{prefix}_z"] = [lookup_vz.get(c, 0.0) for c in coords]
            
        # Combine everything
        new_df = pd.DataFrame(new_cols, index=merged_df.index)
        final_df = pd.concat([merged_df[list(df_input.columns)], new_df], axis=1)
        
        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_pickle(output_file, compression='gzip')
        return True
    except Exception as e:
        return f"Error processing {input_file}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Create cubes per line by gathering neighbor velocities.")
    parser.add_argument("--first_only", action="store_true", help="Only process the first file and exit cleanly.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8).")
    args = parser.parse_args()
    
    # Paths as per issue context
    input_root = "/Users/kkreth/PycharmProjects/data/Cubed_OG_Data"
    map_path = "/Users/kkreth/PycharmProjects/cgan/cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv"
    output_root = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data"
    
    if not os.path.exists(map_path):
        print(f"Mapping file not found at {map_path}")
        return

    # Walk the input directory to find all .pkl.gz files
    files_to_process = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".pkl.gz"):
                files_to_process.append(os.path.join(root, file))
    
    # Sort for consistent behavior
    files_to_process.sort()
    
    if not files_to_process:
        print(f"No .pkl.gz files found in {input_root}")
        return

    if args.first_only:
        files_to_process = files_to_process[:1]
        print("Flag --first_only is set. Processing only the first file.")
    
    print(f"Starting parallel processing of {len(files_to_process)} file(s) with {args.workers} workers...")
    print("Interpretation of progress bar:")
    print("  Percentage | Completed/Total [Elapsed < Remaining, Speed]")
    
    start_time = time.time()
    
    # Prepare task arguments
    tasks = []
    for input_file in files_to_process:
        rel_path = os.path.relpath(input_file, input_root)
        output_file = os.path.join(output_root, rel_path)
        tasks.append((input_file, output_file))
    
    # Run in parallel using ProcessPoolExecutor
    # Using initializer to load the large map file once per process instead of serializing it thousands of times
    results_count = 0
    errors = []
    
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(map_path,)) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file_task, t[0], t[1]): t for t in tasks}
        
        # Track progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is True:
                results_count += 1
            else:
                errors.append(result)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted {results_count} files in {elapsed:.2f} seconds.")
    if errors:
        print(f"Encountered {len(errors)} errors during processing.")
        for err in errors[:5]: # Show first 5 errors
            print(err)

if __name__ == "__main__":
    main()
