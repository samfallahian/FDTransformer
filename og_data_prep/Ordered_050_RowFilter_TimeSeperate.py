"""
Ordered_050_RowFilter_TimeSeperate.py

This script filters large simulation data files based on a set of specific 
coordinate triplets (centroids and their neighbors) and then reorganizes 
the filtered data into time-separated files.

It verifies that ALL required coordinates are present for every time step.
If any data is missing, it explains where and crashes gracefully.

Largely based on cube_centroid_mapping/Ordered_030_RowFilter_TimeSeperate.py
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import argparse
from tqdm import tqdm

from pipeline_config import add_config_argument, resolve_path

class RowFilterTimeSeperate:
    def __init__(self, input_dir: str, output_parent_dir: str, filter_csv_path: str):
        self.input_dir = input_dir
        self.output_parent_dir = output_parent_dir
        self.filter_csv_path = filter_csv_path
        self.centroids = None

    def load_filter(self):
        """
        Load all unique x,y,z coordinate combinations from the filter CSV into a set.
        These are the points needed to form full cubes later.
        """
        print(f"Loading filter from {self.filter_csv_path}...")
        try:
            df_filter = pd.read_csv(self.filter_csv_path)
            
            all_coords = set()
            
            # Extract unique (x, y, z) triplets from all coordinate columns
            # This includes centroid_x,y,z AND all nbr_..._x,y,z columns
            x_cols = [c for c in df_filter.columns if c.endswith('_x')]
            
            for x_col in x_cols:
                base = x_col[:-2]
                y_col = base + '_y'
                z_col = base + '_z'
                if y_col in df_filter.columns and z_col in df_filter.columns:
                    # Zip the columns and add to the set
                    triplets = zip(df_filter[x_col].astype(int), 
                                   df_filter[y_col].astype(int), 
                                   df_filter[z_col].astype(int))
                    all_coords.update(triplets)
            
            self.centroids = all_coords
            print(f"Loaded {len(self.centroids)} unique coordinate combinations (centroids + neighbors).")
        except Exception as e:
            print(f"Error loading filter CSV: {e}")
            raise

    def process_file(self, file_path):
        """Read a file, filter its rows, verify completeness, and split by time."""
        try:
            file_name = os.path.basename(file_path)
            
            # Read the input file (pickle with gzip)
            df = pd.read_pickle(file_path, compression='gzip')
            
            # Filter rows where (x, y, z) is in the centroids set
            # We use a MultiIndex for efficient filtering
            df_indexed = df.set_index(['x', 'y', 'z'])
            mask = df_indexed.index.isin(self.centroids)
            df_filtered = df[mask].copy()
            
            # Group by time for efficient splitting
            grouped = df_filtered.groupby('time')

            # Prepare output subdirectory named after the input file
            # e.g., if file is 10p4.pkl.gz, subdirectory is 10p4
            base_name = file_name.replace('.pkl.gz', '')
            file_output_dir = os.path.join(self.output_parent_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Iterate through the requested time periods 1-1200
            for t in range(1, 1201):
                if t not in grouped.groups:
                    # Data is missing for this time step
                    # Let's find exactly what is missing to explain the crash
                    df_time_orig = df[df['time'] == t]
                    if df_time_orig.empty:
                        print(f"\nCRITICAL ERROR: Time step {t} is completely missing in {file_name}")
                    else:
                        present_coords = set(zip(df_time_orig['x'], df_time_orig['y'], df_time_orig['z']))
                        missing = self.centroids - present_coords
                        print(f"\nCRITICAL ERROR: Time step {t} in {file_name} is missing {len(missing)} required coordinates.")
                        print(f"Sample of missing coordinates: {list(missing)[:10]}")
                    sys.exit(1)
                
                df_time = grouped.get_group(t)
                
                # Check if all expected coordinates are present in this group
                if len(df_time) < len(self.centroids):
                    # Some coordinates are missing for this time step
                    df_time_orig = df[df['time'] == t]
                    present_coords = set(zip(df_time_orig['x'], df_time_orig['y'], df_time_orig['z']))
                    missing = self.centroids - present_coords
                    print(f"\nCRITICAL ERROR: Time step {t} in {file_name} is missing {len(missing)} required coordinates.")
                    print(f"Expected {len(self.centroids)} coordinates, but only {len(df_time)} were found in the filtered set.")
                    print(f"Sample of missing coordinates: {list(missing)[:10]}")
                    sys.exit(1)
                
                # Write out 1 file for each time period
                output_file = os.path.join(file_output_dir, f"{t:04d}.pkl.gz")
                df_time.to_pickle(output_file, compression='gzip')
            
            return True
        except Exception as e:
            print(f"\nError processing {os.path.basename(file_path)}: {e}")
            sys.exit(1)

    def run(self):
        self.load_filter()
        
        if not os.path.exists(self.input_dir):
            print(f"Input directory {self.input_dir} does not exist.")
            return

        # Ensure output parent directory exists
        os.makedirs(self.output_parent_dir, exist_ok=True)
        
        # Collect .pkl.gz files from input directory
        files = sorted([os.path.join(self.input_dir, f) 
                 for f in os.listdir(self.input_dir) 
                 if f.endswith('.pkl.gz') and not f.startswith('.')])
        
        if not files:
            print(f"No .pkl.gz files found in {self.input_dir}")
            return

        print(f"Found {len(files)} files to process.")
        
        # Process files sequentially with tqdm for progress tracking
        # Sequential processing ensures clean console output and immediate termination on missing data
        for f in tqdm(files, desc="Processing files"):
            self.process_file(f)
        
        print("\nAll files processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter scaled data and split each file into per-time-step files.")
    add_config_argument(parser)
    parser.add_argument("--input_dir", help="Directory containing scaled .pkl.gz files.")
    parser.add_argument("--output_dir", help="Directory to write cubed, time-separated files.")
    parser.add_argument("--filter_csv", help="Centroid-neighbor mapping CSV.")
    args = parser.parse_args()

    processor = RowFilterTimeSeperate(
        input_dir=resolve_path(args.config, "scaled_data_dir", args.input_dir),
        output_parent_dir=resolve_path(args.config, "cubed_data_dir", args.output_dir),
        filter_csv_path=resolve_path(args.config, "cube_mapping_csv", args.filter_csv),
    )
    processor.run()
