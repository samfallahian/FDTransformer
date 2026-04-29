import os
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

from pipeline_config import add_config_argument, resolve_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_current_dtypes(file_path):
    """Read only the first row or just the dtypes if possible."""
    try:
        df = pd.read_pickle(file_path, compression='gzip')
        return os.path.basename(file_path), df.dtypes.to_dict()
    except Exception as e:
        return os.path.basename(file_path), str(e)

def process_single_file(file_info: Dict[str, str]):
    input_path = file_info['input_path']
    output_path = file_info['output_path']
    
    try:
        old_size = os.path.getsize(input_path)
        df = pd.read_pickle(input_path, compression='gzip')
        
        # Handle time column name (it might be 't' or 'time')
        # The instruction says "time,x,y,z", implying the target or source name.
        # We will check both.
        time_col = None
        for col in ['time', 't']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col and time_col != 'time':
            df.rename(columns={time_col: 'time'}, inplace=True)
            time_col = 'time'

        # Type conversions
        # time, x, y, z to integer (32-bit)
        cols_to_int = ['time', 'x', 'y', 'z']
        for col in cols_to_int:
            if col in df.columns:
                df[col] = df[col].astype(np.int32)
            else:
                logger.warning(f"Column '{col}' not found in {os.path.basename(input_path)}")
        
        # vx, vy, vz to 32-bit float AND rename
        vel_cols = ['vx', 'vy', 'vz']
        rename_map = {}
        for col in vel_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
                rename_map[col] = f"original_{col}"
            else:
                logger.warning(f"Column '{col}' not found in {os.path.basename(input_path)}")
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save newly saved file
        df.to_pickle(output_path, compression='gzip')
        new_size = os.path.getsize(output_path)
        
        return {
            'file': os.path.basename(input_path),
            'success': True,
            'old_size': old_size,
            'new_size': new_size,
            'new_dtypes': df.dtypes.to_dict()
        }
    except Exception as e:
        return {
            'file': os.path.basename(input_path),
            'success': False,
            'error': str(e)
        }

def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = 1024 ** i
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Correct dtypes and rename columns in .pkl.gz files.")
    add_config_argument(parser)
    parser.add_argument("--dir", help="Directory to search for .pkl.gz files")
    parser.add_argument("--output_dir", help="Directory to save corrected files")
    args = parser.parse_args()

    search_dir = resolve_path(args.config, "unmodified_data_dir", args.dir)
    output_dir = resolve_path(args.config, "corrected_data_dir", args.output_dir)
    if not os.path.exists(search_dir):
        print(f"Input directory not found: {search_dir}")
        return

    print(f"Input directory: {search_dir}")
    print(f"Output directory: {output_dir}")
    
    pkl_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.pkl.gz') and not file.startswith('.'):
                pkl_files.append(os.path.join(root, file))
    
    if not pkl_files:
        print("No .pkl.gz files found.")
        return

    print(f"Found {len(pkl_files)} files.")

    # 1. Show dtypes for all files
    print("\n--- Current Data Types (First Thing) ---")
    with ProcessPoolExecutor() as executor:
        initial_dtypes = list(executor.map(get_current_dtypes, pkl_files))
    
    # Check if all files have the same dtypes to avoid spam
    unique_dtype_sets = {}
    for fname, dtypes in initial_dtypes:
        if isinstance(dtypes, str):
            print(f"Error reading {fname}: {dtypes}")
            continue
        
        # Convert dict to a hashable tuple for comparison
        dtype_tuple = tuple(sorted((k, str(v)) for k, v in dtypes.items()))
        if dtype_tuple not in unique_dtype_sets:
            unique_dtype_sets[dtype_tuple] = []
        unique_dtype_sets[dtype_tuple].append(fname)
    
    for dtype_tuple, files in unique_dtype_sets.items():
        print(f"\nFiles: {', '.join(files[:3])}{' and ' + str(len(files)-3) + ' more' if len(files) > 3 else ''}")
        print("-" * 30)
        for col, dtype in dtype_tuple:
            print(f"{col:<15}: {dtype}")

    # 2. Perform conversions
    print("\n--- Performing dType Corrections and Renaming ---")
    file_infos = [
        {'input_path': f, 'output_path': os.path.join(output_dir, os.path.basename(f))}
        for f in pkl_files
    ]
    
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, file_infos))
    
    # 3. Confirmation and Report
    print("\n--- Confirmation of New Data Types ---")
    success_results = [r for r in results if r['success']]
    if success_results:
        # Show dtypes of the first successful one
        first_success = success_results[0]
        print(f"Example from {first_success['file']}:")
        for col, dtype in first_success['new_dtypes'].items():
            print(f"{col:<20}: {dtype}")
    
    print("\n--- File Size Savings Report ---")
    print(f"{'File':<25} | {'Old Size':<12} | {'New Size':<12} | {'Savings':<12}")
    print("-" * 70)
    
    total_old = 0
    total_new = 0
    for res in sorted(results, key=lambda x: x['file']):
        if not res['success']:
            print(f"{res['file']:<25} | ERROR: {res['error']}")
            continue
        
        old = res['old_size']
        new = res['new_size']
        total_old += old
        total_new += new
        savings = old - new
        percent = (savings / old * 100) if old > 0 else 0
        
        print(f"{res['file']:<25} | {format_size(old):<12} | {format_size(new):<12} | {format_size(savings):<12} ({percent:.1f}%)")

    print("-" * 70)
    grand_savings = total_old - total_new
    grand_percent = (grand_savings / total_old * 100) if total_old > 0 else 0
    print(f"{'TOTAL':<25} | {format_size(total_old):<12} | {format_size(total_new):<12} | {format_size(grand_savings):<12} ({grand_percent:.1f}%)")

if __name__ == "__main__":
    main()
