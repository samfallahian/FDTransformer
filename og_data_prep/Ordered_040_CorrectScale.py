import os
import sys
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

# Add project root to sys.path to import TransformLatent and HostPreferences
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from TransformLatent import FloatConverter
try:
    from Ordered_001_Initialize import HostPreferences
except ImportError:
    HostPreferences = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(file_info: Dict[str, Any]):
    input_path = file_info['input_path']
    output_path = file_info['output_path']
    # Instantiate converter inside the worker if needed, 
    # but passing it should work if it's pickleable.
    converter = file_info['converter']
    
    try:
        df = pd.read_pickle(input_path, compression='gzip')
        
        # New columns based on original_vx, original_vy, original_vz
        vel_cols = ['vx', 'vy', 'vz']
        for col in vel_cols:
            orig_col = f"original_{col}"
            if orig_col in df.columns:
                # Use the converter to set values between 0 and 1
                # Ensure it's 32-bit float
                df[col] = converter.convert(df[orig_col]).astype(np.float32)
            else:
                logger.warning(f"Column '{orig_col}' not found in {os.path.basename(input_path)}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save newly scaled file
        df.to_pickle(output_path, compression='gzip')
        
        # Collect sample stats for reporting
        stats = {}
        for col in vel_cols:
            if col in df.columns:
                stats[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return {
            'file': os.path.basename(input_path),
            'success': True,
            'stats': stats,
            'new_dtypes': df.dtypes.to_dict()
        }
    except Exception as e:
        return {
            'file': os.path.basename(input_path),
            'success': False,
            'error': str(e)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scale velocity columns between 0 and 1.")
    parser.add_argument("--dir", help="Directory to search for .pkl.gz files (e.g. Corrected_OG_Data)")
    parser.add_argument("--output_dir", help="Directory to save scaled files")
    args = parser.parse_args()

    # Determine input directory
    search_dir = args.dir
    if not search_dir and HostPreferences:
        try:
            prefs = HostPreferences()
            # Try to find Corrected_OG_Data relative to root_path or raw_input
            possible_dirs = [
                os.path.join(prefs.root_path, "Corrected_OG_Data"),
                os.path.join(os.path.dirname(prefs.raw_input.rstrip('/')), "Corrected_OG_Data"),
                "/Users/kkreth/PycharmProjects/data/Corrected_OG_Data"
            ]
            for d in possible_dirs:
                if d and os.path.exists(d):
                    search_dir = d
                    break
        except Exception as e:
            logger.warning(f"Could not load preferences: {e}")

    if not search_dir or not os.path.exists(search_dir):
        # Last resort fallback
        search_dir = "/Users/kkreth/PycharmProjects/data/Corrected_OG_Data"
        if not os.path.exists(search_dir):
            print(f"Error: Input directory {search_dir} not found.")
            return

    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        # Default to a "Scaled_OG_Data" next to the input
        parent = os.path.dirname(search_dir.rstrip('/'))
        output_dir = os.path.join(parent, "Scaled_OG_Data")

    print(f"Input directory: {search_dir}")
    print(f"Output directory: {output_dir}")
    
    pkl_files = [
        os.path.join(search_dir, f) for f in os.listdir(search_dir) 
        if f.endswith('.pkl.gz') and not f.startswith('.')
    ]
    
    if not pkl_files:
        print("No .pkl.gz files found.")
        return

    print(f"Found {len(pkl_files)} files.")

    converter = FloatConverter()
    print(f"Initialized FloatConverter with min={converter.min_value}, max={converter.max_value}")

    file_infos = [
        {
            'input_path': f, 
            'output_path': os.path.join(output_dir, os.path.basename(f)),
            'converter': converter
        }
        for f in pkl_files
    ]
    
    print("\nProcessing files in parallel...")
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, file_infos))
    
    # Report results
    success_count = sum(1 for r in results if r['success'])
    print(f"\nSuccessfully processed {success_count}/{len(results)} files.")

    print("\n--- Scaling Verification (Min/Max of scaled columns) ---")
    print(f"{'File':<25} | {'vx Min/Max':<20} | {'vy Min/Max':<20} | {'vz Min/Max':<20}")
    print("-" * 95)
    
    for res in sorted(results, key=lambda x: x['file']):
        if not res['success']:
            print(f"{res['file']:<25} | ERROR: {res['error']}")
            continue
        
        stats = res['stats']
        vx_str = f"{stats.get('vx', {}).get('min', 0):.4f}/{stats.get('vx', {}).get('max', 0):.4f}"
        vy_str = f"{stats.get('vy', {}).get('min', 0):.4f}/{stats.get('vy', {}).get('max', 0):.4f}"
        vz_str = f"{stats.get('vz', {}).get('min', 0):.4f}/{stats.get('vz', {}).get('max', 0):.4f}"
        
        print(f"{res['file']:<25} | {vx_str:<20} | {vy_str:<20} | {vz_str:<20}")

    if success_count > 0:
        # Filter for first successful result
        first_success = next(r for r in results if r['success'])
        print("\n--- Final dTypes Confirmation ---")
        for col, dtype in first_success['new_dtypes'].items():
            if col in ['vx', 'vy', 'vz', 'original_vx', 'original_vy', 'original_vz']:
                print(f"{col:<20}: {dtype}")

if __name__ == "__main__":
    main()
