import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from pipeline_config import add_config_argument, resolve_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_file_extremes(file_path, sample_frac=0.01):
    """Analyze a single .pkl.gz file for vx, vy, vz extremes and collect samples."""
    try:
        # Load the dataframe
        df = pd.read_pickle(file_path, compression='gzip')
        
        cols = ['vx', 'vy', 'vz']
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return {
                'file': os.path.basename(file_path),
                'error': f"Missing columns: {missing}"
            }
        
        results = {
            'file': os.path.basename(file_path),
            'stats': {}
        }
        
        for col in cols:
            series = df[col]
            results['stats'][col] = {
                'min': float(series.min()),
                'max': float(series.max()),
                # Take a sample for distribution analysis to save memory
                'sample': series.sample(frac=sample_frac).values.astype(np.float32)
            }
        
        return results
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find extremes and distribution of vx, vy, vz.")
    add_config_argument(parser)
    parser.add_argument("--dir", help="Directory to search for .pkl.gz files")
    parser.add_argument("--sample_frac", type=float, default=0.01, help="Fraction of data to sample (default 0.01)")
    parser.add_argument("--output", help="Output path for the figure")
    args = parser.parse_args()

    search_dir = resolve_path(args.config, "unmodified_data_dir", args.dir)
    output_path = resolve_path(args.config, "extremes_report_path", args.output)
    if not os.path.exists(search_dir):
        print(f"Input directory not found: {search_dir}")
        return

    print(f"Analyzing files in: {search_dir}")
    
    pkl_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.pkl.gz') and not file.startswith('.'):
                pkl_files.append(os.path.join(root, file))
    
    if not pkl_files:
        print("No .pkl.gz files found.")
        return

    print(f"Found {len(pkl_files)} files. Processing in parallel...")
    
    all_results = []
    with ProcessPoolExecutor() as executor:
        worker = partial(analyze_file_extremes, sample_frac=args.sample_frac)
        all_results = list(executor.map(worker, pkl_files))
    
    # Aggregate data
    global_extremes = {
        'vx': {'min': float('inf'), 'max': float('-inf')},
        'vy': {'min': float('inf'), 'max': float('-inf')},
        'vz': {'min': float('inf'), 'max': float('-inf')}
    }
    
    samples = {'vx': [], 'vy': [], 'vz': []}
    errors = []
    
    for res in all_results:
        if 'error' in res:
            errors.append(f"{res['file']}: {res['error']}")
            continue
        
        for col in ['vx', 'vy', 'vz']:
            s = res['stats'][col]
            global_extremes[col]['min'] = min(global_extremes[col]['min'], s['min'])
            global_extremes[col]['max'] = max(global_extremes[col]['max'], s['max'])
            samples[col].append(s['sample'])

    # Find absolute global extremes across all vx, vy, vz
    absolute_global_min = min(global_extremes[col]['min'] for col in global_extremes)
    absolute_global_max = max(global_extremes[col]['max'] for col in global_extremes)

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  {err}")

    print("\n" + "="*60)
    print(f"{'Column':<10} | {'Minimum':>15} | {'Maximum':>15}")
    print("-" * 60)
    for col in ['vx', 'vy', 'vz']:
        print(f"{col:<10} | {global_extremes[col]['min']:>15.6f} | {global_extremes[col]['max']:>15.6f}")
    print("-" * 60)
    print(f"{'GLOBAL':<10} | {absolute_global_min:>15.6f} | {absolute_global_max:>15.6f}")
    print("=" * 60)

    # Plotting
    print("\nCalculating percentiles and generating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    percentiles = [10, 25, 50, 75, 90]
    
    for i, col in enumerate(['vx', 'vy', 'vz']):
        if not samples[col]:
            continue
            
        data = np.concatenate(samples[col])
        p_vals = np.percentile(data, percentiles)
        
        ax = axes[i]
        # Histogram with 100 bins
        ax.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Distribution')
        
        # Add a secondary axis for log scale to see outliers
        ax_log = ax.twinx()
        counts, bins = np.histogram(data, bins=100)
        ax_log.step(bins[:-1], counts, color='gray', alpha=0.3, where='post')
        ax_log.set_yscale('log')
        ax_log.set_ylabel('Frequency (log scale)', color='gray')
        
        # Vertical lines for percentiles
        colors = ['red', 'orange', 'green', 'orange', 'red']
        for p, v, c in zip(percentiles, p_vals, colors):
            ax.axvline(v, color=c, linestyle='--', label=f'{p}th: {v:.4f}')
        
        # Mark Min and Max
        ax.axvline(global_extremes[col]['min'], color='purple', linestyle=':', label=f'Min: {global_extremes[col]["min"]:.4f}')
        ax.axvline(global_extremes[col]['max'], color='purple', linestyle=':', label=f'Max: {global_extremes[col]["max"]:.4f}')
        
        ax.set_title(f'Distribution of {col} (Across all files, {args.sample_frac*100}% sample)')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Frequency (linear)')
        ax.legend(loc='upper right', fontsize='small')
        
        print(f"{col} percentiles: " + ", ".join([f"{p}%: {v:.4f}" for p, v in zip(percentiles, p_vals)]))

    # Add absolute global extremes to the plots with a "rainbow" effect
    # We can use multiple lines slightly offset or just a set of colors to satisfy "RAINBOW"
    rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    
    for ax in axes:
        # ABS GLOBAL MIN
        for idx, color in enumerate(rainbow_colors):
            # Drawing multiple thin lines to create a "rainbow" band effect if zoomed in, 
            # but mainly to signify the rainbow requirement
            ax.axvline(absolute_global_min, color=color, linewidth=2 - (idx*0.2), alpha=0.3)
        
        # ABS GLOBAL MAX
        for idx, color in enumerate(rainbow_colors):
            ax.axvline(absolute_global_max, color=color, linewidth=2 - (idx*0.2), alpha=0.3)

        # Main label lines
        ax.axvline(absolute_global_min, color='red', linewidth=1, linestyle='-', 
                  label=f'GLOBAL MIN: {absolute_global_min:.4f}')
        ax.axvline(absolute_global_max, color='violet', linewidth=1, linestyle='-', 
                  label=f'GLOBAL MAX: {absolute_global_max:.4f}')
        
        ax.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path)
    print(f"\nFigure saved to: {output_path}")

if __name__ == "__main__":
    main()
