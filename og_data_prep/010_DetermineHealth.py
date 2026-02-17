import os
import sys
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to sys.path to import HostPreferences if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Ordered_001_Initialize import HostPreferences
except ImportError:
    HostPreferences = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_file(file_path):
    """Analyze a single .pkl.gz file for time step health."""
    try:
        # Load the dataframe
        df = pd.read_pickle(file_path, compression='gzip')
        
        # Identify time column
        time_col = None
        for col in ['t', 'time']:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            return {
                'file': os.path.basename(file_path),
                'error': f"No time column found. Available: {list(df.columns)}"
            }
        
        time_data = df[time_col]
        unique_times = sorted(time_data.unique())
        time_counts = time_data.value_counts()
        
        min_time = min(unique_times) if unique_times else None
        max_time = max(unique_times) if unique_times else None
        num_steps = len(unique_times)
        
        # Check consistency of row counts per time step
        counts = time_counts.unique()
        is_consistent_count = len(counts) == 1
        common_count = counts[0] if is_consistent_count else None
        
        # Check for missing steps in the range found
        expected_steps = set(range(int(min_time), int(max_time) + 1)) if unique_times else set()
        actual_steps = set(unique_times)
        missing_steps = sorted(list(expected_steps - actual_steps))
        
        # Check if it covers 1-1200 specifically
        ideal_steps = set(range(1, 1201))
        missing_ideal_steps = sorted(list(ideal_steps - actual_steps))
        is_1_to_1200 = len(missing_ideal_steps) == 0
        
        return {
            'file': os.path.basename(file_path),
            'min_t': min_time,
            'max_t': max_time,
            'num_steps': num_steps,
            'consistent_rows': is_consistent_count,
            'rows_per_step': common_count if is_consistent_count else "Varies",
            'is_1_to_1200': is_1_to_1200,
            'missing_steps': missing_steps,
            'missing_ideal_steps': missing_ideal_steps,
            'total_rows': len(df)
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'error': str(e)
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze health of .pkl.gz files.")
    parser.add_argument("--dir", help="Directory to search for .pkl.gz files")
    args = parser.parse_args()

    # Try to get path from HostPreferences or use a default
    search_dir = args.dir
    if not search_dir and HostPreferences:
        try:
            prefs = HostPreferences()
            # Try a few common locations based on preferences
            possible_dirs = [
                os.path.join(prefs.root_path, "Unmodified_OG_Data"),
                prefs.raw_input,
                prefs.root_path
            ]
            for d in possible_dirs:
                if d and os.path.exists(d):
                    search_dir = d
                    break
        except Exception as e:
            logger.warning(f"Could not load preferences: {e}")

    if not search_dir or not os.path.exists(search_dir):
        # Fallback to current directory or a hardcoded path if needed
        search_dir = "/Users/kkreth/PycharmProjects/data/Unmodified_OG_Data"
        if not os.path.exists(search_dir):
            search_dir = "."

    print(f"Walking directory: {search_dir}")
    
    pkl_files = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.pkl.gz'):
                pkl_files.append(os.path.join(root, file))
    
    if not pkl_files:
        print("No .pkl.gz files found.")
        return

    print(f"Found {len(pkl_files)} files. Analyzing...")
    
    results = []
    # Using ProcessPoolExecutor for faster processing as reading pickles can be CPU intensive
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(analyze_file, pkl_files))
    
    # Report
    print("\n" + "="*100)
    print(f"{'File':<20} | {'Steps':<6} | {'Min T':<6} | {'Max T':<6} | {'Consistent':<10} | {'Rows/Step':<10} | {'1-1200':<7}")
    print("-" * 100)
    
    for res in sorted(results, key=lambda x: x['file']):
        if 'error' in res:
            print(f"{res['file']:<20} | ERROR: {res['error']}")
            continue
            
        print(f"{res['file']:<20} | {res['num_steps']:<6} | {res['min_t']:<6} | {res['max_t']:<6} | "
              f"{str(res['consistent_rows']):<10} | {str(res['rows_per_step']):<10} | {str(res['is_1_to_1200']):<7}")

    print("="*100)
    
    # Summary of inconsistencies
    print("\nDetailed Issues:")
    has_issues = False
    for res in sorted(results, key=lambda x: x['file']):
        if 'error' in res:
            continue
        
        issues = []
        if not res['is_1_to_1200']:
            m_ideal = res['missing_ideal_steps']
            if len(m_ideal) > 10:
                issues.append(f"Missing {len(m_ideal)} steps from 1-1200 range (e.g., {m_ideal[:5]}...)")
            else:
                issues.append(f"Missing steps from 1-1200: {m_ideal}")
                
        if not res['consistent_rows']:
            issues.append("Inconsistent row counts across time steps")
        
        if res['missing_steps']:
            m_range = res['missing_steps']
            if len(m_range) > 10:
                issues.append(f"Missing {len(m_range)} steps in own range {res['min_t']}-{res['max_t']} (e.g., {m_range[:5]}...)")
            else:
                issues.append(f"Missing steps in own range {res['min_t']}-{res['max_t']}: {m_range}")
            
        if issues:
            has_issues = True
            print(f"- {res['file']}: {'; '.join(issues)}")
            
    if not has_issues:
        print("No issues found in any file.")

if __name__ == "__main__":
    main()
