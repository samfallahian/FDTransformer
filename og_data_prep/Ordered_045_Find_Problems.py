"""
Ordered_045_Find_Problems.py

This script scans simulation data files in 'Scaled_OG_Data' and checks if 
all required coordinates (from the centroid filter CSV) are present 
for every time step (1-1200).

Unlike the RowFilter script, this one does not crash on the first error. 
Instead, it collects all missing data occurrences and produces a report.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

class ProblemFinder:
    def __init__(self):
        # Paths as defined in the pipeline
        self.input_dir = "/Users/kkreth/PycharmProjects/data/Scaled_OG_Data"
        self.filter_csv_path = "/Users/kkreth/PycharmProjects/cgan/cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv"
        self.required_coords = None
        self.problems = []

    def load_filter(self):
        """Load all unique x,y,z coordinate combinations from the filter CSV."""
        print(f"Loading filter from {self.filter_csv_path}...")
        try:
            df_filter = pd.read_csv(self.filter_csv_path)
            all_coords = set()
            
            # Extract unique (x, y, z) triplets from centroid and neighbor columns
            x_cols = [c for c in df_filter.columns if c.endswith('_x')]
            
            for x_col in x_cols:
                base = x_col[:-2]
                y_col = base + '_y'
                z_col = base + '_z'
                if y_col in df_filter.columns and z_col in df_filter.columns:
                    triplets = zip(df_filter[x_col].astype(int), 
                                   df_filter[y_col].astype(int), 
                                   df_filter[z_col].astype(int))
                    all_coords.update(triplets)
            
            self.required_coords = all_coords
            print(f"Loaded {len(self.required_coords)} unique coordinate combinations required for each time step.")
        except Exception as e:
            print(f"Error loading filter CSV: {e}")
            raise

    def process_file(self, file_path):
        """Check a single file for missing coordinates across all time steps."""
        file_name = os.path.basename(file_path)
        try:
            # Read simulation data
            df = pd.read_pickle(file_path, compression='gzip')
            
            # Use MultiIndex for efficient membership testing
            df_indexed = df.set_index(['x', 'y', 'z'])
            
            # Filter to only the rows we expect to find
            df_filtered = df[df_indexed.index.isin(self.required_coords)].copy()
            
            # Group by time and get counts per step
            counts = df_filtered.groupby('time').size()
            expected_count = len(self.required_coords)
            
            # Check each expected time step (1-1200)
            for t in range(1, 1201):
                actual_count = counts.get(t, 0)
                
                if actual_count < expected_count:
                    if actual_count == 0:
                        self.problems.append({
                            'file': file_name,
                            'time': t,
                            'missing_count': expected_count,
                            'issue': 'Time step completely missing or no coords match filter',
                            'sample_missing': 'ALL'
                        })
                    else:
                        # Find exactly which coordinates are missing for this specific step
                        df_time = df_filtered[df_filtered['time'] == t]
                        present_coords = set(zip(df_time['x'], df_time['y'], df_time['z']))
                        missing = self.required_coords - present_coords
                        
                        self.problems.append({
                            'file': file_name,
                            'time': t,
                            'missing_count': len(missing),
                            'issue': 'Partial data missing',
                            'sample_missing': str(list(missing)[:3])
                        })
            
            # Free up memory
            del df
            del df_filtered
            del df_indexed
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            self.problems.append({
                'file': file_name,
                'time': 'N/A',
                'missing_count': 'N/A',
                'issue': f'CRITICAL ERROR: {str(e)}',
                'sample_missing': 'N/A'
            })

    def run(self):
        self.load_filter()
        
        if not os.path.exists(self.input_dir):
            print(f"Input directory {self.input_dir} does not exist.")
            return

        # Collect files to check
        files = sorted([os.path.join(self.input_dir, f) 
                        for f in os.listdir(self.input_dir) 
                        if f.endswith('.pkl.gz') and not f.startswith('.')])
        
        if not files:
            print(f"No .pkl.gz files found in {self.input_dir}")
            return

        print(f"Found {len(files)} files to scan.")
        
        for f in tqdm(files, desc="Scanning for problems"):
            self.process_file(f)

        self.generate_report()

    def generate_report(self):
        """Print a summary and save a detailed report to CSV."""
        print("\n" + "="*60)
        print("DATA INTEGRITY REPORT")
        print("="*60)
        
        if not self.problems:
            print("Status: SUCCESS")
            print("Result: No missing coordinates found. All files are complete for steps 1-1200.")
        else:
            df_probs = pd.DataFrame(self.problems)
            
            unique_files_with_issues = df_probs['file'].nunique()
            print(f"Status: PROBLEMS FOUND")
            print(f"Total time steps affected: {len(df_probs)}")
            print(f"Files affected: {unique_files_with_issues}")
            
            # Print a snippet of the issues
            print("\nSnippet of detected issues:")
            display_cols = ['file', 'time', 'missing_count', 'issue']
            print(df_probs[display_cols].head(20).to_string(index=False))
            
            if len(df_probs) > 20:
                print(f"\n... and {len(df_probs) - 20} more rows in the full report.")
            
            # Save the full detailed report
            report_path = os.path.join(os.path.dirname(__file__), "data_problems_report.csv")
            df_probs.to_csv(report_path, index=False)
            print(f"\nFull detailed report saved to: {report_path}")
            
            # Print summary by file
            print("\nIssues count per file:")
            summary = df_probs.groupby('file').size().reset_index(name='affected_steps')
            print(summary.to_string(index=False))

if __name__ == "__main__":
    finder = ProblemFinder()
    finder.run()
