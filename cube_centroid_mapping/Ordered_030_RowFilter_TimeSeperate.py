import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

class RowFilterTimeSeperate:
    def __init__(self):
        self.input_dir = "/Users/kkreth/PycharmProjects/data/simplified_output"
        self.output_parent_dir = "/Users/kkreth/PycharmProjects/data/simplified_output_broken_down_filtered"
        self.filter_csv_path = "/Users/kkreth/PycharmProjects/cgan/cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv"
        self.centroids = None

    def load_filter(self):
        """Load all unique x,y,z coordinate combinations from CSV into a set for fast lookup."""
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
        """Read a file, filter its rows, and split by time."""
        start_time = time.time()
        try:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            # Read the input file (pickle with gzip)
            df = pd.read_pickle(file_path, compression='gzip')
            
            # Filter rows where (x, y, z) is in the centroids set
            # We use a MultiIndex for efficient filtering
            mask = df.set_index(['x', 'y', 'z']).index.isin(self.centroids)
            df_filtered = df[mask].copy()
            
            if df_filtered.empty:
                print(f"No rows matched filter in {file_name}")
                return True

            # Prepare output subdirectory named after the input file
            # e.g., if file is 10p4.pkl.gz, subdirectory is 10p4
            base_name = file_name.split('.')[0]
            file_output_dir = os.path.join(self.output_parent_dir, base_name)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Group by time for efficient splitting
            grouped = df_filtered.groupby('time')
            
            count = 0
            # Iterate through the requested time periods 1-1200
            for t in range(1, 1201):
                if t in grouped.groups:
                    df_time = grouped.get_group(t)
                    # Write out 1 file for each time period
                    output_file = os.path.join(file_output_dir, f"{t:04d}.pkl.gz")
                    df_time.to_pickle(output_file, compression='gzip')
                    count += 1
            
            duration = time.time() - start_time
            print(f"Finished {file_name}: wrote {count} time files in {duration:.2f}s")
            return True
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            return False

    def run(self):
        self.load_filter()
        
        if not os.path.exists(self.input_dir):
            print(f"Input directory {self.input_dir} does not exist.")
            return

        # Ensure output parent directory exists
        os.makedirs(self.output_parent_dir, exist_ok=True)
        
        # Collect .pkl.gz files from input directory
        files = [os.path.join(self.input_dir, f) 
                 for f in os.listdir(self.input_dir) 
                 if f.endswith('.pkl.gz')]
        
        print(f"Found {len(files)} files to process.")
        
        # Use ThreadPoolExecutor for parallel processing
        # Adjust max_workers as needed based on RAM
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(self.process_file, files))

if __name__ == "__main__":
    processor = RowFilterTimeSeperate()
    processor.run()
