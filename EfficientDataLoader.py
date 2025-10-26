import os
import pickle
import random
import numpy as np
from typing import List, Iterator, Dict, Any, Union, Tuple
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import gzip
import re
import time
from collections import Counter
import hashlib
from threading import RLock


# ANSI color codes
class Colors:
    ORANGE = '\033[38;5;208m'  # Orange color
    RESET = '\033[0m'          # Reset to default


class EfficientDataLoader:
    """
    A memory-efficient dataloader for sampling rows from multiple pickle files.
    Optimized for handling velocity data with columns named in the pattern:
    vx_1, vy_1, vz_1, ..., vx_125, vy_125, vz_125 (total 375 columns).
    Supports both regular and compressed (gzipped) pickle files.
    """
    
    def __init__(
        self, 
        root_directory: str,
        batch_size: int = 32, 
        num_workers: int = 4,
        cache_size: int = 10,
        shuffle: bool = True,
        seed: int = None,
        pin_memory: bool = False,
        enable_manifest_cache: bool = True,
        cache_filename: str = ".efficient_dataloader_cache.pkl",
        enable_profiling: bool = True

    ):
        """
        Initialize the dataloader.
        
        Args:
            root_directory: Root directory to recursively search for .pkl files
            batch_size: Number of rows to sample in each batch
            num_workers: Number of parallel workers for file operations
            cache_size: Maximum number of files to keep in memory cache
            shuffle: Whether to shuffle the batches
            seed: Random seed for reproducibility
        """
        self.root_directory = root_directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory  # Store the pin_memory parameter
        self.enable_manifest_cache = enable_manifest_cache
        self.cache_path = os.path.join(self.root_directory, cache_filename)
        self.enable_profiling = enable_profiling
        self.profiling: Dict[str, Any] = {"timings": {}, "notes": {}}
        # Thread-safety for in-memory cache
        self.cache_lock = RLock()

        t0 = time.perf_counter()

        # Total number of velocity values expected (vx, vy, vz for 125 points)
        self.total_velocity_values = 375
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Try to load cached manifest/metadata if enabled
        self.all_files = None
        self.file_metadata = None
        cache_loaded = False
        if self.enable_manifest_cache and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and cached.get("root_directory") == self.root_directory:
                    # Recompute current manifest hash quickly (stat only) and compare
                    current_manifest = self._build_directory_manifest()
                    current_hash = self._hash_manifest(current_manifest)
                    if cached.get("manifest_hash") == current_hash:
                        self.all_files = [item["path"] for item in current_manifest]
                        self.file_metadata = cached.get("file_metadata")
                        cache_loaded = True
                        self.profiling["notes"]["cache"] = "Loaded metadata from cache (hash match)."
                    else:
                        self.profiling["notes"]["cache"] = "Cache invalid: manifest hash mismatch. Recomputing."
                else:
                    self.profiling["notes"]["cache"] = "Cache file exists but is incompatible. Recomputing."
            except Exception as e:
                self.profiling["notes"]["cache_error"] = f"Failed to load cache: {e}"

        t1 = time.perf_counter()
        if self.enable_profiling:
            self.profiling["timings"]["cache_check_seconds"] = t1 - t0

        if not cache_loaded:
            # Find all pickle files in subdirectories (manifest)
            t_start_list = time.perf_counter()
            manifest = self._build_directory_manifest()
            self.all_files = [item["path"] for item in manifest]
            t_end_list = time.perf_counter()
            if not self.all_files:
                raise ValueError(f"No .pkl files found in {root_directory} or its subdirectories")
            if self.enable_profiling:
                self.profiling["timings"]["list_files_seconds"] = t_end_list - t_start_list
            
            # Initialize file cache
            self.file_cache = {}  # {file_path: dataframe}
            
            # Pre-compute file metadata for better sampling
            t_start_meta = time.perf_counter()
            self.file_metadata = self._compute_file_metadata()
            t_end_meta = time.perf_counter()
            if self.enable_profiling:
                self.profiling["timings"]["metadata_seconds"] = t_end_meta - t_start_meta

            # Save cache
            if self.enable_manifest_cache:
                try:
                    cache_payload = {
                        "root_directory": self.root_directory,
                        "manifest": manifest,
                        "manifest_hash": self._hash_manifest(manifest),
                        "file_metadata": self.file_metadata,
                        "created_at": time.time(),
                    }
                    with open(self.cache_path, "wb") as f:
                        pickle.dump(cache_payload, f)
                except Exception as e:
                    self.profiling["notes"]["cache_save_error"] = f"Failed to save cache: {e}"
        else:
            # Ensure runtime structures are present when loading from cache
            self.file_cache = {}
            if not self.file_metadata or not self.all_files:
                raise ValueError("Cache loaded but missing required entries 'file_metadata' or 'all_files'.")
    
    def _find_all_pkl_files(self) -> List[str]:
        """Find all .pkl files in the root directory and subdirectories."""
        return glob.glob(os.path.join(self.root_directory, "**/*.pkl"), recursive=True)

    def _build_directory_manifest(self) -> List[Dict[str, Any]]:
        """Build a manifest of .pkl files with lightweight attributes for change detection."""
        files = self._find_all_pkl_files()
        manifest: List[Dict[str, Any]] = []
        for p in files:
            try:
                st = os.stat(p)
                manifest.append({
                    "path": p,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                })
            except FileNotFoundError:
                # File disappeared between listing and stat; skip
                continue
        # Sort manifest deterministically by path to ensure stable hashing
        manifest.sort(key=lambda x: x["path"])
        return manifest

    def _hash_manifest(self, manifest: List[Dict[str, Any]]) -> str:
        """Hash the manifest content to detect changes without opening files."""
        hasher = hashlib.sha256()
        for item in manifest:
            hasher.update(item["path"].encode("utf-8", errors="ignore"))
            hasher.update(str(item["size"]).encode("ascii"))
            hasher.update(str(item["mtime"]).encode("ascii"))
        return hasher.hexdigest()
    
    def _is_gzipped(self, file_path: str) -> bool:
        """Check if a file is gzipped by examining its magic number."""
        with open(file_path, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'
    
    def _load_pickle_file(self, file_path: str) -> Any:
        """Load a pickle file, automatically detecting if it's compressed."""
        try:
            # First check if the file is gzipped
            is_gzipped = self._is_gzipped(file_path)
            
            if is_gzipped:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load {file_path}: {e}")
    
    def _get_ordered_velocity_columns(self, df) -> List[str]:
        """
        Get velocity columns in the correct order: vx_1, vy_1, vz_1, ..., vx_125, vy_125, vz_125
        Returns an empty list if the required columns are not present.
        """
        # Check if we have velocity columns
        velocity_pattern = re.compile(r'v[xyz]_\d+')
        velocity_cols = [col for col in df.columns if velocity_pattern.match(col)]
        
        if not velocity_cols:
            return []
        
        # Extract unique indices from column names
        indices = set()
        for col in velocity_cols:
            match = re.search(r'_(\d+)$', col)
            if match:
                indices.add(int(match.group(1)))
        
        # Check if we have the expected number of points (125)
        if len(indices) != 125:
            print(f"Warning: Found {len(indices)} unique points instead of expected 125")
        
        # Sort indices numerically
        sorted_indices = sorted(indices)
        
        # Create the properly ordered column list
        ordered_columns = []
        for idx in sorted_indices:
            # Add columns in vx, vy, vz order for each index
            for component in ['vx', 'vy', 'vz']:
                col_name = f"{component}_{idx}"
                if col_name in df.columns:
                    ordered_columns.append(col_name)
                else:
                    # This column is missing, which is unexpected
                    print(f"Warning: Expected column {col_name} not found in dataframe")
                    # Use placeholder or return empty to indicate issue
                    return []
        
        # Ensure we have exactly 375 velocity values
        if len(ordered_columns) != self.total_velocity_values:
            print(f"Warning: Found {len(ordered_columns)} velocity columns instead of expected {self.total_velocity_values}")
            return []
            
        return ordered_columns
    
    def _compute_file_metadata(self) -> List[Dict[str, Any]]:
        """Pre-compute metadata about each file (row count, etc.) for efficient sampling."""
        metadata = []
        
        def process_file(file_path):
            try:
                # Load the file with automatic compression detection
                df_sample = self._load_pickle_file(file_path)
                
                # Get row count
                row_count = len(df_sample)
                
                # Get ordered velocity columns
                velocity_columns = self._get_ordered_velocity_columns(df_sample)
                has_velocity = len(velocity_columns) == self.total_velocity_values
                
                return {
                    'file_path': file_path,
                    'row_count': row_count,
                    'has_velocity': has_velocity,
                    'velocity_columns': velocity_columns if has_velocity else [],
                    'is_gzipped': self._is_gzipped(file_path)
                }
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                return None
        
        # Process metadata in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_file, self.all_files))
        
        # Filter out any failures
        metadata = [m for m in results if m is not None]
        
        # Verify we have files with velocity data
        valid_files = [m for m in metadata if m['has_velocity']]
        if not valid_files:
            raise ValueError(f"No files with complete velocity columns found in {self.root_directory}")
            
        return valid_files
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Preload file into the cache if it's not already loaded and shuffle the data once.
        Thread-safe around cache access.
        """
        t0 = time.perf_counter() if self.enable_profiling else None
        with self.cache_lock:
            cached = self.file_cache.get(file_path)
        if cached is not None:
            if self.enable_profiling:
                dt = time.perf_counter() - t0
                self.profiling["timings"].setdefault("_load_file_seconds_total", 0.0)
                self.profiling["timings"]["_load_file_seconds_total"] += dt
                self.profiling["timings"].setdefault("_load_file_calls", 0)
                self.profiling["timings"]["_load_file_calls"] += 1
            return cached

        # Load the file fully or partially, depending on mode
        df = self._load_pickle_file(file_path)

        if self.shuffle:
            # Shuffle data initially to avoid repeated shuffles
            df = df.sample(frac=1, random_state=42)

        # Store in cache with eviction
        with self.cache_lock:
            if len(self.file_cache) >= self.cache_size:
                # Remove the oldest cached file to respect the cache size
                try:
                    self.file_cache.pop(next(iter(self.file_cache)))
                except StopIteration:
                    pass
            self.file_cache[file_path] = df

        if self.enable_profiling:
            dt = time.perf_counter() - t0
            self.profiling["timings"].setdefault("_load_file_seconds_total", 0.0)
            self.profiling["timings"]["_load_file_seconds_total"] += dt
            self.profiling["timings"].setdefault("_load_file_calls", 0)
            self.profiling["timings"]["_load_file_calls"] += 1
        return df
    
    def _sample_rows_from_file(self, file_metadata: Dict[str, Any], num_rows: int) -> Tuple[np.ndarray, str]:
        """Sample a specific number of rows from a file."""
        t0 = time.perf_counter() if self.enable_profiling else None
        file_path = file_metadata['file_path']
        row_count = file_metadata['row_count']
        
        # Generate random row indices
        row_indices = np.random.choice(row_count, min(num_rows, row_count), replace=False)
        
        # Load the file
        df = self._load_file(file_path)
        
        # Extract velocity columns in the correct order
        vel_columns = file_metadata['velocity_columns']
        velocity_data = df.iloc[row_indices][vel_columns].values
        
        if self.enable_profiling:
            dt = time.perf_counter() - t0
            self.profiling["timings"].setdefault("_sample_rows_seconds_total", 0.0)
            self.profiling["timings"]["_sample_rows_seconds_total"] += dt
            self.profiling["timings"].setdefault("_sample_rows_calls", 0)
            self.profiling["timings"]["_sample_rows_calls"] += 1
        return velocity_data, file_path
    
    def get_batch(self, NUMBER_OF_ROWS: int, ROW_FLOOR: int = 20) -> Dict[str, Union[np.ndarray, List[str]]]:
        """
        Get a batch of randomly sampled rows from random files, ensuring a minimum number of
        rows (`ROW_FLOOR`) are sampled from any given file.

        Args:
            NUMBER_OF_ROWS: Total number of rows to sample in the batch.
            ROW_FLOOR: Minimum number of rows to sample from any file.
        
        Returns:
            Dictionary with:
                'velocity_data': Numpy array of shape (NUMBER_OF_ROWS, 375)
                'source_files': List of source file paths for each row.
        """
        t_total0 = time.perf_counter() if self.enable_profiling else None
        # Calculate file sampling weights based on row counts
        t0 = time.perf_counter() if self.enable_profiling else None
        weights = [m['row_count'] for m in self.file_metadata]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select files for the current batch
        max_files = NUMBER_OF_ROWS // ROW_FLOOR  # Max number of files to sample from
        selected_file_indices = np.random.choice(
            len(self.file_metadata),
            size=min(max_files, len(self.file_metadata)),
            replace=False,
            p=weights
        )
        if self.enable_profiling:
            self.profiling['timings'].setdefault('get_batch_select_seconds_total', 0.0)
            self.profiling['timings']['get_batch_select_seconds_total'] += (time.perf_counter() - t0)
            self.profiling['timings'].setdefault('get_batch_calls', 0)
            self.profiling['timings']['get_batch_calls'] += 1

        # Determine rows to sample from each selected file
        t1 = time.perf_counter() if self.enable_profiling else None
        file_sample_counts = {}
        leftover_rows = NUMBER_OF_ROWS

        # Assign at least ROW_FLOOR rows to each selected file
        for idx in selected_file_indices:
            rows_to_sample = min(self.file_metadata[idx]['row_count'], ROW_FLOOR)
            file_sample_counts[idx] = rows_to_sample
            leftover_rows -= rows_to_sample

        # Distribute any remaining rows among the selected files
        if leftover_rows > 0:
            for idx in selected_file_indices:
                if leftover_rows == 0:
                    break
                extra_rows = min(self.file_metadata[idx]['row_count'] - file_sample_counts[idx], leftover_rows)
                file_sample_counts[idx] += extra_rows
                leftover_rows -= extra_rows
        if self.enable_profiling:
            self.profiling['timings'].setdefault('get_batch_allocation_seconds_total', 0.0)
            self.profiling['timings']['get_batch_allocation_seconds_total'] += (time.perf_counter() - t1)

        # Sample rows from each selected file (parallelized across files)
        t2 = time.perf_counter() if self.enable_profiling else None
        all_velocity_data = []
        source_files = []

        if file_sample_counts:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for file_idx, num_rows in file_sample_counts.items():
                    futures.append(executor.submit(
                        self._sample_rows_from_file,
                        self.file_metadata[file_idx],
                        num_rows
                    ))
                for fut in futures:
                    velocity_data, file_path = fut.result()
                    all_velocity_data.append(velocity_data)
                    source_files.extend([file_path] * len(velocity_data))
        if self.enable_profiling:
            self.profiling['timings'].setdefault('get_batch_sampling_seconds_total', 0.0)
            self.profiling['timings']['get_batch_sampling_seconds_total'] += (time.perf_counter() - t2)

        # Combine all sampled data
        t3 = time.perf_counter() if self.enable_profiling else None
        combined_velocity_data = np.vstack(all_velocity_data)

        # Adjust to exactly NUMBER_OF_ROWS if more rows were sampled
        if len(combined_velocity_data) > NUMBER_OF_ROWS:
            indices = np.random.choice(len(combined_velocity_data), NUMBER_OF_ROWS, replace=False)
            combined_velocity_data = combined_velocity_data[indices]
            source_files = [source_files[i] for i in indices]

        # Shuffle the data if requested
        if self.shuffle:
            shuffle_indices = np.random.permutation(len(combined_velocity_data))
            combined_velocity_data = combined_velocity_data[shuffle_indices]
            source_files = [source_files[i] for i in shuffle_indices]
        if self.enable_profiling:
            self.profiling['timings'].setdefault('get_batch_concat_shuffle_seconds_total', 0.0)
            self.profiling['timings']['get_batch_concat_shuffle_seconds_total'] += (time.perf_counter() - t3)

        if self.enable_profiling:
            self.profiling['timings'].setdefault('get_batch_seconds_total', 0.0)
            self.profiling['timings']['get_batch_seconds_total'] += (time.perf_counter() - t_total0)
        return {
            'velocity_data': combined_velocity_data,
            'source_files': source_files
        }
    
    def __iter__(self) -> Iterator[Dict[str, Union[np.ndarray, List[str]]]]:
        """Create an iterator that yields batches."""
        while True:
            yield self.get_batch(self.batch_size)
    
    def __len__(self) -> int:
        """Return approximate number of batches."""
        total_rows = sum(m['row_count'] for m in self.file_metadata)
        return total_rows // self.batch_size


def format_file_path(file_path):
    """
    Format file path to show parent directory in orange and basename.
    Example: /path/to/11p4/454.pkl -> 11p4 - 454.pkl (with 11p4 in orange)
    """
    basename = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    # Format with orange color for parent directory
    formatted = f"{Colors.ORANGE}{parent_dir}{Colors.RESET} - {basename}"
    return formatted


# Example usage:
if __name__ == "__main__":
    # Define the root directory containing .pkl files
    root_dir = "/Users/kkreth/PycharmProjects/data/all_data_broken_down_1200_each_directory_PARTIAL"
    
    # Create the dataloader
    start_time = time.time()
    dataloader = EfficientDataLoader(
        root_directory=root_dir,
        batch_size=64,    # Default batch size
        num_workers=4,    # Parallel workers for file operations
        cache_size=10,    # Number of files to keep in memory
        shuffle=True      # Enable shuffling of batches
    )
    initialization_time = time.time() - start_time
    
    print(f"Dataloader initialization took {initialization_time:.3f} seconds")
    print(f"Found {len(dataloader.all_files)} .pkl files in the directory")
    print(f"Found {len(dataloader.file_metadata)} valid files with complete velocity data")
    
    # Print compression statistics
    gzipped_files = sum(1 for m in dataloader.file_metadata if m.get('is_gzipped', False))
    print(f"Files compression: {gzipped_files} gzipped, {len(dataloader.file_metadata) - gzipped_files} uncompressed")
    
    # Test 1: Sample a specific number of rows with timing
    print("\nTest 1: Sampling specific number of rows (10 samples)")
    NUMBER_OF_ROWS = 128
    
    total_time = 0
    for i in range(10):
        start_time = time.time()
        batch = dataloader.get_batch(NUMBER_OF_ROWS=NUMBER_OF_ROWS)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Access the velocity data
        velocity_data = batch['velocity_data']
        
        # Verify the data has exactly 375 columns
        if velocity_data.shape[1] != 375:
            print(f"ERROR: Expected 375 velocity values, but got {velocity_data.shape[1]}")
        
        # Count source files and show distribution
        source_files = batch['source_files']
        unique_files = set(source_files)
        file_counts = Counter(source_files)
        
        print(f"Sample {i+1}: {len(velocity_data)} rows from {len(unique_files)} files in {elapsed:.3f} seconds")
        print(f"  File distribution: {', '.join([f'{format_file_path(f)}: {count}' for f, count in file_counts.most_common(3)])}...")
    
    print(f"\nTest 1 Summary: Average time per batch: {total_time/10:.3f} seconds")
    
    # Show sample of data for last batch
    print("\nSample of velocity data from last batch:")
    for i in range(min(3, len(velocity_data))):
        formatted_path = format_file_path(batch['source_files'][i])
        print(f"Row {i} from {formatted_path}: {velocity_data[i, :5]} ... (showing first 5 values)")
    
    # Test 2: Use as an iterator with timing
    print("\nTest 2: Using as an iterator (10 batches)")
    total_time = 0
    batch_count = 0
    
    # Create iterator
    data_iterator = iter(dataloader)
    
    for _ in range(10):  # Get exactly 10 batches
        batch_count += 1
        
        # Start timer before getting batch
        start_time = time.time()
        
        # Get batch from iterator
        batch = next(data_iterator)
        
        # Perform some operation on the batch to simulate usage
        velocity_data = batch['velocity_data']
        mean_values = np.mean(velocity_data, axis=0)
        
        # Stop timer after processing
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Analyze source file distribution
        source_files = batch['source_files']
        unique_files = set(source_files)
        file_counts = Counter(source_files)
        
        print(f"Batch {batch_count}: {len(velocity_data)} rows from {len(unique_files)} files in {elapsed:.3f} seconds")
        print(f"  File distribution: {', '.join([f'{format_file_path(f)}: {count}' for f, count in file_counts.most_common(3)])}...")
    
    print(f"\nTest 2 Summary: Average time per batch: {total_time/batch_count:.3f} seconds")
    
    # Performance comparison
    print("\nCache statistics:")
    print(f"Cache size: {len(dataloader.file_cache)} files")
    print(f"Cache limit: {dataloader.cache_size} files")
    
    print("\nDataLoader performance test completed successfully!")