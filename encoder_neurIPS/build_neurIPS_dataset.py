import os
import sys
import argparse
import logging
import time
import pickle
import numpy as np
import random

# parent is the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from EfficientDataLoader import EfficientDataLoader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_excluded(path, excluded_terms):
    for term in excluded_terms:
        if term in path:
            return True
    return False

def build_neurips_dataset(
    source_root="/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data",
    dest_root="/Users/kkreth/PycharmProjects/data/encoder_neurIPS",
    train_rows=1_000_000,
    val_rows=1_000_000,
    excluded_from_train=["4p4", "6p4"], # 4p4 corresponds to 6.9, 6p4 corresponds to 10.0
    seed=42
):
    os.makedirs(dest_root, exist_ok=True)
    
    # We want to sample 1MM for train (excluding U6.9, U10)
    # and 1MM for val (specifically from U6.9, U10)
    
    # 1. Setup loader for Training (Excluding U6.9 and U10)
    logger.info("Setting up loader for Training data (Excluding U6.9, U10)...")
    train_loader = EfficientDataLoader(
        root_directory=source_root,
        batch_size=128,
        num_workers=10,
        cache_size=50,
        shuffle=True,
        seed=seed,
        enable_manifest_cache=False, # Disable cache to ensure we can filter
        show_progress=True,
        allowed_extensions=['.pkl.gz']
    )
    
    all_files = train_loader.all_files
    train_files = [f for f in all_files if not is_excluded(f, excluded_from_train)]
    val_source_files = [f for f in all_files if is_excluded(f, excluded_from_train)]
    
    logger.info(f"Total files: {len(all_files)}")
    logger.info(f"Training candidate files: {len(train_files)}")
    logger.info(f"Validation candidate files (Exclusions): {len(val_source_files)}")
    
    # Override loader's file list for training
    train_loader.all_files = train_files
    # Re-compute metadata for the filtered list
    new_metadata = []
    # Efficiency fix: build a map of path -> metadata
    # The key is 'file_path' in EfficientDataLoader metadata
    meta_map = {m['file_path']: m for m in train_loader.file_metadata}
    for f in train_files:
        if f in meta_map:
            new_metadata.append(meta_map[f])
    train_loader.file_metadata = new_metadata
    
    logger.info(f"Sampling {train_rows:,} training rows...")
    t0 = time.perf_counter()
    
    # Robust Sampling implementation
    def robust_sample(loader, n_rows, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        all_metadata = loader.file_metadata
        total_available_rows = sum(m['row_count'] for m in all_metadata)
        logger.info(f"Total available rows in candidate files: {total_available_rows:,}")
        
        # Select files proportionally to their row count
        weights = np.array([m['row_count'] for m in all_metadata], dtype=np.float64)
        weights /= weights.sum()
        
        # To avoid the ValueError, we'll use a slightly different approach if needed, 
        # but let's try a very explicit normalization first.
        weights = weights / weights.sum()
        
        # If still fails, we'll use a safer sampling method
        try:
            selected_indices = np.random.choice(len(all_metadata), size=min(1000, len(all_metadata)), replace=False, p=weights)
        except ValueError:
            logger.warning("Numpy choice failed, falling back to uniform file selection")
            selected_indices = np.random.choice(len(all_metadata), size=min(1000, len(all_metadata)), replace=False)
            
        sampled_data = []
        rows_per_file = n_rows // len(selected_indices) + 1
        
        from concurrent.futures import ThreadPoolExecutor
        def load_and_sample(idx):
            meta = all_metadata[idx]
            try:
                data, _ = loader._sample_rows_from_file(meta, min(meta['row_count'], rows_per_file))
                return data
            except:
                return None

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(load_and_sample, selected_indices))
        
        sampled_data = [r for r in results if r is not None]
        combined = np.vstack(sampled_data).astype(np.float32)
        if len(combined) > n_rows:
            combined = combined[:n_rows]
        return combined

    train_path = os.path.join(dest_root, "training_auto_encoder.pkl")
    if os.path.exists(train_path):
        logger.info(f"Training data already exists at {train_path}. Skipping training sampling.")
    else:
        logger.info(f"Sampling {train_rows:,} training rows...")
        t0 = time.perf_counter()
        train_data = robust_sample(train_loader, train_rows, seed)
        t1 = time.perf_counter()
        logger.info(f"Collected {len(train_data):,} training rows in {t1 - t0:.2f}s")
        
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved training data to {train_path}")
        del train_data
    
    # Cleanup
    try:
        del train_data
    except:
        pass
    del train_loader
    
    # 2. Setup loader for Validation (ONLY U6.9 and U10)
    logger.info("Setting up loader for Validation data (ONLY U6.9, U10)...")
    val_loader = EfficientDataLoader(
        root_directory=source_root,
        batch_size=128,
        num_workers=10,
        cache_size=50,
        shuffle=True,
        seed=seed + 1, # Different seed for validation
        enable_manifest_cache=False,
        show_progress=True,
        allowed_extensions=['.pkl.gz']
    )
    val_loader.all_files = val_source_files
    new_metadata_val = []
    meta_map_val = {m['file_path']: m for m in val_loader.file_metadata}
    for f in val_source_files:
        if f in meta_map_val:
            new_metadata_val.append(meta_map_val[f])
    val_loader.file_metadata = new_metadata_val
    
    logger.info(f"Sampling {val_rows:,} validation rows...")
    t0 = time.perf_counter()
    val_data = robust_sample(val_loader, val_rows, seed + 1)
    t1 = time.perf_counter()
    logger.info(f"Collected {len(val_data):,} validation rows in {t1 - t0:.2f}s")
    
    val_path = os.path.join(dest_root, "validation_auto_encoder.pkl")
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved validation data to {val_path}")
    
    # Verification of disjointness (though by construction they are from different files)
    logger.info("Dataset construction complete and disjoint.")

if __name__ == "__main__":
    build_neurips_dataset()
