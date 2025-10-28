#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds two large non-IO-bound datasets for rapid auto-encoder training and validation.

It samples rows uniformly at random across the existing training files using
`EfficientDataLoader`, then writes two pickle files to the configured `root_path`:

- training_auto_encoder.pkl
- validation_auto_encoder.pkl

Defaults to 1,000,000 rows for each split. Paths are sourced from `experiment.preferences`.
Data source is `training_data_path`; outputs are written under `root_path`.
"""

import os
import sys
import argparse
import logging
import time
import pickle
import numpy as np

# Resolve project root to import local modules
# The script is in cgan/, so parent is the project root containing cgan/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # This script is already in the cgan directory

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Ordered_001_Initialize import HostPreferences  # noqa: E402
from EfficientDataLoader import EfficientDataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoEncoderDatasetBuilder:
    """
    Build fixed training/validation datasets for rapid, non-IO-bound AE training.
    """
    def __init__(self,
                 preferences_path: str = None,
                 train_rows: int = 1_000_000,
                 val_rows: int = 1_000_000,
                 seed: int | None = 42,
                 num_workers: int = 10,
                 cache_size: int = 50,
                 row_floor: int = 20,
                 enable_manifest_cache: bool = True,
                 enable_profiling: bool = False):
        self.project_root = PROJECT_ROOT
        
        # Resolve preferences path: use provided path, or look in script's directory
        if preferences_path is None:
            self.preferences_path = os.path.join(SCRIPT_DIR, "experiment.preferences")
        else:
            self.preferences_path = preferences_path
            
        self.train_rows = int(train_rows)
        self.val_rows = int(val_rows)
        self.seed = seed
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.row_floor = row_floor
        self.enable_manifest_cache = enable_manifest_cache
        self.enable_profiling = enable_profiling

        self.preferences = HostPreferences(filename=self.preferences_path)
        # Source directory for sampling
        self.source_root = getattr(self.preferences, 'training_data_path', None) or self.preferences.root_path
        # Destination directory for the new datasets
        self.dest_root = self.preferences.root_path

        # Adopt logging level from preferences when available
        lvl = getattr(logging, str(self.preferences.logging_level).upper(), None) if hasattr(self.preferences, 'logging_level') else None
        if isinstance(lvl, int):
            logger.setLevel(lvl)

        logger.info(f"AutoEncoderDatasetBuilder initialized: source_root={self.source_root} dest_root={self.dest_root}")

    def _build_loader(self) -> EfficientDataLoader:
        t0 = time.perf_counter()
        loader = EfficientDataLoader(
            root_directory=self.source_root,
            batch_size=128,
            num_workers=self.num_workers,
            cache_size=self.cache_size,
            shuffle=True,
            seed=self.seed,
            enable_manifest_cache=self.enable_manifest_cache,
            enable_profiling=self.enable_profiling
        )
        t1 = time.perf_counter()
        logger.info(f"EfficientDataLoader ready in {t1 - t0:.2f}s (workers={self.num_workers}, cache_size={self.cache_size})")
        return loader

    def _sample_rows(self, loader: EfficientDataLoader, n_rows: int) -> np.ndarray:
        logger.info(f"Sampling {n_rows:,} rows...")
        t0 = time.perf_counter()
        batch = loader.get_batch(NUMBER_OF_ROWS=n_rows, ROW_FLOOR=self.row_floor)
        data = batch['velocity_data']
        # Ensure a compact dtype for storage and training
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        t1 = time.perf_counter()
        logger.info(f"Collected {len(data):,} rows in {t1 - t0:.2f}s; shape={data.shape}, dtype={data.dtype}")
        return data

    def _ensure_dest(self):
        os.makedirs(self.dest_root, exist_ok=True)

    def _save_pickle(self, arr: np.ndarray, filename: str):
        path = os.path.join(self.dest_root, filename)
        t0 = time.perf_counter()
        with open(path, 'wb') as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        t1 = time.perf_counter()
        size_gb = os.path.getsize(path) / (1024**3)
        logger.info(f"Wrote {filename} to {self.dest_root} in {t1 - t0:.2f}s (size={size_gb:.2f} GiB)")

    def build(self,
              training_filename: str = "training_auto_encoder.pkl",
              validation_filename: str = "validation_auto_encoder.pkl"):
        self._ensure_dest()

        loader = self._build_loader()

        # Train split
        train_data = self._sample_rows(loader, self.train_rows)
        self._save_pickle(train_data, training_filename)
        # Free memory before sampling validation
        del train_data

        # Validation split (draws another random set)
        val_data = self._sample_rows(loader, self.val_rows)
        self._save_pickle(val_data, validation_filename)
        del val_data

        logger.info("Dataset build complete.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create non-IO-bound train/validation datasets for AE training.")
    parser.add_argument('--preferences', type=str, default=None,
                        help='Path to experiment.preferences (defaults to script directory).')
    parser.add_argument('--train_rows', type=int, default=1_000_000,
                        help='Number of rows for training split.')
    parser.add_argument('--val_rows', type=int, default=1_000_000,
                        help='Number of rows for validation split.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (pass to EfficientDataLoader).')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers for loader sampling.')
    parser.add_argument('--cache_size', type=int, default=50,
                        help='Max number of files to keep in loader cache.')
    parser.add_argument('--row_floor', type=int, default=20,
                        help='Minimum rows to draw from any single file during sampling.')
    parser.add_argument('--no_manifest_cache', action='store_true',
                        help='Disable manifest on-disk cache for loader.')

    args = parser.parse_args(argv)

    builder = AutoEncoderDatasetBuilder(
        preferences_path=args.preferences,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        seed=args.seed,
        num_workers=args.workers,
        cache_size=args.cache_size,
        row_floor=args.row_floor,
        enable_manifest_cache=not args.no_manifest_cache,
        enable_profiling=False,
    )
    builder.build()


if __name__ == '__main__':
    main()
