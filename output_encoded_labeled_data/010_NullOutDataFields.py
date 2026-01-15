"""
Null Out Data Fields - Remove latent columns and repackage with proper compression

This script walks through all pickle files in the data directory and:
1. Removes all data in columns latent_1 through latent_47
2. Validates that all requested columns are set to 0
3. Re-packages the file with gzip compression and new naming convention (N.pkl.gz)
"""

import pickle
import gzip
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


class NullOutDataFields:
    """Process pickle files to null out latent data fields"""

    def __init__(self, data_root: str, n_threads: int = 10):
        """
        Initialize the processor

        Args:
            data_root: Root directory containing subdirectories with pkl files
            n_threads: Number of threads for parallel processing (default: 10)
        """
        self.data_root = Path(data_root)
        self.latent_columns = self._generate_latent_columns()
        self.n_threads = n_threads
        self.print_lock = Lock()

    def _generate_latent_columns(self) -> List[str]:
        """Generate list of latent column names (latent_1 through latent_47)"""
        return [f'latent_{i}' for i in range(1, 48)]  # 1 to 47 inclusive

    def load_pickle(self, file_path: Path) -> pd.DataFrame:
        """Load a pickle file (handles gzip compression)"""
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except (OSError, gzip.BadGzipFile):
            # Try without gzip if not compressed
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    def null_latent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set all latent columns to 0

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with latent columns set to 0
        """
        df_copy = df.copy()

        # Find which latent columns exist in the dataframe
        existing_latent_cols = [col for col in self.latent_columns if col in df_copy.columns]

        if existing_latent_cols:
            df_copy[existing_latent_cols] = 0

        return df_copy

    def validate_nulled_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all latent columns are set to 0

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_non_zero_columns)
        """
        existing_latent_cols = [col for col in self.latent_columns if col in df.columns]
        non_zero_cols = []

        for col in existing_latent_cols:
            if not (df[col] == 0).all():
                non_zero_cols.append(col)

        is_valid = len(non_zero_cols) == 0
        return is_valid, non_zero_cols

    def save_pickle_compressed(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame as gzip-compressed pickle"""
        with gzip.open(output_path, 'wb', compresslevel=9) as f:
            pickle.dump(df, f)

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a single file: load, null latents, validate, save

        Args:
            input_path: Path to input pickle file
            output_path: Path to output compressed pickle file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the file
            df = self.load_pickle(input_path)

            # Null out latent columns
            df = self.null_latent_columns(df)

            # Validate
            is_valid, non_zero_cols = self.validate_nulled_columns(df)

            if not is_valid:
                with self.print_lock:
                    print(f"❌ Validation failed for {input_path.name}: {non_zero_cols}")
                return False

            # Save with new naming convention
            self.save_pickle_compressed(df, output_path)

            return True

        except Exception as e:
            with self.print_lock:
                print(f"❌ Error processing {input_path.name}: {str(e)}")
            return False

    def _process_file_worker(self, args):
        """
        Worker function for parallel processing

        Args:
            args: Tuple of (pkl_file, output_file, dry_run)

        Returns:
            Tuple of (status, pkl_file) where status is 'processed', 'failed', or 'skipped'
        """
        pkl_file, output_file, dry_run = args

        if dry_run:
            with self.print_lock:
                print(f"  Would process: {pkl_file.name} -> {output_file.name}")
            return ('processed', pkl_file)

        # Skip if output already exists
        if output_file.exists():
            return ('skipped', pkl_file)

        # Process the file
        success = self.process_file(pkl_file, output_file)

        if success:
            return ('processed', pkl_file)
        else:
            return ('failed', pkl_file)

    def process_directory(self, subdir_name: str, dry_run: bool = False) -> dict:
        """
        Process all pickle files in a subdirectory using multithreading

        Args:
            subdir_name: Name of subdirectory (e.g., '3p6', '4p4')
            dry_run: If True, only report what would be done without processing

        Returns:
            Dictionary with statistics
        """
        input_dir = self.data_root / subdir_name
        output_dir = self.data_root / subdir_name

        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return {'error': 'Directory not found'}

        # Find all .pkl files
        pkl_files = sorted(input_dir.glob("*.pkl"))

        stats = {
            'total': len(pkl_files),
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

        print(f"\n📁 Processing {subdir_name}: {stats['total']} files with {self.n_threads} threads")

        # Prepare work items
        work_items = []
        for pkl_file in pkl_files:
            # Extract number from filename (e.g., "1000_with_latent.pkl" -> "1000")
            file_number = pkl_file.stem.split('_')[0]
            output_file = output_dir / f"{file_number}.pkl.gz"
            work_items.append((pkl_file, output_file, dry_run))

        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {executor.submit(self._process_file_worker, item): item for item in work_items}

            for future in as_completed(futures):
                status, pkl_file = future.result()

                if status == 'processed':
                    stats['processed'] += 1
                    if stats['processed'] % 100 == 0:
                        with self.print_lock:
                            print(f"  ✓ Processed {stats['processed']}/{stats['total']} files")
                elif status == 'failed':
                    stats['failed'] += 1
                elif status == 'skipped':
                    stats['skipped'] += 1

        print(f"  ✅ Complete: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")

        return stats

    def process_all_directories(self, dry_run: bool = False) -> dict:
        """
        Process all subdirectories in the data root

        Args:
            dry_run: If True, only report what would be done

        Returns:
            Dictionary mapping subdirectory names to their statistics
        """
        all_stats = {}

        # Get all subdirectories
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]

        print(f"🚀 Found {len(subdirs)} directories to process")

        for subdir in sorted(subdirs):
            stats = self.process_directory(subdir.name, dry_run=dry_run)
            all_stats[subdir.name] = stats

        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        total_processed = sum(s.get('processed', 0) for s in all_stats.values())
        total_failed = sum(s.get('failed', 0) for s in all_stats.values())
        total_skipped = sum(s.get('skipped', 0) for s in all_stats.values())
        print(f"Total processed: {total_processed}")
        print(f"Total skipped: {total_skipped}")
        print(f"Total failed: {total_failed}")

        return all_stats


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Null out latent data fields and repackage')
    parser.add_argument('--data-root', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate',
                       help='Root directory containing subdirectories with pkl files')
    parser.add_argument('--subdir', type=str, default=None,
                       help='Process only this subdirectory (e.g., "3p6")')
    parser.add_argument('--threads', type=int, default=10,
                       help='Number of threads for parallel processing (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Report what would be done without actually processing')

    args = parser.parse_args()

    processor = NullOutDataFields(args.data_root, n_threads=args.threads)

    if args.subdir:
        # Process single directory
        processor.process_directory(args.subdir, dry_run=args.dry_run)
    else:
        # Process all directories
        processor.process_all_directories(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
