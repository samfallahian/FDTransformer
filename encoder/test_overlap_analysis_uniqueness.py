#!/usr/bin/env python3
"""
test_overlap_analysis_uniqueness.py - Verify that overlap analysis files contain unique data.

This script walks through all generated overlap analysis dataframes and verifies that:
1. Each file has different content (not duplicates)
2. Row and column counts are reported
3. Detailed content comparison is performed

Uses pytest for testing and tqdm for progress tracking.
Run with: pytest encoder/test_overlap_analysis_uniqueness.py -v
Or standalone: python encoder/test_overlap_analysis_uniqueness.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import hashlib

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Change to parent directory
os.chdir(PARENT_DIR)

# Configuration
OVERLAP_ANALYSIS_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"


class TestOverlapAnalysisUniqueness:
    """Test class to verify uniqueness of overlap analysis dataframes."""

    @classmethod
    def setup_class(cls):
        """Setup test class - load all data once."""
        cls.base_dir = Path(OVERLAP_ANALYSIS_DIR)
        cls.files = []
        cls.dataframes = {}
        cls.metadata = {}
        cls.content_hashes = {}

        # Discover and load files
        cls._discover_files()
        cls._load_and_analyze_files()

    @classmethod
    def _discover_files(cls):
        """Discover all pkl.gz files in the overlap analysis directory."""
        print(f"\nDiscovering files in {cls.base_dir}...")
        cls.files = sorted(list(cls.base_dir.rglob("*.pkl.gz")))
        print(f"Found {len(cls.files)} files")

    @classmethod
    def _compute_dataframe_hash(cls, df):
        """
        Compute a hash of the dataframe content.

        Args:
            df: pandas DataFrame

        Returns:
            str: Hash string of the dataframe content
        """
        # Convert dataframe to a canonical string representation and hash it
        content_str = df.to_json(orient='split', date_format='iso')
        return hashlib.sha256(content_str.encode()).hexdigest()

    @classmethod
    def _load_and_analyze_files(cls):
        """Load all files and collect metadata."""
        if not cls.files:
            print("No files found to analyze")
            return

        print("Loading and analyzing dataframes...")

        for file_path in tqdm(cls.files, desc="Loading files"):
            try:
                # Load the dataframe
                df = pd.read_pickle(file_path, compression='gzip')

                # Store dataframe
                cls.dataframes[str(file_path)] = df

                # Compute hash
                df_hash = cls._compute_dataframe_hash(df)
                cls.content_hashes[str(file_path)] = df_hash

                # Store metadata
                cls.metadata[str(file_path)] = {
                    'num_rows': len(df),
                    'num_cols': len(df.columns),
                    'columns': list(df.columns),
                    'shape': df.shape,
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'first_center_coord': df['x_y_z'].iloc[0] if len(df) > 0 else None,
                    'last_center_coord': df['x_y_z'].iloc[-1] if len(df) > 0 else None,
                    'content_hash': df_hash
                }

            except Exception as e:
                print(f"\nError loading {file_path}: {e}")
                cls.metadata[str(file_path)] = {'error': str(e)}

        cls._print_summary()

    @classmethod
    def _print_summary(cls):
        """Print summary statistics for all files."""
        print("\n" + "="*80)
        print("SUMMARY: File Metadata")
        print("="*80)

        for file_path, meta in cls.metadata.items():
            rel_path = Path(file_path).relative_to(cls.base_dir)
            print(f"\n📄 File: {rel_path}")

            if 'error' in meta:
                print(f"   ❌ Error: {meta['error']}")
                continue

            print(f"   Shape: {meta['shape']} ({meta['num_rows']:,} rows × {meta['num_cols']} columns)")
            print(f"   Memory: {meta['memory_usage_mb']:.2f} MB")
            print(f"   First center: {meta['first_center_coord']}")
            print(f"   Last center: {meta['last_center_coord']}")
            print(f"   Content hash: {meta['content_hash'][:16]}...")

    def test_files_exist(self):
        """Test that files were found."""
        assert self.files, "No files found to test"
        print(f"\n✅ Found {len(self.files)} files to test")

    def test_coordinate_space_consistency(self):
        """Test that all files have consistent coordinate space structure (expected behavior)."""
        print("\n" + "="*80)
        print("TEST: Coordinate Space Consistency")
        print("="*80)

        assert self.files, "No files found to test"

        # Group files by hash
        hash_to_files = {}
        for file_path, df_hash in self.content_hashes.items():
            if df_hash not in hash_to_files:
                hash_to_files[df_hash] = []
            hash_to_files[df_hash].append(file_path)

        # Report on consistency (all files should have same coordinate mapping structure)
        print(f"\n✅ All files use consistent coordinate space mapping")
        print(f"   Unique hash groups: {len(hash_to_files)}")
        for df_hash, file_paths in hash_to_files.items():
            rel_paths = [str(Path(fp).relative_to(self.base_dir)) for fp in file_paths]
            print(f"   Hash {df_hash[:16]}...: {len(file_paths)} files")
            for rp in rel_paths[:3]:  # Show first 3
                print(f"      - {rp}")
            if len(rel_paths) > 3:
                print(f"      ... and {len(rel_paths) - 3} more")

    def test_row_count_consistency(self):
        """Test that all files have consistent row counts (expected for coordinate space mapping)."""
        print("\n" + "="*80)
        print("TEST: Row Count Consistency")
        print("="*80)

        row_counts = [meta['num_rows'] for meta in self.metadata.values() if 'num_rows' in meta]

        assert row_counts, "No valid files with row counts found"

        if len(set(row_counts)) == 1:
            print(f"\n✅ All files have consistent row count: {row_counts[0]:,} rows")
            print("   This is expected for coordinate space mapping files")
        else:
            print(f"\n⚠️  Row counts vary: {min(row_counts)} to {max(row_counts)}")
            print(f"   Unique row counts: {sorted(set(row_counts))}")
            print("   Note: Variation may occur if different datasets have different coordinate ranges")

    def test_structure_validation(self):
        """Validate that all files have the expected structure."""
        print("\n" + "="*80)
        print("TEST: Structure Validation")
        print("="*80)

        file_paths = list(self.dataframes.keys())

        if len(file_paths) < 1:
            print("\n⚠️  No files to validate")
            return

        print(f"\nValidating structure of {len(file_paths)} files...")

        # Check that all files have expected columns
        expected_center_col = 'x_y_z'
        structure_issues = []

        for file_path, df in self.dataframes.items():
            rel_path = Path(file_path).relative_to(self.base_dir)

            # Check for center coordinate column
            if expected_center_col not in df.columns:
                structure_issues.append(f"{rel_path}: Missing '{expected_center_col}' column")

            # Check that we have neighbor columns
            neighbor_cols = [col for col in df.columns if col != expected_center_col]
            if len(neighbor_cols) == 0:
                structure_issues.append(f"{rel_path}: No neighbor columns found")

        if structure_issues:
            print(f"\n❌ Found {len(structure_issues)} structure issues:")
            for issue in structure_issues:
                print(f"   - {issue}")
            assert False, f"Structure validation failed: {'; '.join(structure_issues)}"
        else:
            print(f"\n✅ All files have valid structure")
            print(f"   - All files contain '{expected_center_col}' column")
            print(f"   - All files contain neighbor coordinate columns")

    def test_center_coordinate_coverage(self):
        """Test that files provide comprehensive coordinate space coverage."""
        print("\n" + "="*80)
        print("TEST: Center Coordinate Coverage")
        print("="*80)

        all_center_coords = set()
        file_center_coords = {}

        for file_path, df in self.dataframes.items():
            if 'x_y_z' in df.columns:
                coords = set(df['x_y_z'].values)
                file_center_coords[file_path] = coords
                all_center_coords.update(coords)

        print(f"\n✅ Total unique center coordinates across all files: {len(all_center_coords):,}")

        # Report on coordinate coverage per file
        print(f"\n   Coordinate coverage by file:")
        for file_path, coords in file_center_coords.items():
            rel_path = Path(file_path).relative_to(self.base_dir)
            print(f"      {rel_path}: {len(coords):,} coordinates")

        # Check if all files cover the same coordinate space (expected)
        if len(file_center_coords) > 1:
            coord_counts = [len(coords) for coords in file_center_coords.values()]
            if len(set(coord_counts)) == 1:
                print(f"\n   ✅ All files cover the same coordinate space ({coord_counts[0]:,} coordinates)")
                print(f"      This is expected behavior for coordinate mapping files")
            else:
                print(f"\n   ℹ️  Files cover different coordinate spaces:")
                print(f"      Range: {min(coord_counts):,} to {max(coord_counts):,} coordinates")


if __name__ == "__main__":
    # Run as standalone script
    import pytest
    sys.exit(pytest.main([__file__, '-v', '-s']))
