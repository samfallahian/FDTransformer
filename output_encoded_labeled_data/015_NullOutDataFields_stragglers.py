"""
Null Out Data Fields for Stragglers - Find and reprocess zero-byte files

This script identifies .pkl.gz files with size 0 that were created by 010_NullOutDataFields.py
but failed to write properly. It then reprocesses them from their source files.

The script:
1. Scans all directories for zero-byte .pkl.gz files
2. Identifies the corresponding source files (*_with_latent.pkl)
3. Reprocesses them with DEBUG logging to diagnose failures
4. Reports detailed statistics on what was found and fixed
"""

import pickle
import gzip
import os
import sys
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'null_stragglers_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class NullOutStragglers:
    """Find and reprocess zero-byte files created by 010_NullOutDataFields.py"""

    def __init__(self, data_root: str):
        """
        Initialize the processor

        Args:
            data_root: Root directory containing subdirectories with pkl files
        """
        self.data_root = Path(data_root)
        self.latent_columns = [f'latent_{i}' for i in range(1, 48)]

    def find_zero_byte_files(self) -> List[Dict]:
        """
        Scan all directories for zero-byte .pkl.gz files

        Returns:
            List of dictionaries with file information
        """
        logger.info("🔍 Scanning for zero-byte .pkl.gz files...")

        zero_byte_files = []

        # Get all subdirectories
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Found {len(subdirs)} directories to scan")

        for subdir in sorted(subdirs):
            logger.debug(f"\n📁 Scanning directory: {subdir.name}")

            # Find all .pkl.gz files
            pkl_gz_files = sorted(subdir.glob("*.pkl.gz"))
            logger.debug(f"  Found {len(pkl_gz_files)} .pkl.gz files")

            for pkl_gz_file in pkl_gz_files:
                file_size = pkl_gz_file.stat().st_size

                if file_size == 0:
                    logger.warning(f"  ⚠️  ZERO BYTE: {pkl_gz_file.name}")

                    # Extract the file number from the name (e.g., "1034.pkl.gz" -> "1034")
                    file_number = pkl_gz_file.stem.replace('.pkl', '')

                    # Look for the source file
                    source_file = subdir / f"{file_number}_with_latent.pkl"

                    file_info = {
                        'zero_byte_file': pkl_gz_file,
                        'source_file': source_file,
                        'source_exists': source_file.exists(),
                        'directory': subdir.name,
                        'file_number': file_number
                    }

                    if source_file.exists():
                        source_size = source_file.stat().st_size
                        file_info['source_size'] = source_size
                        logger.info(f"    ✓ Source file exists: {source_file.name} ({source_size:,} bytes)")
                    else:
                        file_info['source_size'] = 0
                        logger.error(f"    ✗ Source file NOT found: {source_file.name}")

                    zero_byte_files.append(file_info)

        logger.info(f"\n✅ Scan complete")
        logger.info(f"🔴 Found {len(zero_byte_files)} zero-byte files")

        # Count how many have source files
        with_source = sum(1 for f in zero_byte_files if f['source_exists'])
        logger.info(f"✓ {with_source} have source files available")
        logger.info(f"✗ {len(zero_byte_files) - with_source} missing source files")

        return zero_byte_files

    def load_pickle(self, file_path: Path) -> Tuple[pd.DataFrame, str]:
        """
        Load a pickle file (handles gzip compression)

        Returns:
            Tuple of (DataFrame, error_message). Error_message is empty string if successful.
        """
        logger.debug(f"Loading: {file_path}")
        logger.debug(f"  File exists: {file_path.exists()}")
        logger.debug(f"  File size: {file_path.stat().st_size if file_path.exists() else 'N/A'} bytes")

        try:
            # Try gzip first
            with gzip.open(file_path, 'rb') as f:
                df = pickle.load(f)
                logger.debug(f"  Loaded as gzip: {len(df)} rows, {len(df.columns)} columns")
                return df, ""
        except (OSError, gzip.BadGzipFile) as e:
            logger.debug(f"  Not gzip, trying uncompressed: {e}")
            try:
                # Try uncompressed
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.debug(f"  Loaded uncompressed: {len(df)} rows, {len(df.columns)} columns")
                    return df, ""
            except Exception as e2:
                error_msg = f"Failed to load: gzip={e}, uncompressed={e2}"
                logger.error(f"  {error_msg}")
                return None, error_msg

    def null_latent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set all latent columns to 0

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with latent columns set to 0
        """
        logger.debug("Nulling latent columns...")
        df_copy = df.copy()

        # Find which latent columns exist
        existing_latent_cols = [col for col in self.latent_columns if col in df_copy.columns]
        logger.debug(f"  Found {len(existing_latent_cols)} latent columns in DataFrame")

        if existing_latent_cols:
            # Check before nulling
            non_zero_before = sum(1 for col in existing_latent_cols if not (df_copy[col] == 0).all())
            logger.debug(f"  Non-zero columns before: {non_zero_before}")

            # Null them out
            df_copy[existing_latent_cols] = 0

            # Verify after nulling
            non_zero_after = sum(1 for col in existing_latent_cols if not (df_copy[col] == 0).all())
            logger.debug(f"  Non-zero columns after: {non_zero_after}")

            if non_zero_after == 0:
                logger.debug(f"  ✓ Successfully nulled {len(existing_latent_cols)} columns")
            else:
                logger.warning(f"  ⚠️  Still have {non_zero_after} non-zero columns!")

        return df_copy

    def validate_nulled_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all latent columns are set to 0

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
        logger.debug(f"Saving to: {output_path}")
        logger.debug(f"  DataFrame: {len(df)} rows, {len(df.columns)} columns")

        try:
            with gzip.open(output_path, 'wb', compresslevel=9) as f:
                pickle.dump(df, f)

            file_size = output_path.stat().st_size
            logger.debug(f"  ✓ Saved successfully: {file_size:,} bytes")

            if file_size == 0:
                logger.error(f"  ❌ WARNING: Output file is still 0 bytes!")
                return False

            return True

        except Exception as e:
            logger.error(f"  ❌ Failed to save: {e}")
            return False

    def process_straggler(self, file_info: Dict) -> bool:
        """
        Reprocess a single zero-byte file from its source

        Args:
            file_info: Dictionary with file information

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Processing: {file_info['directory']}/{file_info['file_number']}")
        logger.info(f"   Zero-byte file: {file_info['zero_byte_file'].name}")
        logger.info(f"   Source file: {file_info['source_file'].name}")
        logger.info(f"   Source exists: {file_info['source_exists']}")
        if file_info['source_exists']:
            logger.info(f"   Source size: {file_info['source_size']:,} bytes")
        logger.info(f"{'='*60}")

        if not file_info['source_exists']:
            logger.error("❌ Cannot process: source file does not exist")
            return False

        try:
            # Step 1: Load source file
            logger.debug("Step 1: Loading source file...")
            df, error = self.load_pickle(file_info['source_file'])

            if df is None:
                logger.error(f"❌ Cannot load source file: {error}")
                return False

            logger.info(f"✓ Source loaded: {len(df)} rows, {len(df.columns)} columns")

            # Step 2: Check what latent columns exist
            logger.debug("Step 2: Checking latent columns...")
            existing_latent = [col for col in self.latent_columns if col in df.columns]
            logger.info(f"✓ Found {len(existing_latent)} latent columns")

            # Check if any have data
            non_zero_latent = [col for col in existing_latent if not (df[col] == 0).all()]
            if non_zero_latent:
                logger.info(f"  {len(non_zero_latent)} columns have non-zero data")
            else:
                logger.info(f"  All latent columns already zero")

            # Step 3: Null out latent columns
            logger.debug("Step 3: Nulling latent columns...")
            df = self.null_latent_columns(df)
            logger.info(f"✓ Latent columns nulled")

            # Step 4: Validate
            logger.debug("Step 4: Validating...")
            is_valid, non_zero_cols = self.validate_nulled_columns(df)

            if not is_valid:
                logger.error(f"❌ Validation failed: {len(non_zero_cols)} columns still non-zero: {non_zero_cols}")
                return False

            logger.info(f"✓ Validation passed")

            # Step 5: Save
            logger.debug("Step 5: Saving compressed file...")
            success = self.save_pickle_compressed(df, file_info['zero_byte_file'])

            if not success:
                logger.error(f"❌ Failed to save file")
                return False

            # Final verification
            final_size = file_info['zero_byte_file'].stat().st_size
            logger.info(f"✓ File saved: {final_size:,} bytes")

            if final_size == 0:
                logger.error(f"❌ CRITICAL: File is still 0 bytes after save!")
                return False

            logger.info(f"✅ SUCCESS: {file_info['file_number']} processed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ FAILED: {file_info['file_number']}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def process_all_stragglers(self, zero_byte_files: List[Dict]):
        """
        Process all zero-byte files

        Args:
            zero_byte_files: List of file info dictionaries
        """
        if not zero_byte_files:
            logger.info("✅ No zero-byte files found - all files are healthy!")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting to process {len(zero_byte_files)} zero-byte files")
        logger.info(f"{'='*60}")

        success_count = 0
        fail_count = 0
        skipped_count = 0

        for idx, file_info in enumerate(zero_byte_files, 1):
            logger.info(f"\n[{idx}/{len(zero_byte_files)}]")

            if not file_info['source_exists']:
                logger.warning(f"⚠️  Skipping {file_info['file_number']}: no source file")
                skipped_count += 1
                continue

            if self.process_straggler(file_info):
                success_count += 1
            else:
                fail_count += 1

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total zero-byte files: {len(zero_byte_files)}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Skipped (no source): {skipped_count}")
        logger.info(f"{'='*60}")

        # List files by directory
        if zero_byte_files:
            logger.info(f"\n📁 Files by directory:")
            by_dir = {}
            for f in zero_byte_files:
                dir_name = f['directory']
                if dir_name not in by_dir:
                    by_dir[dir_name] = []
                by_dir[dir_name].append(f['file_number'])

            for dir_name in sorted(by_dir.keys()):
                logger.info(f"  {dir_name}: {', '.join(sorted(by_dir[dir_name]))}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Find and reprocess zero-byte .pkl.gz files with DEBUG logging'
    )
    parser.add_argument('--data-root', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate',
                       help='Root directory containing subdirectories with pkl files')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan for zero-byte files without processing them')

    args = parser.parse_args()

    # Check if data root exists
    if not os.path.exists(args.data_root):
        logger.error(f"Data root not found: {args.data_root}")
        sys.exit(1)

    processor = NullOutStragglers(args.data_root)

    # Scan for zero-byte files
    zero_byte_files = processor.find_zero_byte_files()

    if args.scan_only:
        logger.info("\n--scan-only mode: exiting without processing")

        # Print detailed report
        if zero_byte_files:
            print("\n" + "="*60)
            print("ZERO-BYTE FILES REPORT")
            print("="*60)

            for f in zero_byte_files:
                print(f"\nFile: {f['directory']}/{f['file_number']}")
                print(f"  Zero-byte: {f['zero_byte_file']}")
                print(f"  Source: {f['source_file']}")
                print(f"  Source exists: {f['source_exists']}")
                if f['source_exists']:
                    print(f"  Source size: {f['source_size']:,} bytes")

        sys.exit(0)

    # Process zero-byte files
    processor.process_all_stragglers(zero_byte_files)


if __name__ == '__main__':
    main()
