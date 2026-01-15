"""
Populate Stragglers - Find and reprocess failed files with DEBUG logging

This script identifies files that failed during 020_populate_latent.py processing
and reprocesses them with DEBUG-level logging to diagnose the failure reason.

It scans all directories to find files that either:
1. Don't have populated latent fields (all zeros)
2. Are corrupted or unreadable
3. Have incomplete data

The script runs with DEBUG logging to provide detailed diagnostics for each failure.
"""

import pickle
import gzip
import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Add the root directory to the path for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the WAE model
from encoder.model_WAE_01 import WAE

# Configure logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'stragglers_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning about dtype incompatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class StragglerProcessor:
    """Find and reprocess failed files with detailed debugging"""

    def __init__(self, data_root: str, model_path: str):
        """
        Initialize the processor

        Args:
            data_root: Root directory containing subdirectories with pkl files
            model_path: Path to the WAE model file
        """
        self.data_root = Path(data_root)
        self.model_path = model_path

        # Load the model
        self.model, self.device = self._load_wae_model()
        logger.info(f"Model loaded on device: {self.device}")

    def _load_wae_model(self):
        """
        Load the WAE model from the specified path.

        Returns:
            Tuple of (model, device)
        """
        logger.info(f"Loading WAE model from: {self.model_path}")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else
                             "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                             "cpu")
        logger.info(f"Using device: {device}")

        # Initialize the model
        model = WAE().to(device)

        # Load the model weights with compatibility for PyTorch 2.6+ "weights_only" changes
        checkpoint = None
        torch_version = getattr(torch, "__version__", "unknown")
        logger.info(f"PyTorch version detected: {torch_version}")
        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning(f"torch.load with weights_only=False failed: {e}")
            try:
                from torch.serialization import add_safe_globals
                try:
                    from torch.torch_version import TorchVersion
                except Exception:
                    TorchVersion = None
                if 'add_safe_globals' in dir(__import__('torch').serialization) and TorchVersion is not None:
                    add_safe_globals([TorchVersion])
                try:
                    checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(self.model_path, map_location=device)
            except Exception as e2:
                logger.error(f"Failed to load checkpoint. Original: {e}; Fallback: {e2}")
                raise

        # Extract the model state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model, device

    def load_pickle(self, file_path: Path) -> Tuple[pd.DataFrame, str]:
        """
        Load a pickle file (handles gzip compression)

        Returns:
            Tuple of (DataFrame, error_message). Error_message is empty string if successful.
        """
        logger.debug(f"Attempting to load: {file_path}")
        logger.debug(f"  File exists: {file_path.exists()}")
        logger.debug(f"  File size: {file_path.stat().st_size if file_path.exists() else 'N/A'} bytes")

        try:
            with gzip.open(file_path, 'rb') as f:
                df = pickle.load(f)
                logger.debug(f"  Successfully loaded as gzip: {len(df)} rows, {len(df.columns)} columns")
                return df, ""
        except (OSError, gzip.BadGzipFile) as e:
            logger.debug(f"  Not a gzip file, trying uncompressed: {e}")
            try:
                with open(file_path, 'rb') as f:
                    df = pickle.load(f)
                    logger.debug(f"  Successfully loaded as uncompressed: {len(df)} rows, {len(df.columns)} columns")
                    return df, ""
            except Exception as e2:
                error_msg = f"Failed to load pickle: gzip error={e}, uncompressed error={e2}"
                logger.error(f"  {error_msg}")
                return None, error_msg

    def check_file_status(self, file_path: Path) -> Dict:
        """
        Check the status of a file to determine if it needs reprocessing

        Returns:
            Dictionary with status information
        """
        status = {
            'file': file_path.name,
            'path': str(file_path),
            'needs_processing': False,
            'reason': '',
            'details': {}
        }

        # Try to load the file
        df, error = self.load_pickle(file_path)

        if df is None:
            status['needs_processing'] = True
            status['reason'] = 'load_failed'
            status['details']['error'] = error
            return status

        status['details']['rows'] = len(df)
        status['details']['columns'] = len(df.columns)

        # Check if latent columns exist
        latent_cols = [f'latent_{i}' for i in range(1, 48)]
        missing_latent_cols = [col for col in latent_cols if col not in df.columns]

        if missing_latent_cols:
            status['needs_processing'] = True
            status['reason'] = 'missing_latent_columns'
            status['details']['missing_count'] = len(missing_latent_cols)
            logger.debug(f"  Missing {len(missing_latent_cols)} latent columns")
            return status

        # Check if all latent columns are zeros
        zero_latent_cols = []
        for col in latent_cols:
            if (df[col] == 0).all():
                zero_latent_cols.append(col)

        if len(zero_latent_cols) == 47:
            status['needs_processing'] = True
            status['reason'] = 'all_latent_zeros'
            logger.debug(f"  All latent columns are zeros")
            return status

        # Check for velocity columns
        velocity_cols = []
        for i in range(1, 126):
            velocity_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])

        missing_velocity_cols = [col for col in velocity_cols if col not in df.columns]
        if missing_velocity_cols:
            status['needs_processing'] = True
            status['reason'] = 'missing_velocity_columns'
            status['details']['missing_velocity_count'] = len(missing_velocity_cols)
            logger.debug(f"  Missing {len(missing_velocity_cols)} velocity columns")
            return status

        # Check for NaN values in velocity data
        velocity_data = df[velocity_cols]
        nan_count = velocity_data.isna().sum().sum()
        if nan_count > 0:
            status['needs_processing'] = True
            status['reason'] = 'nan_in_velocity_data'
            status['details']['nan_count'] = int(nan_count)
            logger.debug(f"  Found {nan_count} NaN values in velocity data")
            return status

        logger.debug(f"  File appears healthy")
        return status

    def scan_for_stragglers(self) -> List[Dict]:
        """
        Scan all directories for files that need reprocessing

        Returns:
            List of file status dictionaries for files that need processing
        """
        logger.info("🔍 Scanning for files that need reprocessing...")

        stragglers = []

        # Get all subdirectories
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        logger.info(f"Found {len(subdirs)} directories to scan")

        total_files = 0
        for subdir in sorted(subdirs):
            logger.info(f"\n📁 Scanning directory: {subdir.name}")

            # Find all pickle files
            pkl_files = sorted(subdir.glob("*.pkl.gz"))
            if not pkl_files:
                pkl_files = sorted(subdir.glob("*.pkl"))

            logger.info(f"  Found {len(pkl_files)} files")
            total_files += len(pkl_files)

            for pkl_file in pkl_files:
                status = self.check_file_status(pkl_file)

                if status['needs_processing']:
                    logger.warning(f"  ⚠️  NEEDS PROCESSING: {pkl_file.name} - {status['reason']}")
                    stragglers.append(status)

        logger.info(f"\n✅ Scan complete: {total_files} total files")
        logger.info(f"🔴 Found {len(stragglers)} files needing reprocessing")

        return stragglers

    def populate_latent_columns(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """
        Populate latent columns by encoding velocity data using batch processing.
        """
        logger.debug(f"Starting latent population for {file_name}")
        df_copy = df.copy()

        # Ensure latent columns exist with float32 dtype
        for i in range(1, 48):
            if f'latent_{i}' not in df_copy.columns:
                df_copy[f'latent_{i}'] = np.float32(0.0)
            else:
                df_copy[f'latent_{i}'] = df_copy[f'latent_{i}'].astype(np.float32)

        total_rows = len(df_copy)
        logger.debug(f"  Processing {total_rows} rows")

        # Extract all velocity columns at once
        velocity_cols = []
        for i in range(1, 126):
            velocity_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])

        # Check if all columns exist
        missing_cols = [col for col in velocity_cols if col not in df_copy.columns]
        if missing_cols:
            error_msg = f"Missing {len(missing_cols)} velocity columns: {missing_cols[:5]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extract all velocity data as a numpy array
        logger.debug("  Extracting velocity data...")
        velocity_data = df_copy[velocity_cols].to_numpy().astype(np.float32)
        logger.debug(f"  Velocity data shape: {velocity_data.shape}")
        logger.debug(f"  Velocity data stats: min={velocity_data.min():.4f}, max={velocity_data.max():.4f}, mean={velocity_data.mean():.4f}")

        # Process in batches
        batch_size = 512
        num_batches = (total_rows + batch_size - 1) // batch_size
        logger.debug(f"  Processing in {num_batches} batches of size {batch_size}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)

            logger.debug(f"  Batch {batch_idx+1}/{num_batches}: rows {start_idx}-{end_idx}")

            # Get batch of velocity data
            batch_velocities = velocity_data[start_idx:end_idx]

            # Convert to tensor and move to device
            velocity_tensor = torch.tensor(batch_velocities, dtype=torch.float32).to(self.device)
            logger.debug(f"    Tensor shape: {velocity_tensor.shape}, device: {velocity_tensor.device}")

            # Compute latent representations
            with torch.no_grad():
                latent_batch = self.model.encode(velocity_tensor)

            logger.debug(f"    Latent shape: {latent_batch.shape}")

            # Move to CPU and convert to numpy
            latent_np = latent_batch.cpu().numpy()
            logger.debug(f"    Latent stats: min={latent_np.min():.4f}, max={latent_np.max():.4f}, mean={latent_np.mean():.4f}")

            # Update the dataframe with batch results
            df_indices = df_copy.index[start_idx:end_idx]
            for i in range(1, 48):
                df_copy.loc[df_indices, f'latent_{i}'] = latent_np[:, i-1].astype(np.float32)

        logger.debug(f"  ✅ Completed encoding {total_rows} rows")
        return df_copy

    def save_pickle_compressed(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame as gzip-compressed pickle"""
        logger.debug(f"Saving to: {output_path}")
        logger.debug(f"  DataFrame: {len(df)} rows, {len(df.columns)} columns")

        with gzip.open(output_path, 'wb', compresslevel=9) as f:
            pickle.dump(df, f)

        file_size = output_path.stat().st_size
        logger.debug(f"  Saved successfully: {file_size:,} bytes")

    def process_straggler(self, straggler_info: Dict) -> bool:
        """
        Reprocess a single straggler file with detailed debugging

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(straggler_info['path'])
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 Processing straggler: {file_path.name}")
        logger.info(f"   Reason: {straggler_info['reason']}")
        logger.info(f"   Details: {straggler_info['details']}")
        logger.info(f"{'='*60}")

        try:
            # Load the file
            logger.debug("Step 1: Loading file...")
            df, error = self.load_pickle(file_path)

            if df is None:
                logger.error(f"❌ Cannot load file: {error}")
                return False

            logger.info(f"✓ File loaded: {len(df)} rows, {len(df.columns)} columns")

            # Populate latent columns
            logger.debug("Step 2: Populating latent columns...")
            df = self.populate_latent_columns(df, file_path.name)
            logger.info(f"✓ Latent columns populated")

            # Validate
            logger.debug("Step 3: Validating results...")
            latent_cols = [f'latent_{i}' for i in range(1, 48)]
            zero_count = sum(1 for col in latent_cols if (df[col] == 0).all())

            if zero_count > 0:
                logger.warning(f"⚠️  {zero_count} latent columns still all zeros")
            else:
                logger.info(f"✓ All latent columns populated successfully")

            # Save with compression
            logger.debug("Step 4: Saving file...")
            output_path = file_path.parent / f"{file_path.stem}.pkl.gz" if file_path.suffix != '.gz' else file_path
            self.save_pickle_compressed(df, output_path)
            logger.info(f"✓ File saved: {output_path.name}")

            logger.info(f"✅ SUCCESS: {file_path.name} processed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ FAILED: {file_path.name}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.exception("Full traceback:")
            return False

    def process_all_stragglers(self, stragglers: List[Dict]):
        """
        Process all straggler files

        Args:
            stragglers: List of straggler info dictionaries
        """
        if not stragglers:
            logger.info("✅ No stragglers found - all files are healthy!")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting to process {len(stragglers)} stragglers")
        logger.info(f"{'='*60}")

        success_count = 0
        fail_count = 0

        for idx, straggler in enumerate(stragglers, 1):
            logger.info(f"\n[{idx}/{len(stragglers)}]")

            if self.process_straggler(straggler):
                success_count += 1
            else:
                fail_count += 1

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total stragglers: {len(stragglers)}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"{'='*60}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Find and reprocess failed files with DEBUG logging')
    parser.add_argument('--data-root', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate',
                       help='Root directory containing subdirectories with pkl files')
    parser.add_argument('--model-path', type=str,
                       default='/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt',
                       help='Path to the WAE model file')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan for stragglers without processing them')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    processor = StragglerProcessor(args.data_root, args.model_path)

    # Scan for stragglers
    stragglers = processor.scan_for_stragglers()

    if args.scan_only:
        logger.info("\n--scan-only mode: exiting without processing")
        sys.exit(0)

    # Process stragglers
    processor.process_all_stragglers(stragglers)


if __name__ == '__main__':
    main()
