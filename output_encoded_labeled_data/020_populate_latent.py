"""
Populate Latent Data Fields - Generate latent space representations using WAE model

This script walks through all pickle files in the data directory and:
1. Loads each file (expects velocity columns vx_1 through vz_125)
2. Uses the WAE model to encode velocity data into 47-dimensional latent space
3. Populates the latent_1 through latent_47 columns with the encoded values
4. Saves the updated file with gzip compression

This script is designed to process files that have been processed by 010_NullOutDataFields.py
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
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Add the root directory to the path for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Model imports - we'll dynamically load the correct one from checkpoint
# Keep legacy import for backwards compatibility with old checkpoints
try:
    from encoder.model_WAE_01 import WAE
except ImportError:
    WAE = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning about dtype incompatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class LatentPopulator:
    """Process pickle files to populate latent space from velocity data"""

    def __init__(self, data_root: str, model_path: str, n_threads: int = 6):
        """
        Initialize the processor

        Args:
            data_root: Root directory containing subdirectories with pkl files
            model_path: Path to the WAE model file
            n_threads: Number of threads for parallel processing (default: 10)
        """
        self.data_root = Path(data_root)
        self.model_path = model_path
        self.model_filename = Path(model_path).name
        self.n_threads = n_threads
        self.print_lock = Lock()

        # Load the model once (will be shared across threads)
        self.model, self.device = self._load_wae_model()
        logger.info(f"Model loaded on device: {self.device}")

    def _load_wae_model(self):
        """
        Load the model from the specified path with dynamic architecture detection.

        Supports both:
        1. New checkpoints with embedded model architecture info
        2. Legacy checkpoints (assumes WAE architecture)

        Returns:
            Tuple of (model, device)
        """
        logger.info(f"Loading model from: {self.model_path}")

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else
                             "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
                             "cpu")
        logger.info(f"Using device: {device}")

        # Load checkpoint with compatibility for PyTorch 2.6+ "weights_only" changes
        checkpoint = None
        torch_version = getattr(torch, "__version__", "unknown")
        logger.info(f"PyTorch version detected: {torch_version}")
        try:
            # In PyTorch 2.6+ default weights_only=True can break legacy checkpoints; override explicitly
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning(
                "torch.load with weights_only=False failed on first attempt. Will try safe allowlist path. "
                f"Error: {e}"
            )
            try:
                # Attempt to allowlist TorchVersion which is commonly needed by older checkpoints
                from torch.serialization import add_safe_globals
                try:
                    # TorchVersion class location
                    from torch.torch_version import TorchVersion
                except Exception:
                    TorchVersion = None
                if 'add_safe_globals' in dir(__import__('torch').serialization) and TorchVersion is not None:
                    add_safe_globals([TorchVersion])
                # Retry load; if available, keep weights_only=True for extra safety; otherwise omit
                try:
                    checkpoint = torch.load(self.model_path, map_location=device, weights_only=True)
                except TypeError:
                    # Older torch without weights_only argument
                    checkpoint = torch.load(self.model_path, map_location=device)
            except Exception as e2:
                logger.error(
                    "Failed to load checkpoint even after adding safe globals. "
                    "If you trust the checkpoint source, ensure this process has permissions and the file is not corrupted. "
                    f"Original error: {e}; Fallback error: {e2}"
                )
                raise

        # Dynamically instantiate the correct model architecture
        if isinstance(checkpoint, dict) and 'model_class' in checkpoint and 'model_module' in checkpoint:
            # NEW FORMAT: checkpoint contains architecture info
            model_class_name = checkpoint['model_class']
            model_module_name = checkpoint['model_module']
            model_config = checkpoint.get('model_config', {})

            logger.info(f"✓ Found model architecture info in checkpoint:")
            logger.info(f"  Class: {model_class_name}")
            logger.info(f"  Module: {model_module_name}")
            logger.info(f"  Config: {model_config}")

            # Dynamically import the model class
            try:
                # Import the module
                import importlib
                module = importlib.import_module(model_module_name)
                model_class = getattr(module, model_class_name)

                # Instantiate with config
                model = model_class(**model_config)
                logger.info(f"✓ Successfully instantiated {model_class_name}")

            except Exception as e:
                logger.error(f"Failed to dynamically load model class {model_class_name} from {model_module_name}: {e}")
                raise RuntimeError(
                    f"Could not load model architecture. The checkpoint specifies {model_class_name} "
                    f"from module {model_module_name}, but it could not be imported. "
                    f"Error: {e}"
                )
        else:
            # LEGACY FORMAT: checkpoint doesn't have architecture info, assume WAE
            logger.warning("⚠ Checkpoint does not contain model architecture info (legacy format)")
            logger.warning("  Falling back to hardcoded WAE architecture")
            logger.warning("  Future checkpoints should include 'model_class' and 'model_module' fields")

            if WAE is None:
                raise RuntimeError(
                    "Legacy checkpoint detected but WAE class could not be imported. "
                    "Please ensure encoder.model_WAE_01 is available or use a checkpoint "
                    "that includes model architecture info."
                )

            model = WAE()
            logger.info("✓ Using legacy WAE architecture")

        model = model.to(device)

        # Extract and load the model state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # Set to evaluation mode
        model.eval()

        logger.info("✓ Model loaded successfully")
        return model, device

    def load_pickle(self, file_path: Path) -> pd.DataFrame:
        """Load a pickle file (handles gzip compression and metadata)"""
        try:
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
        except (OSError, gzip.BadGzipFile):
            # Try without gzip if not compressed
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        # Handle both raw DataFrames and dict with metadata
        if isinstance(data, dict) and 'dataframe' in data:
            return data['dataframe']
        else:
            return data

    def extract_velocity_data(self, row: pd.Series) -> np.ndarray:
        """
        Extract velocity data from a single row into a normalized array.

        Args:
            row: Series containing velocity data

        Returns:
            Array of 375 normalized velocity values
        """
        # Extract velocity columns (vx_1 to vz_125)
        velocity_cols = []
        for i in range(1, 126):
            velocity_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])

        # Check if all columns exist
        missing_cols = [col for col in velocity_cols if col not in row.index]
        if missing_cols:
            raise ValueError(f"Row is missing required velocity columns")

        # Extract velocity data into numpy array
        raw_velocities = row[velocity_cols].to_numpy()

        # Ensure we have 375 velocity values (125 points × 3 values per point)
        if raw_velocities.shape[0] != 375:
            raise ValueError(f"Expected 375 velocity values, got {raw_velocities.shape[0]}")

        return raw_velocities.astype(np.float32)

    def compute_latent_space(self, velocity_data: np.ndarray) -> np.ndarray:
        """
        Compute the latent space representation using the WAE model.

        Args:
            velocity_data: Array of 375 velocity values

        Returns:
            The 47-dimensional latent space representation
        """
        # Convert to tensor and move to device
        velocity_tensor = torch.tensor(velocity_data, dtype=torch.float32).to(self.device)

        # Ensure input has correct shape
        if len(velocity_tensor.shape) == 1:
            velocity_tensor = velocity_tensor.unsqueeze(0)  # Add batch dimension

        # Compute latent representation
        with torch.no_grad():
            latent_representation = self.model.encode(velocity_tensor)

        # Move to CPU and convert to numpy
        latent_np = latent_representation.cpu().numpy()

        # Remove batch dimension if present
        if len(latent_np.shape) > 1 and latent_np.shape[0] == 1:
            latent_np = latent_np.squeeze(0)

        return latent_np

    def populate_latent_columns(self, df: pd.DataFrame, file_name: str = "") -> pd.DataFrame:
        """
        Populate latent columns by encoding velocity data using batch processing.

        Args:
            df: Input DataFrame with velocity data
            file_name: Name of file being processed (for logging)

        Returns:
            DataFrame with populated latent columns
        """
        df_copy = df.copy()

        # Ensure latent columns exist with float32 dtype
        for i in range(1, 48):
            if f'latent_{i}' not in df_copy.columns:
                df_copy[f'latent_{i}'] = np.float32(0.0)
            else:
                # Convert existing columns to float32 to avoid dtype warnings
                df_copy[f'latent_{i}'] = df_copy[f'latent_{i}'].astype(np.float32)

        total_rows = len(df_copy)
        logger.info(f"  Processing {file_name}: {total_rows} rows")

        # Extract all velocity columns at once
        velocity_cols = []
        for i in range(1, 126):
            velocity_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])

        # Check if all columns exist
        missing_cols = [col for col in velocity_cols if col not in df_copy.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required velocity columns: {len(missing_cols)} columns")

        # Extract all velocity data as a numpy array (batch processing)
        velocity_data = df_copy[velocity_cols].to_numpy().astype(np.float32)

        # Process in batches for efficiency
        batch_size = 512
        num_batches = (total_rows + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)

            # Get batch of velocity data
            batch_velocities = velocity_data[start_idx:end_idx]

            # Convert to tensor and move to device
            velocity_tensor = torch.tensor(batch_velocities, dtype=torch.float32).to(self.device)

            # Compute latent representations for the entire batch
            with torch.no_grad():
                latent_batch = self.model.encode(velocity_tensor)

            # Move to CPU and convert to numpy
            latent_np = latent_batch.cpu().numpy()

            # Update the dataframe with batch results
            df_indices = df_copy.index[start_idx:end_idx]
            for i in range(1, 48):
                df_copy.loc[df_indices, f'latent_{i}'] = latent_np[:, i-1].astype(np.float32)

        logger.info(f"  Completed encoding {total_rows} rows for {file_name}")
        return df_copy

    def validate_populated_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all latent columns have been populated (not all zeros).

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_still_zero_columns)
        """
        latent_cols = [f'latent_{i}' for i in range(1, 48)]
        still_zero_cols = []

        for col in latent_cols:
            if col in df.columns and (df[col] == 0).all():
                still_zero_cols.append(col)

        is_valid = len(still_zero_cols) == 0
        return is_valid, still_zero_cols

    def save_pickle_compressed(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame as gzip-compressed pickle with metadata"""
        from datetime import datetime

        # Create data structure with metadata
        data_with_metadata = {
            'dataframe': df,
            'metadata': {
                'model_file': f"Created by running the model file {self.model_filename}",
                'processing_date': datetime.now().isoformat(),
                'latent_dimensions': 47
            }
        }

        with gzip.open(output_path, 'wb', compresslevel=9) as f:
            pickle.dump(data_with_metadata, f)

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a single file: load, populate latents, validate, save

        Args:
            input_path: Path to input pickle file
            output_path: Path to output compressed pickle file

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.print_lock:
                logger.info(f"📄 Loading {input_path.name}")

            # Load the file
            df = self.load_pickle(input_path)

            # Populate latent columns
            df = self.populate_latent_columns(df, input_path.name)

            # Validate
            is_valid, zero_cols = self.validate_populated_columns(df)

            if not is_valid:
                with self.print_lock:
                    logger.warning(f"⚠️  Some latent columns still zero for {input_path.name}: {zero_cols}")
                # Note: We continue anyway as some rows might have zero velocities

            # Save with compression
            with self.print_lock:
                logger.info(f"💾 Saving {output_path.name}")
            self.save_pickle_compressed(df, output_path)

            with self.print_lock:
                logger.info(f"✅ Completed {input_path.name}")

            return True

        except Exception as e:
            with self.print_lock:
                logger.error(f"❌ Error processing {input_path.name}: {str(e)}")
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

        # Process the file (overwrite existing files with populated latent data)
        logger.debug(f"Processing {pkl_file.name} -> {output_file.name}")
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

        logger.debug(f"Processing directory: {input_dir}")
        logger.debug(f"Output directory: {output_dir}")

        if not input_dir.exists():
            print(f"❌ Directory not found: {input_dir}")
            return {'error': 'Directory not found'}

        # Find all .pkl.gz files (output from 010_NullOutDataFields.py)
        pkl_files = sorted(input_dir.glob("*.pkl.gz"))
        logger.debug(f"Found {len(pkl_files)} .pkl.gz files")

        # If no .pkl.gz files, try .pkl files
        if not pkl_files:
            pkl_files = sorted(input_dir.glob("*.pkl"))
            logger.debug(f"Found {len(pkl_files)} .pkl files instead")

        # Show first few files found
        if pkl_files:
            logger.debug(f"First file example: {pkl_files[0]}")
            logger.debug(f"Last file example: {pkl_files[-1]}")

        stats = {
            'total': len(pkl_files),
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }

        print(f"\n📁 Processing {subdir_name}: {stats['total']} files with {self.n_threads} threads")

        # Prepare work items
        work_items = []
        for idx, pkl_file in enumerate(pkl_files):
            # Keep same filename but ensure .pkl.gz extension
            if pkl_file.suffix == '.gz':
                output_file = pkl_file  # Already .pkl.gz, overwrite
            else:
                output_file = output_dir / f"{pkl_file.stem}.pkl.gz"

            # Debug first file to see the mapping
            if idx == 0:
                logger.debug(f"First file mapping: {pkl_file} -> {output_file}")
                logger.debug(f"Output file exists: {output_file.exists()}")

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
        logger.debug(f"Data root: {self.data_root}")
        logger.debug(f"Subdirectories: {[d.name for d in subdirs]}")

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

    parser = argparse.ArgumentParser(description='Populate latent data fields using an arbitrary model')
    parser.add_argument('--data-root', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate',
                       help='Root directory containing subdirectories with pkl files')
    parser.add_argument('--model-path', type=str,
                       default='/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/Model_09_Residual_AE_epoch_500.pt',
                       help='Path to the WAE model file')
    parser.add_argument('--subdir', type=str, default=None,
                       help='Process only this subdirectory (e.g., "3p6")')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads for parallel processing (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Report what would be done without actually processing')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    processor = LatentPopulator(args.data_root, args.model_path, n_threads=args.threads)

    if args.subdir:
        # Process single directory
        processor.process_directory(args.subdir, dry_run=args.dry_run)
    else:
        # Process all directories
        processor.process_all_directories(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
