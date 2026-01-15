"""
Create Validation Example - Test script for roundtrip velocity reconstruction

This script takes a single file and creates roundtrip velocities by:
1. Loading the file with latent columns (latent_1 through latent_47)
2. Decoding latent vectors back to velocities using the WAE model
3. Adding 375 new columns with _rt suffix: vx_1_rt, vy_1_rt, vz_1_rt ... vz_125_rt
4. Saving to a new .roundtrip.pkl.gz file

This will eventually be extended with assertions to validate the roundtrip quality.
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
from typing import Tuple

# Add the root directory to the path for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the WAE model
from encoder.model_WAE_01 import WAE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning about dtype incompatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class RoundtripValidator:
    """Create roundtrip velocities from latent space for validation"""

    def __init__(self, model_path: str):
        """
        Initialize the validator

        Args:
            model_path: Path to the WAE model file
        """
        self.model_path = model_path
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

    def load_pickle(self, file_path: Path) -> pd.DataFrame:
        """
        Load a pickle file (handles gzip compression)

        Returns:
            DataFrame
        """
        logger.info(f"Loading: {file_path}")

        try:
            with gzip.open(file_path, 'rb') as f:
                df = pickle.load(f)
                logger.info(f"  Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
                return df
        except (OSError, gzip.BadGzipFile):
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                logger.info(f"  Successfully loaded: {len(df)} rows, {len(df.columns)} columns")
                return df

    def create_roundtrip_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create roundtrip velocities by decoding latent vectors back to velocity space.

        Adds 375 columns: vx_1_rt, vy_1_rt, vz_1_rt through vx_125_rt, vy_125_rt, vz_125_rt

        Args:
            df: DataFrame with latent columns (latent_1 through latent_47)

        Returns:
            DataFrame with added roundtrip velocity columns
        """
        logger.info("Creating roundtrip velocities from latent space...")
        df_copy = df.copy()

        # Extract latent columns
        latent_cols = [f'latent_{i}' for i in range(1, 48)]
        missing_cols = [col for col in latent_cols if col not in df_copy.columns]

        if missing_cols:
            raise ValueError(f"Missing {len(missing_cols)} latent columns: {missing_cols[:5]}...")

        # Initialize roundtrip velocity columns with float32 dtype
        roundtrip_cols = []
        for i in range(1, 126):
            roundtrip_cols.extend([f'vx_{i}_rt', f'vy_{i}_rt', f'vz_{i}_rt'])

        # Verify we have exactly 375 columns
        assert len(roundtrip_cols) == 375, f"Expected 375 roundtrip columns, got {len(roundtrip_cols)}"
        logger.info(f"  Creating {len(roundtrip_cols)} roundtrip velocity columns")

        for col in roundtrip_cols:
            df_copy[col] = np.float32(0.0)

        # Extract all latent data as a numpy array
        logger.info("  Extracting latent data...")
        latent_data = df_copy[latent_cols].to_numpy().astype(np.float32)
        logger.info(f"  Latent data shape: {latent_data.shape}")

        # Process in batches
        total_rows = len(df_copy)
        batch_size = 512
        num_batches = (total_rows + batch_size - 1) // batch_size
        logger.info(f"  Processing in {num_batches} batches of size {batch_size}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
                logger.info(f"  Batch {batch_idx+1}/{num_batches}: rows {start_idx}-{end_idx}")

            # Get batch of latent data
            batch_latent = latent_data[start_idx:end_idx]

            # Convert to tensor and move to device
            latent_tensor = torch.tensor(batch_latent, dtype=torch.float32).to(self.device)

            # Decode latent to velocities
            with torch.no_grad():
                velocity_batch = self.model.decode(latent_tensor)

            # Move to CPU and convert to numpy
            velocity_np = velocity_batch.cpu().numpy()

            # Verify shape: should be (batch_size, 375)
            assert velocity_np.shape[1] == 375, f"Expected 375 velocities, got {velocity_np.shape[1]}"

            # Update the dataframe with batch results
            df_indices = df_copy.index[start_idx:end_idx]
            for i, col in enumerate(roundtrip_cols):
                df_copy.loc[df_indices, col] = velocity_np[:, i].astype(np.float32)

        logger.info(f"  ✅ Completed decoding {total_rows} rows")
        return df_copy

    def save_pickle_compressed(self, df: pd.DataFrame, output_path: Path):
        """Save DataFrame as gzip-compressed pickle"""
        logger.info(f"Saving to: {output_path}")
        logger.info(f"  DataFrame: {len(df)} rows, {len(df.columns)} columns")

        with gzip.open(output_path, 'wb', compresslevel=9) as f:
            pickle.dump(df, f)

        file_size = output_path.stat().st_size
        logger.info(f"  Saved successfully: {file_size:,} bytes")

    def process_file(self, input_path: Path, output_path: Path):
        """
        Process a single file to create roundtrip velocities

        Args:
            input_path: Path to input .pkl.gz file
            output_path: Path to output .roundtrip.pkl.gz file
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"{'='*60}")

        # Load the file
        df = self.load_pickle(input_path)

        # Create roundtrip velocities
        df_with_roundtrip = self.create_roundtrip_velocities(df)

        # Save the result
        self.save_pickle_compressed(df_with_roundtrip, output_path)

        logger.info(f"✅ SUCCESS: Created {output_path.name}")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Create validation example with roundtrip velocities')
    parser.add_argument('--input-file', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate/8p4/160.pkl.gz',
                       help='Path to input .pkl.gz file')
    parser.add_argument('--output-file', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate/8p4/160.roundtrip.pkl.gz',
                       help='Path to output .roundtrip.pkl.gz file')
    parser.add_argument('--model-path', type=str,
                       default='/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt',
                       help='Path to the WAE model file')

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)

    # Create output path
    output_path = Path(args.output_file)

    # Create validator and process
    validator = RoundtripValidator(args.model_path)
    validator.process_file(input_path, output_path)


if __name__ == '__main__':
    main()
