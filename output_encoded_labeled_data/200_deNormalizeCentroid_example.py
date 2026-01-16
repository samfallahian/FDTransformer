"""
DeNormalize Centroid Example - Transform roundtrip centroid velocities back to original space

This script takes the roundtrip file created by 100_create_validation_example.py and:
1. Extracts the normalized centroid velocities: vx_63_rt, vy_63_rt, vz_63_rt
2. Applies the unconvert transformation using FloatConverter to denormalize them
3. Creates new columns: centroid_x_rt, centroid_y_rt, centroid_z_rt
4. Calculates total_loss as the sum of absolute differences from original centroids
   total_loss = |centroid_x_rt - vx_original| + |centroid_y_rt - vy_original| + |centroid_z_rt - vz_original|
5. Saves the updated file back to the same location
"""

import pickle
import gzip
import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Add the root directory to the path for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the FloatConverter
from TransformLatent import FloatConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas FutureWarning about dtype incompatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class CentroidDeNormalizer:
    """DeNormalize centroid velocities and calculate reconstruction loss"""

    def __init__(self):
        """Initialize the denormalizer with FloatConverter"""
        self.converter = FloatConverter()
        logger.info("FloatConverter initialized")
        logger.info(f"  Min value: {self.converter.min_value}")
        logger.info(f"  Max value: {self.converter.max_value}")

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

    def denormalize_centroids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalize the roundtrip centroid velocities and calculate loss.

        Creates columns:
        - centroid_x_rt, centroid_y_rt, centroid_z_rt (denormalized)
        - total_loss (sum of absolute differences from original)

        Args:
            df: DataFrame with vx_63_rt, vy_63_rt, vz_63_rt columns

        Returns:
            DataFrame with added centroid and loss columns
        """
        logger.info("DeNormalizing centroid velocities...")
        df_copy = df.copy()

        # Check for required roundtrip centroid columns
        rt_centroid_cols = ['vx_63_rt', 'vy_63_rt', 'vz_63_rt']
        missing_rt = [col for col in rt_centroid_cols if col not in df_copy.columns]
        if missing_rt:
            raise ValueError(f"Missing roundtrip centroid columns: {missing_rt}")

        # Check for original centroid columns
        original_cols = ['vx_original', 'vy_original', 'vz_original']
        missing_orig = [col for col in original_cols if col not in df_copy.columns]
        if missing_orig:
            raise ValueError(f"Missing original centroid columns: {missing_orig}")

        logger.info("  Extracting normalized roundtrip centroid values...")
        logger.info(f"  Total rows to process: {len(df_copy)}")

        # Extract the normalized roundtrip centroid values
        vx_63_rt_normalized = df_copy['vx_63_rt'].to_numpy()
        vy_63_rt_normalized = df_copy['vy_63_rt'].to_numpy()
        vz_63_rt_normalized = df_copy['vz_63_rt'].to_numpy()

        logger.info("  Applying unconvert transformation...")
        # Apply the unconvert transformation to denormalize
        centroid_x_rt = self.converter.unconvert(vx_63_rt_normalized)
        centroid_y_rt = self.converter.unconvert(vy_63_rt_normalized)
        centroid_z_rt = self.converter.unconvert(vz_63_rt_normalized)

        # Add denormalized columns to dataframe
        df_copy['centroid_x_rt'] = centroid_x_rt.astype(np.float32)
        df_copy['centroid_y_rt'] = centroid_y_rt.astype(np.float32)
        df_copy['centroid_z_rt'] = centroid_z_rt.astype(np.float32)

        logger.info("  Calculating total_loss...")
        # Calculate total loss as sum of absolute differences
        vx_original = df_copy['vx_original'].to_numpy()
        vy_original = df_copy['vy_original'].to_numpy()
        vz_original = df_copy['vz_original'].to_numpy()

        total_loss = (np.abs(centroid_x_rt - vx_original) +
                     np.abs(centroid_y_rt - vy_original) +
                     np.abs(centroid_z_rt - vz_original))

        df_copy['total_loss'] = total_loss.astype(np.float32)

        # Log statistics
        logger.info("  Statistics:")
        logger.info(f"    centroid_x_rt: min={centroid_x_rt.min():.6f}, max={centroid_x_rt.max():.6f}, mean={centroid_x_rt.mean():.6f}")
        logger.info(f"    centroid_y_rt: min={centroid_y_rt.min():.6f}, max={centroid_y_rt.max():.6f}, mean={centroid_y_rt.mean():.6f}")
        logger.info(f"    centroid_z_rt: min={centroid_z_rt.min():.6f}, max={centroid_z_rt.max():.6f}, mean={centroid_z_rt.mean():.6f}")
        logger.info(f"    total_loss: min={total_loss.min():.6f}, max={total_loss.max():.6f}, mean={total_loss.mean():.6f}, median={np.median(total_loss):.6f}")

        logger.info(f"  ✅ Completed processing {len(df_copy)} rows")
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
        Process a single file to denormalize centroids and calculate loss

        Args:
            input_path: Path to input .roundtrip.pkl.gz file
            output_path: Path to output file (same or different)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"{'='*60}")

        # Load the file
        df = self.load_pickle(input_path)

        # Denormalize centroids and calculate loss
        df_with_loss = self.denormalize_centroids(df)

        # Save the result
        self.save_pickle_compressed(df_with_loss, output_path)

        logger.info(f"✅ SUCCESS: Created {output_path.name}")
        logger.info(f"{'='*60}\n")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='DeNormalize centroid velocities and calculate reconstruction loss')
    parser.add_argument('--input-file', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate/8p4/160.roundtrip.pkl.gz',
                       help='Path to input .roundtrip.pkl.gz file')
    parser.add_argument('--output-file', type=str,
                       default='/Users/kkreth/PycharmProjects/data/all_data_ready_to_populate/8p4/160.roundtrip.pkl.gz',
                       help='Path to output file (default: overwrites input)')

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Create output path
    output_path = Path(args.output_file)

    # Create denormalizer and process
    denormalizer = CentroidDeNormalizer()
    denormalizer.process_file(input_path, output_path)


if __name__ == '__main__':
    main()
