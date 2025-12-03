#!/usr/bin/env python3
"""
test_overlap_show_all_velocity_data.py - Decode velocity data for coordinate space analysis.

This script:
1. Loads coordinate space mapping files
2. For each coordinate, loads the velocity cube, encodes it, then decodes it
3. Extracts individual velocity components (vx, vy, vz) from decoded cubes
4. Stores results in a new dataframe with velocity information

Output format: df_compounding_comparison_decoded_velocity_map_{TIME}.pkl.gz
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Change to parent directory
os.chdir(PARENT_DIR)

# Import model and utilities
from encoder.model_WAE_01 import WAE  # noqa: E402
from Ordered_001_Initialize import HostPreferences  # noqa: E402

# Configuration
OVERLAP_ANALYSIS_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"
MODEL_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt"
DATA_BASE_DIR = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path, device):
    """
    Load the trained WAE model.

    Args:
        model_path: Path to the saved model
        device: torch device to load model onto

    Returns:
        Loaded WAE model in eval mode
    """
    logger.info(f"Loading model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Create model with saved config
    model = WAE(
        input_dim=model_config.get('input_dim', 125 * 3),  # 125 points * 3 velocities
        latent_dim=model_config.get('latent_dim', 32),
        hidden_dims=model_config.get('hidden_dims', [256, 128, 64])
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully. Latent dim: {model.latent_dim}")
    return model


def load_velocity_cube(dataset_name, time, x, y, z, data_base_dir):
    """
    Load velocity cube for a specific coordinate and time.

    Args:
        dataset_name: Dataset name (e.g., "7p2")
        time: Time step
        x, y, z: Coordinates
        data_base_dir: Base directory for data

    Returns:
        Velocity cube as numpy array of shape (125, 3) or None if not found
    """
    # Construct path to pickle file
    pickle_path = Path(data_base_dir) / dataset_name / f"{time}.pkl"

    if not pickle_path.exists():
        logger.warning(f"Pickle file not found: {pickle_path}")
        return None

    try:
        # Load the pickle file
        df = pd.read_pickle(pickle_path, compression='gzip')

        # Find the row matching our coordinates
        mask = (df['x'] == x) & (df['y'] == y) & (df['z'] == z)
        matching_rows = df[mask]

        if len(matching_rows) == 0:
            logger.warning(f"No data found for coordinates ({x}, {y}, {z}) in {pickle_path}")
            return None

        # Get the velocity cube (assuming it's stored in a column called 'velocities' or similar)
        # Adjust this based on actual data structure
        row = matching_rows.iloc[0]

        # Extract velocity data - adjust column names as needed
        if 'velocity_cube' in row:
            velocity_cube = row['velocity_cube']
        elif 'velocities' in row:
            velocity_cube = row['velocities']
        else:
            # Try to reconstruct from vx, vy, vz columns if they exist
            logger.warning(f"Standard velocity column not found, attempting reconstruction")
            return None

        return np.array(velocity_cube)

    except Exception as e:
        logger.error(f"Error loading velocity cube: {e}")
        return None


def decode_velocity_cube(model, velocity_cube, device):
    """
    Encode and decode a velocity cube using the WAE model.

    Args:
        model: Trained WAE model
        velocity_cube: Velocity cube array of shape (125, 3)
        device: torch device

    Returns:
        Decoded velocity cube as numpy array of shape (125, 3)
    """
    # Flatten velocity cube to (375,) for model input
    velocity_flat = velocity_cube.flatten()

    # Convert to tensor and add batch dimension
    velocity_tensor = torch.from_numpy(velocity_flat).float().unsqueeze(0).to(device)

    # Encode and decode
    with torch.no_grad():
        # Encode to latent space
        z = model.encode(velocity_tensor)

        # Decode back to velocity space
        decoded_tensor = model.decode(z)

    # Convert back to numpy and reshape
    decoded_cube = decoded_tensor.cpu().numpy().reshape(125, 3)

    return decoded_cube


def process_coordinate_map_file(coord_map_path, model, device, dataset_name, time, data_base_dir):
    """
    Process a coordinate map file and generate decoded velocity data.

    Args:
        coord_map_path: Path to coordinate map pickle file
        model: Trained WAE model
        device: torch device
        dataset_name: Dataset name
        time: Time step
        data_base_dir: Base directory for data

    Returns:
        DataFrame with decoded velocity data
    """
    logger.info(f"Processing coordinate map: {coord_map_path}")

    # Load coordinate map
    coord_df = pd.read_pickle(coord_map_path, compression='gzip')
    logger.info(f"Loaded coordinate map with {len(coord_df)} rows")

    # Create column names for decoded velocities
    # Same structure as coordinate map, but with __vx, __vy, __vz suffixes for each position
    columns = ['x_y_z']  # Center coordinate

    # Generate column names for all neighbors (same as coordinate map, excluding center 0,0,0)
    for x_offset in range(-2, 3):
        for y_offset in range(-2, 3):
            for z_offset in range(-2, 3):
                # Skip the center point (0, 0, 0)
                if x_offset == 0 and y_offset == 0 and z_offset == 0:
                    continue

                # Format: x_minus_2__y_minus_1__z_0 (same as coordinate map)
                x_str = f"x_minus_{abs(x_offset)}" if x_offset < 0 else f"x_plus_{x_offset}" if x_offset > 0 else "x_0"
                y_str = f"y_minus_{abs(y_offset)}" if y_offset < 0 else f"y_plus_{y_offset}" if y_offset > 0 else "y_0"
                z_str = f"z_minus_{abs(z_offset)}" if z_offset < 0 else f"z_plus_{z_offset}" if z_offset > 0 else "z_0"

                base_name = f"{x_str}__{y_str}__{z_str}"

                # Add three columns for each position: vx, vy, vz
                columns.append(f"{base_name}__vx")
                columns.append(f"{base_name}__vy")
                columns.append(f"{base_name}__vz")

    # Initialize result dataframe
    velocity_df = pd.DataFrame(index=range(len(coord_df)), columns=columns, dtype=float)

    # Process each row
    logger.info("Processing coordinates and decoding velocities...")
    for idx, row in tqdm(coord_df.iterrows(), total=len(coord_df), desc="Decoding velocities"):
        # Parse center coordinate
        center_coord = row['x_y_z']
        x, y, z = map(int, center_coord.split('_'))

        # Store center coordinate
        velocity_df.at[idx, 'x_y_z'] = center_coord

        # Process all neighbors (excluding center, same as coordinate map)
        for x_offset in range(-2, 3):
            for y_offset in range(-2, 3):
                for z_offset in range(-2, 3):
                    # Skip the center point (0, 0, 0)
                    if x_offset == 0 and y_offset == 0 and z_offset == 0:
                        continue

                    # Calculate neighbor coordinate
                    neighbor_x = x + x_offset
                    neighbor_y = y + y_offset
                    neighbor_z = z + z_offset

                    # Load velocity cube
                    velocity_cube = load_velocity_cube(
                        dataset_name, time, neighbor_x, neighbor_y, neighbor_z, data_base_dir
                    )

                    if velocity_cube is not None and velocity_cube.shape == (125, 3):
                        # Decode the velocity cube
                        decoded_cube = decode_velocity_cube(model, velocity_cube, device)

                        # Extract velocity components (using mean of decoded cube)
                        vx_mean = decoded_cube[:, 0].mean()
                        vy_mean = decoded_cube[:, 1].mean()
                        vz_mean = decoded_cube[:, 2].mean()

                        # Generate column names (same format as coordinate map)
                        x_str = f"x_minus_{abs(x_offset)}" if x_offset < 0 else f"x_plus_{x_offset}" if x_offset > 0 else "x_0"
                        y_str = f"y_minus_{abs(y_offset)}" if y_offset < 0 else f"y_plus_{y_offset}" if y_offset > 0 else "y_0"
                        z_str = f"z_minus_{abs(z_offset)}" if z_offset < 0 else f"z_plus_{z_offset}" if z_offset > 0 else "z_0"

                        base_name = f"{x_str}__{y_str}__{z_str}"
                        vx_col = f"{base_name}__vx"
                        vy_col = f"{base_name}__vy"
                        vz_col = f"{base_name}__vz"

                        # Store velocity components
                        velocity_df.at[idx, vx_col] = vx_mean
                        velocity_df.at[idx, vy_col] = vy_mean
                        velocity_df.at[idx, vz_col] = vz_mean
                    else:
                        # Store NaN for missing data
                        logger.debug(f"Missing velocity data for ({neighbor_x}, {neighbor_y}, {neighbor_z})")

    return velocity_df


def main():
    """Main processing function."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(MODEL_PATH, device)

    # Find all coordinate map files
    overlap_dir = Path(OVERLAP_ANALYSIS_DIR)
    coord_map_files = sorted(list(overlap_dir.rglob("df_compounding_comparison_coordinate_space_map_*.pkl.gz")))

    logger.info(f"Found {len(coord_map_files)} coordinate map files to process")

    # Process each file
    for coord_map_path in coord_map_files:
        # Extract dataset name and time from path
        # e.g., 7p2/df_compounding_comparison_coordinate_space_map_1000.pkl.gz
        rel_path = coord_map_path.relative_to(overlap_dir)
        dataset_name = rel_path.parts[0]
        filename = rel_path.name

        # Extract time from filename
        time_str = filename.replace('df_compounding_comparison_coordinate_space_map_', '').replace('.pkl.gz', '')
        time = int(time_str)

        logger.info(f"Processing dataset: {dataset_name}, time: {time}")

        # Process the coordinate map file
        velocity_df = process_coordinate_map_file(
            coord_map_path, model, device, dataset_name, time, DATA_BASE_DIR
        )

        # Save the velocity dataframe
        output_filename = f"df_compounding_comparison_decoded_velocity_map_{time:04d}.pkl.gz"
        output_dir = overlap_dir / dataset_name
        output_path = output_dir / output_filename

        logger.info(f"Saving to {output_path}")
        velocity_df.to_pickle(output_path, compression='gzip')
        logger.info(f"Saved velocity data with shape {velocity_df.shape}")

    logger.info("All files processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
