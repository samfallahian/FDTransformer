#!/usr/bin/env python3
"""
generate_decoded_velocity_analysis.py - Generate decoded velocity data for spatial analysis.

PURPOSE:
This script analyzes how the WAE autoencoder reconstructs velocity fields by comparing
decoded velocities against their spatial positions. This enables analysis of reconstruction
quality as a function of spatial location, which can be compared against centroid-based
encoding approaches.

PROCESS:
1. Loads a dataset for a specific time step and dataset name
2. For each unique (x,y,z) coordinate in the dataset:
   - Uses CoordinateSpace to determine the 125 neighboring grid positions (5x5x5 cube)
   - Loads the velocity cube (125 points × 3 velocity components = 375 values)
   - Encodes the velocity cube to latent space, then decodes it back
   - Records both the decoded velocities AND their corresponding spatial positions

OUTPUT:
Generates TWO synchronized files per time step in OUTPUT_DIR/{dataset}/
   - df_decoded_velocity_{time}.pkl.gz:
     Columns: x_y_z, vx_1, vy_1, vz_1, ..., vx_125, vy_125, vz_125
     Contains the decoded velocity values (375 columns)

   - df_position_mapping_{time}.pkl.gz:
     Columns: x_y_z, x_1, y_1, z_1, ..., x_125, y_125, z_125
     Contains the actual spatial coordinates (375 columns)

Both files share the same 'x_y_z' index column (format: "x_y_z" e.g., "50_35_12")
and can be joined for spatial analysis of reconstruction errors.

USAGE:
    python encoder/generate_decoded_velocity_analysis.py --dataset 7p2 --time 1000

ARGUMENTS:
    --dataset: Dataset name (e.g., "7p2", "3p6", "11p4")
               This should match a subdirectory in DATA_BASE_DIR

    --time:    Time step to process (e.g., 1000, 1500, 2000)
               This should match a pickle file: {time}.pkl in the dataset directory

EXAMPLE:
    python encoder/generate_decoded_velocity_analysis.py --dataset 7p2 --time 1000

    This will:
    - Load /Users/kkreth/PycharmProjects/data/all_data_ready_for_training/7p2/1000.pkl
    - Process all coordinates in that file
    - Output to /Users/kkreth/PycharmProjects/data/overlap_analysis/7p2/
      * df_decoded_velocity_1000.pkl.gz
      * df_position_mapping_1000.pkl.gz
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging
import json

# Resolve project directories
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Change to parent directory
os.chdir(PARENT_DIR)

# Import model and utilities
from encoder.model_WAE_01 import WAE  # noqa: E402
from Ordered_001_Initialize import HostPreferences  # noqa: E402
from CoordinateSpace import givenXYZreplyVelocityCube  # noqa: E402

# Configuration
MODEL_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt"
DATA_BASE_DIR = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training"
OUTPUT_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"

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
    try:
        # PyTorch 2.6+ defaults to weights_only=True, which blocks globals like TorchVersion.
        # We set it to False to allow full unpickling for trusted local checkpoints.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support the weights_only argument
        checkpoint = torch.load(model_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Create model with saved config
    try:
        model = WAE(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dims=model_config['hidden_dims']
        )
    except (KeyError, TypeError):
        # Fallback if config is missing keys or WAE class doesn't accept these arguments
        # This handles the case where WAE is defined as in model_WAE_01.py (hardcoded dims)
        logger.info("Initializing WAE with default parameters.")
        model = WAE()

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")
    return model


def load_velocity_cube(df, x, y, z):
    """
    Load velocity cube for a specific coordinate from a dataframe.

    Args:
        df: DataFrame containing velocity data
        x, y, z: Coordinates

    Returns:
        Velocity cube as numpy array of shape (125, 3) or None if not found
    """
    # Find the row matching our coordinates
    mask = (df['x'] == x) & (df['y'] == y) & (df['z'] == z)
    matching_rows = df[mask]

    if len(matching_rows) == 0:
        return None

    row = matching_rows.iloc[0]

    # Extract velocity data in the correct order: vx_1, vy_1, vz_1, ..., vx_125, vy_125, vz_125
    velocity_data = []
    for i in range(1, 126):  # 1 to 125
        vx = row[f'vx_{i}']
        vy = row[f'vy_{i}']
        vz = row[f'vz_{i}']
        velocity_data.extend([vx, vy, vz])

    # Reshape to (125, 3)
    velocity_cube = np.array(velocity_data).reshape(125, 3)
    return velocity_cube


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


def get_position_mapping(dataset_name, x, y, z):
    """
    Get the 125 position coordinates for a given center coordinate.

    Args:
        dataset_name: Name of the dataset (e.g., "7p2.pkl")
        x, y, z: Center coordinates

    Returns:
        List of 125 (x, y, z) tuples in the order matching CoordinateSpace
    """
    coordinator = givenXYZreplyVelocityCube(
        pickle_filename=dataset_name,
        x=x,
        y=y,
        z=z
    )

    # Get the 125 neighbor coordinates in the correct order
    triplets = coordinator.locateNeighbors()

    return triplets


def process_dataset(dataset_name, time, model, device, data_base_dir, output_dir):
    """
    Process a single dataset file and generate velocity and position mapping files.

    Args:
        dataset_name: Dataset name (e.g., "7p2")
        time: Time step
        model: Trained WAE model
        device: torch device
        data_base_dir: Base directory for data
        output_dir: Output directory for results

    Returns:
        Tuple of (velocity_df, position_df)
    """
    logger.info(f"Processing dataset: {dataset_name}, time: {time}")

    # Construct path to pickle file
    pickle_filename = f"{time}.pkl"
    pickle_path = Path(data_base_dir) / dataset_name / pickle_filename

    if not pickle_path.exists():
        logger.error(f"Pickle file not found: {pickle_path}")
        return None, None

    # Load the dataset
    logger.info(f"Loading dataset from {pickle_path}")
    df = pd.read_pickle(pickle_path, compression='gzip')
    logger.info(f"Loaded {len(df)} rows")

    # Get unique coordinates
    coordinates = df[['x', 'y', 'z']].drop_duplicates().values
    logger.info(f"Found {len(coordinates)} unique coordinates")

    # Initialize result dataframes
    velocity_columns = ['x_y_z'] + [f'{comp}_{i}' for i in range(1, 126) for comp in ['vx', 'vy', 'vz']]
    position_columns = ['x_y_z'] + [f'{comp}_{i}' for i in range(1, 126) for comp in ['x', 'y', 'z']]

    velocity_rows = []
    position_rows = []

    # Process each coordinate
    logger.info("Processing coordinates and decoding velocities...")
    for x, y, z in tqdm(coordinates, desc="Processing coordinates"):
        x, y, z = int(x), int(y), int(z)

        # Create coordinate identifier
        coord_id = f"{x}_{y}_{z}"

        try:
            # Get position mapping for this coordinate (using dataset name, not time filename)
            dataset_pkl_name = f"{dataset_name}.pkl"
            position_triplets = get_position_mapping(dataset_pkl_name, x, y, z)

            # Load velocity cube for this coordinate
            velocity_cube = load_velocity_cube(df, x, y, z)

            if velocity_cube is None:
                logger.warning(f"No velocity data found for ({x}, {y}, {z})")
                continue

            # Decode the velocity cube
            decoded_cube = decode_velocity_cube(model, velocity_cube, device)

            # Create velocity row: flatten decoded_cube (125, 3) to (375,) in order vx_1, vy_1, vz_1, ...
            velocity_row = {'x_y_z': coord_id}
            for i in range(125):
                velocity_row[f'vx_{i+1}'] = decoded_cube[i, 0]
                velocity_row[f'vy_{i+1}'] = decoded_cube[i, 1]
                velocity_row[f'vz_{i+1}'] = decoded_cube[i, 2]
            velocity_rows.append(velocity_row)

            # Create position row: flatten position_triplets to match velocity ordering
            position_row = {'x_y_z': coord_id}
            for i, (px, py, pz) in enumerate(position_triplets):
                position_row[f'x_{i+1}'] = px
                position_row[f'y_{i+1}'] = py
                position_row[f'z_{i+1}'] = pz
            position_rows.append(position_row)

        except Exception as e:
            logger.error(f"Error processing coordinate ({x}, {y}, {z}): {e}")
            continue

    # Create dataframes
    velocity_df = pd.DataFrame(velocity_rows, columns=velocity_columns)
    position_df = pd.DataFrame(position_rows, columns=position_columns)

    logger.info(f"Created velocity dataframe with shape {velocity_df.shape}")
    logger.info(f"Created position dataframe with shape {position_df.shape}")

    # Save the dataframes
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    velocity_output = output_path / f"df_decoded_velocity_{time:04d}.pkl.gz"
    position_output = output_path / f"df_position_mapping_{time:04d}.pkl.gz"

    logger.info(f"Saving velocity data to {velocity_output}")
    velocity_df.to_pickle(velocity_output, compression='gzip')

    logger.info(f"Saving position mapping to {position_output}")
    position_df.to_pickle(position_output, compression='gzip')

    return velocity_df, position_df


def main():
    """Main processing function."""
    import argparse

    parser = argparse.ArgumentParser(description='Decode velocity data for coordinate space analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., "7p2")')
    parser.add_argument('--time', type=int, required=True, help='Time step (e.g., 1000)')
    args = parser.parse_args()

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

    # Process the dataset
    velocity_df, position_df = process_dataset(
        args.dataset, args.time, model, device, DATA_BASE_DIR, OUTPUT_DIR
    )

    if velocity_df is not None and position_df is not None:
        logger.info("Processing completed successfully!")
        return 0
    else:
        logger.error("Processing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
