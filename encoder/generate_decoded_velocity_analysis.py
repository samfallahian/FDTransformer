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

# Configure logging - set to DEBUG for detailed progress
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ANSI color codes for rainbow logging
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @staticmethod
    def rainbow(text):
        """Create rainbow effect for text"""
        colors = [Colors.RED, Colors.YELLOW, Colors.GREEN, Colors.CYAN, Colors.BLUE, Colors.MAGENTA]
        result = []
        for i, char in enumerate(text):
            result.append(f"{colors[i % len(colors)]}{char}")
        return ''.join(result) + Colors.RESET

# Initialize host preferences to get correct paths
try:
    host_prefs = HostPreferences()
    # Derive project root from metadata_location
    # metadata_location is like: /home/kkreth_umassd_edu/cgan/configs/Experiment_MetaData.json
    # So project root is two levels up
    metadata_path = Path(host_prefs.metadata_location)
    PROJECT_ROOT = metadata_path.parent.parent

    MODEL_PATH = PROJECT_ROOT / "encoder" / "saved_models" / "Model_09_Residual_AE_epoch_500.pt"
    DATA_BASE_DIR = Path(host_prefs.training_data_path) / "all_data_ready_for_training"
    OUTPUT_DIR = Path(host_prefs.training_data_path) / "overlap_analysis"

    logger.info(f"Initialized paths from HostPreferences:")
    logger.info(f"  Project root: {PROJECT_ROOT}")
    logger.info(f"  Model path: {MODEL_PATH}")
    logger.info(f"  Data base dir: {DATA_BASE_DIR}")
    logger.info(f"  Output dir: {OUTPUT_DIR}")
except Exception as e:
    logger.warning(f"Could not load HostPreferences, using default paths: {e}")
    # Fallback to hardcoded paths
    MODEL_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/Model_09_Residual_AE_epoch_500.pt"
    DATA_BASE_DIR = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training"
    OUTPUT_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"


def load_model(model_path, device):
    """
    Load the trained model with embedded definition or dynamic class loading.

    Args:
        model_path: Path to the saved model
        device: torch device to load model onto

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {model_path}")
    logger.debug(f"Model file size: {Path(model_path).stat().st_size / (1024**2):.2f} MB")
    logger.debug(f"Target device: {device}")

    # Load checkpoint
    logger.debug("Loading checkpoint with torch.load()...")
    try:
        # PyTorch 2.6+ defaults to weights_only=True, which blocks globals like TorchVersion.
        # We set it to False to allow full unpickling for trusted local checkpoints.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        logger.debug("Checkpoint loaded successfully")
    except TypeError:
        # Fallback for older PyTorch versions that don't support the weights_only argument
        logger.debug("Retrying without weights_only parameter (older PyTorch version)")
        checkpoint = torch.load(model_path, map_location=device)
        logger.debug("Checkpoint loaded successfully (compatibility mode)")

    # New checkpoints embed the model definition directly
    if 'model' in checkpoint:
        logger.info("Loading model from embedded definition in checkpoint")
        logger.debug("Extracting model from checkpoint['model']...")
        model = checkpoint['model']
        logger.debug(f"Moving model to device: {device}")
        model = model.to(device)
        logger.debug("Setting model to eval mode...")
        model.eval()
        logger.info("Model loaded successfully from embedded definition.")
        logger.debug(f"Model type: {type(model).__name__}")
        return model

    # Try dynamic loading from model_class and model_module
    if 'model_class' in checkpoint and 'model_module' in checkpoint:
        logger.info(f"Loading model dynamically: {checkpoint['model_module']}.{checkpoint['model_class']}")
        import importlib

        try:
            # Import the module containing the model class
            logger.debug(f"Importing module: {checkpoint['model_module']}")
            module = importlib.import_module(checkpoint['model_module'])
            model_class = getattr(module, checkpoint['model_class'])
            logger.debug(f"Model class: {model_class}")

            # Get model config
            model_config = checkpoint.get('model_config', {})
            logger.debug(f"Model config: {model_config}")

            # Instantiate the model
            logger.debug("Instantiating model...")
            model = model_class(**model_config)

            # Load state dict
            logger.debug("Loading state dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.debug(f"Moving model to device: {device}")
            model = model.to(device)
            logger.debug("Setting model to eval mode...")
            model.eval()

            logger.info("Model loaded successfully using dynamic loading.")
            return model
        except Exception as e:
            logger.error(f"Failed to dynamically load model: {e}")
            logger.debug("Full traceback:", exc_info=True)
            raise

    # Fallback: Try old-style loading for backwards compatibility
    logger.warning("Checkpoint doesn't contain embedded model or model class info, trying legacy loading method")
    model_config = checkpoint.get('model_config', {})
    logger.debug(f"Legacy model config: {model_config}")

    try:
        logger.debug("Instantiating WAE with config parameters...")
        model = WAE(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dims=model_config['hidden_dims']
        )
    except (KeyError, TypeError) as e:
        logger.warning(f"Could not use model_config ({e}), initializing WAE with default parameters.")
        logger.debug("Using default WAE initialization")
        model = WAE()

    logger.debug("Loading state dict...")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.debug(f"Moving model to device: {device}")
    model = model.to(device)
    logger.debug("Setting model to eval mode...")
    model.eval()

    logger.info("Model loaded successfully using legacy method.")
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
    logger.debug(f"  → load_velocity_cube({x}, {y}, {z})")
    # Find the row matching our coordinates
    mask = (df['x'] == x) & (df['y'] == y) & (df['z'] == z)
    matching_rows = df[mask]
    logger.debug(f"  → Found {len(matching_rows)} matching rows")

    if len(matching_rows) == 0:
        return None

    row = matching_rows.iloc[0]

    # Extract velocity data in the correct order: vx_1, vy_1, vz_1, ..., vx_125, vy_125, vz_125
    logger.debug(f"  → Extracting 375 velocity values...")
    velocity_data = []
    for i in range(1, 126):  # 1 to 125
        vx = row[f'vx_{i}']
        vy = row[f'vy_{i}']
        vz = row[f'vz_{i}']
        velocity_data.extend([vx, vy, vz])

    # Reshape to (125, 3)
    velocity_cube = np.array(velocity_data).reshape(125, 3)
    logger.debug(f"  → Created velocity cube shape: {velocity_cube.shape}")
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
    logger.debug(f"  → decode_velocity_cube(shape={velocity_cube.shape})")
    # Flatten velocity cube to (375,) for model input
    velocity_flat = velocity_cube.flatten()
    logger.debug(f"  → Flattened to shape: {velocity_flat.shape}")

    # Convert to tensor and add batch dimension
    logger.debug(f"  → Converting to tensor on device: {device}")
    velocity_tensor = torch.from_numpy(velocity_flat).float().unsqueeze(0).to(device)
    logger.debug(f"  → Tensor shape: {velocity_tensor.shape}")

    # Encode and decode
    with torch.no_grad():
        # Encode to latent space
        logger.debug(f"  → Encoding to latent space...")
        z = model.encode(velocity_tensor)
        logger.debug(f"  → Latent shape: {z.shape}")

        # Decode back to velocity space
        logger.debug(f"  → Decoding back to velocity space...")
        decoded_tensor = model.decode(z)
        logger.debug(f"  → Decoded tensor shape: {decoded_tensor.shape}")

    # Convert back to numpy and reshape
    logger.debug(f"  → Converting back to numpy and reshaping...")
    decoded_cube = decoded_tensor.cpu().numpy().reshape(125, 3)
    logger.debug(f"  → Final decoded cube shape: {decoded_cube.shape}")

    return decoded_cube


def decode_velocity_cubes_batch(model, velocity_cubes, device):
    """
    Encode and decode multiple velocity cubes in a single batch (FASTER).

    Args:
        model: Trained WAE model
        velocity_cubes: List of velocity cube arrays, each of shape (125, 3)
        device: torch device

    Returns:
        List of decoded velocity cubes as numpy arrays of shape (125, 3)
    """
    if len(velocity_cubes) == 0:
        return []

    # Flatten all cubes and stack into batch tensor
    velocity_flats = [cube.flatten() for cube in velocity_cubes]
    velocity_batch = torch.from_numpy(np.stack(velocity_flats)).float().to(device)

    # Encode and decode in batch
    with torch.no_grad():
        z = model.encode(velocity_batch)
        decoded_batch = model.decode(z)

    # Convert back to list of numpy arrays
    decoded_cubes = [decoded_batch[i].cpu().numpy().reshape(125, 3) for i in range(len(velocity_cubes))]

    return decoded_cubes


def get_position_mapping(dataset_name, x, y, z):
    """
    Get the 125 position coordinates for a given center coordinate.

    Args:
        dataset_name: Name of the dataset (e.g., "7p2.pkl")
        x, y, z: Center coordinates

    Returns:
        List of 125 (x, y, z) tuples in the order matching CoordinateSpace
    """
    logger.debug(f"  → get_position_mapping({dataset_name}, {x}, {y}, {z})")
    coordinator = givenXYZreplyVelocityCube(
        pickle_filename=dataset_name,
        x=x,
        y=y,
        z=z
    )
    logger.debug(f"  → locateNeighbors() call...")

    # Get the 125 neighbor coordinates in the correct order
    triplets = coordinator.locateNeighbors()
    logger.debug(f"  → Got {len(triplets)} position triplets")

    return triplets


def process_dataset(dataset_name, time, model, device, data_base_dir, output_dir, batch_size=64):
    """
    Process a single dataset file and generate velocity and position mapping files.

    Args:
        dataset_name: Dataset name (e.g., "7p2")
        time: Time step
        model: Trained WAE model
        device: torch device
        data_base_dir: Base directory for data
        output_dir: Output directory for results
        batch_size: Number of coordinates to process simultaneously (default: 64)

    Returns:
        Tuple of (velocity_df, position_df)
    """
    # Convert paths to Path objects for consistency
    data_base_dir = Path(data_base_dir)
    output_dir = Path(output_dir)

    logger.info(f"Processing dataset: {dataset_name}, time: {time}")

    # Construct path to pickle file
    pickle_filename = f"{time}.pkl"
    pickle_path = Path(data_base_dir) / dataset_name / pickle_filename

    if not pickle_path.exists():
        logger.error(f"Pickle file not found: {pickle_path}")
        return None, None

    # Load the dataset
    logger.info(f"Loading dataset from {pickle_path}")
    logger.debug(f"Reading pickle file: {pickle_path.stat().st_size / (1024**2):.2f} MB")
    df = pd.read_pickle(pickle_path, compression='gzip')
    logger.info(f"Loaded {len(df)} rows")
    logger.debug(f"DataFrame shape: {df.shape}, Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Filter out metadata rows (if any exist)
    # Metadata rows typically have NaN or special marker values for coordinates
    logger.debug("Filtering out any metadata rows...")
    initial_row_count = len(df)

    # Check if there are any rows with NaN in x, y, or z columns
    has_valid_coords = df[['x', 'y', 'z']].notna().all(axis=1)
    df = df[has_valid_coords].copy()

    # Also filter out any rows that don't have velocity columns (metadata indicator)
    # Check if vx_1 column exists as a marker for valid data rows
    if 'vx_1' in df.columns:
        has_velocity = df['vx_1'].notna()
        df = df[has_velocity].copy()

    filtered_count = initial_row_count - len(df)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} metadata/invalid rows")
    logger.info(f"Working with {len(df)} valid data rows")
    logger.debug(f"DataFrame shape after filtering: {df.shape}")

    # Get unique coordinates
    logger.debug("Finding unique coordinates...")
    coordinates = df[['x', 'y', 'z']].drop_duplicates().values
    logger.info(f"Found {len(coordinates)} unique coordinates")
    logger.debug(f"Coordinate range: X[{coordinates[:, 0].min()}-{coordinates[:, 0].max()}], "
                f"Y[{coordinates[:, 1].min()}-{coordinates[:, 1].max()}], "
                f"Z[{coordinates[:, 2].min()}-{coordinates[:, 2].max()}]")

    # Initialize result dataframes
    velocity_columns = ['x_y_z'] + [f'{comp}_{i}' for i in range(1, 126) for comp in ['vx', 'vy', 'vz']]
    position_columns = ['x_y_z'] + [f'{comp}_{i}' for i in range(1, 126) for comp in ['x', 'y', 'z']]

    velocity_rows = []
    position_rows = []

    # Process coordinates in batches for better GPU utilization
    logger.info(f"Processing coordinates and decoding velocities in batches of {batch_size}...")
    logger.debug(f"Will process {len(coordinates)} coordinates total")

    processed_count = 0
    error_count = 0

    # Process in batches
    for batch_start in tqdm(range(0, len(coordinates), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(coordinates))
        batch_coords = coordinates[batch_start:batch_end]

        # Collect data for this batch
        batch_velocity_cubes = []
        batch_position_triplets = []
        batch_coord_ids = []
        batch_indices = []

        for idx_in_batch, (x, y, z) in enumerate(batch_coords):
            x, y, z = int(x), int(y), int(z)
            coord_id = f"{x}_{y}_{z}"

            try:
                # Get position mapping
                dataset_pkl_name = f"{dataset_name}.pkl"
                position_triplets = get_position_mapping(dataset_pkl_name, x, y, z)

                # Load velocity cube
                velocity_cube = load_velocity_cube(df, x, y, z)

                if velocity_cube is None:
                    logger.warning(f"No velocity data found for ({x}, {y}, {z})")
                    error_count += 1
                    continue

                # Collect for batch processing
                batch_velocity_cubes.append(velocity_cube)
                batch_position_triplets.append(position_triplets)
                batch_coord_ids.append(coord_id)
                batch_indices.append(batch_start + idx_in_batch)

            except Exception as e:
                logger.error(f"Error loading data for coordinate ({x}, {y}, {z}): {e}")
                error_count += 1
                continue

        # Decode entire batch at once (MUCH FASTER)
        if len(batch_velocity_cubes) > 0:
            try:
                decoded_cubes = decode_velocity_cubes_batch(model, batch_velocity_cubes, device)

                # Create rows for all successfully decoded cubes
                for i, decoded_cube in enumerate(decoded_cubes):
                    coord_id = batch_coord_ids[i]
                    position_triplets = batch_position_triplets[i]

                    # Create velocity row
                    velocity_row = {'x_y_z': coord_id}
                    for j in range(125):
                        velocity_row[f'vx_{j+1}'] = decoded_cube[j, 0]
                        velocity_row[f'vy_{j+1}'] = decoded_cube[j, 1]
                        velocity_row[f'vz_{j+1}'] = decoded_cube[j, 2]
                    velocity_rows.append(velocity_row)

                    # Create position row
                    position_row = {'x_y_z': coord_id}
                    for j, (px, py, pz) in enumerate(position_triplets):
                        position_row[f'x_{j+1}'] = px
                        position_row[f'y_{j+1}'] = py
                        position_row[f'z_{j+1}'] = pz
                    position_rows.append(position_row)

                    processed_count += 1

            except Exception as e:
                logger.error(f"Error decoding batch starting at index {batch_start}: {e}")
                logger.debug(f"Full traceback:", exc_info=True)
                error_count += len(batch_velocity_cubes)
                continue

        # Log progress every 10 batches
        if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
            logger.debug(f"Progress: {processed_count}/{len(coordinates)} coordinates processed "
                        f"({100*processed_count/len(coordinates):.1f}%), {error_count} errors")

    logger.info(f"Finished processing: {processed_count} successful, {error_count} errors")

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
    import time

    # Immediate startup message with flush
    print(f"[STARTUP] generate_decoded_velocity_analysis.py starting at {time.time()}", flush=True)
    sys.stderr.write(f"[STARTUP] PID {os.getpid()} starting\n")
    sys.stderr.flush()

    parser = argparse.ArgumentParser(description='Decode velocity data for coordinate space analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., "7p2")')
    parser.add_argument('--time', type=int, required=True, help='Time step (e.g., 1000)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for processing (default: 64)')
    args = parser.parse_args()

    logger.debug(f"[CHECKPOINT 1] Args parsed: dataset={args.dataset}, time={args.time}")
    sys.stderr.write(f"[CHECKPOINT 1] Args parsed: dataset={args.dataset}, time={args.time}\n")
    sys.stderr.flush()

    # Setup device - try MPS first (Mac M-series), then CUDA, then CPU
    logger.debug("[CHECKPOINT 2] Starting device detection...")
    sys.stderr.write("[CHECKPOINT 2] Starting device detection...\n")
    sys.stderr.flush()

    device_name = "CPU"
    if torch.backends.mps.is_available():
        logger.debug("[CHECKPOINT 2a] MPS available, testing...")
        sys.stderr.write("[CHECKPOINT 2a] MPS available, testing...\n")
        sys.stderr.flush()
        try:
            device = torch.device("mps")
            # Test MPS with a small tensor
            test_tensor = torch.randn(10).to(device)
            device_name = "MPS (Apple Silicon GPU)"
            logger.debug("[CHECKPOINT 2b] MPS device test passed")
            sys.stderr.write("[CHECKPOINT 2b] MPS device test passed\n")
            sys.stderr.flush()
        except Exception as e:
            logger.warning(f"MPS available but failed test: {e}, falling back to CUDA/CPU")
            sys.stderr.write(f"[CHECKPOINT 2c] MPS test failed: {e}\n")
            sys.stderr.flush()
            device = None
    else:
        device = None
        logger.debug("[CHECKPOINT 2d] MPS not available")
        sys.stderr.write("[CHECKPOINT 2d] MPS not available\n")
        sys.stderr.flush()

    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA (GPU {torch.cuda.get_device_name(0)})"
        logger.debug(f"[CHECKPOINT 2e] Using CUDA device: {torch.cuda.get_device_name(0)}")
        sys.stderr.write(f"[CHECKPOINT 2e] Using CUDA: {torch.cuda.get_device_name(0)}\n")
        sys.stderr.flush()
    elif device is None:
        device = torch.device("cpu")
        device_name = "CPU (No GPU acceleration)"
        logger.debug("[CHECKPOINT 2f] Falling back to CPU")
        sys.stderr.write("[CHECKPOINT 2f] Falling back to CPU\n")
        sys.stderr.flush()

    # Print device in RAINBOW
    rainbow_msg = Colors.rainbow(f"{'='*80}\n🚀 USING DEVICE: {device_name} 🚀\n{'='*80}")
    print(f"\n{rainbow_msg}\n", flush=True)
    logger.info(f"[CHECKPOINT 3] Device selected: {device} ({device_name})")
    sys.stderr.write(f"[CHECKPOINT 3] Device selected: {device} ({device_name})\n")
    sys.stderr.flush()

    # Load model
    logger.debug(f"[CHECKPOINT 4] Loading model from: {MODEL_PATH}")
    sys.stderr.write(f"[CHECKPOINT 4] Loading model from: {MODEL_PATH}\n")
    sys.stderr.flush()

    t_start_model = time.time()
    model = load_model(MODEL_PATH, device)
    t_end_model = time.time()

    logger.debug(f"[CHECKPOINT 5] Model loaded successfully in {t_end_model - t_start_model:.2f}s")
    sys.stderr.write(f"[CHECKPOINT 5] Model loaded in {t_end_model - t_start_model:.2f}s\n")
    sys.stderr.flush()

    # Process the dataset
    logger.debug(f"Starting dataset processing: {args.dataset} at time {args.time} with batch_size={args.batch_size}")
    velocity_df, position_df = process_dataset(
        args.dataset, args.time, model, device, DATA_BASE_DIR, OUTPUT_DIR, batch_size=args.batch_size
    )
    logger.debug("Dataset processing complete")

    if velocity_df is not None and position_df is not None:
        logger.info("Processing completed successfully!")
        return 0
    else:
        logger.error("Processing failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
