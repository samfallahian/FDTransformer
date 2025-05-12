#!/usr/bin/env python
"""
Ordered_300_GenerateLatentSpace.py

This script:
1) Reads preferences file to get paths
2) Takes --picklefile as a command line argument
3) Extracts time from pickle filename (e.g., "1.pkl" for time 1)
4) Processes each row of data to extract latent space from WAE model
"""

import os
import sys
import pickle
import gzip
import torch
import numpy as np
import pandas as pd
import logging
import click
import re
from pathlib import Path
from tqdm import tqdm

# Add the root directory to the path for import resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the host preferences
from Ordered_001_Initialize import HostPreferences
from encoder.model_WAE_01 import WAE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_pickle_file(file_path):
    """
    Load a pickle file, trying different compression methods.
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        The unpickled data
    """
    logger.info(f"Loading pickle file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try different approaches to load the file
    compression_errors = []
    
    # First, check file signature to detect gzip
    with open(file_path, 'rb') as f:
        header = f.read(2)
        # Check for gzip magic number (0x1f, 0x8b)
        is_likely_gzip = header == b'\x1f\x8b'
    
    # First try with gzip if it looks like gzip
    if is_likely_gzip:
        try:
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            compression_errors.append(f"gzip error: {str(e)}")
    
    # Try without compression
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        compression_errors.append(f"uncompressed error: {str(e)}")
    
    # If the file header matches gzip but normal gzip.open failed, try with a custom approach
    if is_likely_gzip:
        try:
            import zlib
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                # Skip the first 10 bytes (gzip header) and the last 8 bytes (CRC and size)
                decompressed_data = zlib.decompress(compressed_data[10:-8], -zlib.MAX_WBITS)
                return pickle.loads(decompressed_data)
        except Exception as e:
            compression_errors.append(f"manual gzip decompression error: {str(e)}")
    
    # If all approaches failed, raise an error with details
    raise ValueError(f"Failed to load file {file_path} with any method. Errors: {compression_errors}")

def extract_time_from_filename(file_path):
    """
    Extract time value from the pickle filename.
    Example: "1.pkl" -> 1
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        int: The time value
    """
    basename = os.path.basename(file_path)
    time_match = re.match(r'(\d+)\.pkl', basename)
    if time_match:
        return int(time_match.group(1))
    else:
        logger.warning(f"Could not extract time from filename: {basename}")
        return None

def load_wae_model(model_path):
    """
    Load the WAE model from the specified path.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        WAE: The loaded WAE model
    """
    logger.info(f"Loading WAE model from: {model_path}")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                         "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize the model
    model = WAE().to(device)
    
    # Load the model weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    return model, device

def extract_velocity_data(row):
    """
    Extract velocity data from a single row into a normalized array.
    
    Args:
        row (pd.Series): Series containing velocity data
        
    Returns:
        np.ndarray: Array of 375 normalized velocity values
    """
    # Extract velocity columns (vx_1 to vz_125)
    velocity_cols = []
    for i in range(1, 126):
        velocity_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])
    
    # Check if all columns exist
    missing_cols = [col for col in velocity_cols if col not in row.index]
    if missing_cols:
        logger.error(f"Missing velocity columns: {missing_cols[:5]}... (and {len(missing_cols)-5} more)")
        raise ValueError(f"Row is missing required velocity columns")
    
    # Extract velocity data into numpy array
    raw_velocities = row[velocity_cols].to_numpy()
    
    # Ensure we have 375 velocity values (125 points × 3 values per point)
    if raw_velocities.shape[0] != 375:
        raise ValueError(f"Expected 375 velocity values, got {raw_velocities.shape[0]}")
    
    return raw_velocities.astype(np.float32)

def compute_latent_space(model, device, velocity_data):
    """
    Compute the latent space representation using the WAE model.
    
    Args:
        model (WAE): The WAE model
        device (torch.device): The device to use for computation
        velocity_data (np.ndarray): Array of 375 velocity values
        
    Returns:
        np.ndarray: The 47-dimensional latent space representation
    """
    # Convert to tensor and move to device
    velocity_tensor = torch.tensor(velocity_data, dtype=torch.float32).to(device)
    
    # Ensure input has correct shape
    if len(velocity_tensor.shape) == 1:
        velocity_tensor = velocity_tensor.unsqueeze(0)  # Add batch dimension
    
    # Compute latent representation
    with torch.no_grad():
        latent_representation = model.encode(velocity_tensor)
        
    # Move to CPU and convert to numpy
    latent_np = latent_representation.cpu().numpy()
    
    # Remove batch dimension if present
    if len(latent_np.shape) > 1 and latent_np.shape[0] == 1:
        latent_np = latent_np.squeeze(0)
    
    return latent_np

def process_dataframe(df, model, device, time_value=None):
    """
    Process each row in the dataframe to compute latent space.
    
    Args:
        df (pd.DataFrame): DataFrame containing velocity data
        model (WAE): The WAE model
        device (torch.device): The device to use for computation
        time_value (int, optional): Time value to set in each row
        
    Returns:
        pd.DataFrame: Updated DataFrame with latent features
    """
    logger.info(f"Processing DataFrame with {len(df)} rows")
    
    # Add time column if not present
    if 'time' not in df.columns and time_value is not None:
        df['time'] = time_value
    
    # Initialize columns for latent features
    for i in range(1, 48):
        df[f'latent_{i}'] = 0.0
    
    # Process each row individually with tqdm progress bar
    for idx in tqdm(df.index, desc="Processing rows", unit="row", total=len(df)):
        try:
            # Get the row
            row = df.loc[idx]
            
            # Extract velocity data for this row
            velocity_data = extract_velocity_data(row)
            
            # Compute latent space for this row
            latent_features = compute_latent_space(model, device, velocity_data)
            
            # Add latent features to the dataframe
            for i, val in enumerate(latent_features, 1):
                df.at[idx, f'latent_{i}'] = val
                
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            # Continue with next row
    
    return df

@click.command()
@click.option('--picklefile', required=True, help='Path to the pickle file to process')
def main(picklefile):
    """
    Main function to process a pickle file and extract latent space.
    """
    try:
        # 1. Load preferences
        prefs = HostPreferences()
        logger.info(f"Loaded preferences for host: {prefs.hostname}")
        
        # 2. Get time from filename
        time_value = extract_time_from_filename(picklefile)
        logger.info(f"Extracted time value: {time_value}")
        
        # 3. Load the pickle file
        data = load_pickle_file(picklefile)
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index')
            # If the index contains x,y,z coordinates, split them into columns
            if isinstance(df.index[0], tuple) and len(df.index[0]) == 3:
                df.reset_index(inplace=True)
                df.rename(columns={'level_0': 'x', 'level_1': 'y', 'level_2': 'z'}, inplace=True)
        else:
            df = data
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        
        # 4. Load the WAE model
        model_path = os.path.join(current_dir, "encoder/saved_models/WAE_01_epoch_2870.pt")
        wae_model, device = load_wae_model(model_path)
        
        # 5. Process the dataframe
        df = process_dataframe(df, wae_model, device, time_value)
        
        # 6. Save updated DataFrame with gzip compression
        output_path = os.path.splitext(picklefile)[0] + "_with_latent.pkl"
        df.to_pickle(output_path, compression='gzip')  # Added compression parameter
        logger.info(f"Saved compressed DataFrame with latent features to: {output_path}")
        
        # Log row count for verification
        logger.info(f"Processed {len(df)} rows with latent space values")
        
    except Exception as e:
        logger.error(f"Error processing file {picklefile}: {e}")
        raise

if __name__ == "__main__":
    main()