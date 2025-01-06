import pandas as pd
import json
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading dataset...")

INPUT_CSV = "../dataset/3p6-selected-test.csv"
OUTPUT_CSV = "../dataset/3p6-selected-test-processed.csv"
METADATA_JSON = "../configs/MD_3p6.json"

def process_csv(input_csv, output_csv, metadata_json):
    # Load the dataset
    print(f"Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(df.head())

    # Step 1: Drop unused columns
    print("Dropping unused columns...")
    df = df.drop(columns=["px", "py", "pz"], errors='ignore')  # Use 'errors' to avoid issues if columns are missing
    print(df.head())

    # Step 2: Adjust data types
    print("Adjusting data types...")
    if 'x' in df.columns: df['x'] = df['x'].astype(int)
    if 'y' in df.columns: df['y'] = df['y'].astype(int)
    if 'z' in df.columns: df['z'] = df['z'].astype(int)
    if 'time' in df.columns: df['time'] = df['time'].astype(int)
#TODO also convert the distance to integer
    if 'vx' in df.columns: df['vx'] = df['vx'].astype('float32')
    if 'vy' in df.columns: df['vy'] = df['vy'].astype('float32')
    if 'vz' in df.columns: df['vz'] = df['vz'].astype('float32')
    print(df.head())

    # Step 3: Normalize velocity columns (using PyTorch)
    print("Normalizing velocity columns with PyTorch...")
    if 'vx' in df.columns:
        vx_tensor = torch.tensor(df['vx'].values, dtype=torch.float32)
        df['vx_norm'] = (vx_tensor - vx_tensor.min()) / (vx_tensor.max() - vx_tensor.min())
    if 'vy' in df.columns:
        vy_tensor = torch.tensor(df['vy'].values, dtype=torch.float32)
        df['vy_norm'] = (vy_tensor - vy_tensor.min()) / (vy_tensor.max() - vy_tensor.min())
    if 'vz' in df.columns:
        vz_tensor = torch.tensor(df['vz'].values, dtype=torch.float32)
        df['vz_norm'] = (vz_tensor - vz_tensor.min()) / (vz_tensor.max() - vz_tensor.min())
    print(df.head())

    # Step 4: Generate metadata (using PyTorch tensors)
    print("Generating metadata with PyTorch tensors...")
    metadata = {}
    columns_of_interest = ['x', 'y', 'z', 'time', 'distance']  # Add "distance" if it exists in the dataset
    for col in columns_of_interest:
        if col in df.columns:
            tensor = torch.tensor(df[col].values, dtype=torch.float32 if df[col].dtype == 'float32' else torch.int32)
            sorted_tensor = torch.sort(torch.unique(tensor)).values
            metadata[col] = {
                "min": float(sorted_tensor[0]),
                "max": float(sorted_tensor[-1]),
                "4th_min": float(sorted_tensor[3]) if sorted_tensor.numel() > 3 else None,
                "4th_max": float(sorted_tensor[-4]) if sorted_tensor.numel() > 3 else None,
                "enumerated": sorted_tensor.tolist()
            }
    print(df.head())

    # Step 5: Save metadata to JSON
    print(f"Saving metadata to {metadata_json}...")
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Step 6: Save the updated dataset
    print(f"Saving processed dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)

process_csv(INPUT_CSV, OUTPUT_CSV, METADATA_JSON)
