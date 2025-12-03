#!/usr/bin/env python3
"""
validation_WAE_01_overlapping_encoding_detail.py - Benchmark overlapping encodings.

Creates a dataframe with coordinates and their neighbors for model comparison.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np

# Resolve project directories so we can import local modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Change to parent directory so experiment.preferences can be found
os.chdir(PARENT_DIR)

from CoordinateSpace import find_neighbors, CoordinateSpace, givenXYZreplyVelocityCube  # noqa: E402

# REQUIRED INPUT VALUES - UPDATE THESE WITH YOUR ACTUAL DATA
PICKLE_FILENAME = "100.pkl"
DATASET_NAME = "3p6"  # Dataset name (e.g., "3p6", "7p2", etc.) - used for metadata lookup and output path
CENTER_X = -113
CENTER_Y = 35
CENTER_Z = 3
PICKLE_PATH = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training"
OUTPUT_DIR = "/Users/kkreth/PycharmProjects/data/overlap_analysis"

# Auto-extract TIME from PICKLE_FILENAME (e.g., "1000.pkl" -> TIME=1000)
TIME = int(PICKLE_FILENAME.replace('.pkl', '').replace('.gz', ''))

# DEBUG mode: if True, only process single coordinate; if False, process all coordinates
DEBUG = False
MIN_EXPECTED_ROWS = 20000  # Minimum expected rows when not in DEBUG mode

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_column_names():
    """Generate column names for all neighboring coordinates from -2 to +2."""
    columns = ['x_y_z']  # First column for the center coordinate

    # Generate all combinations of offsets from -2 to +2
    for x_offset in range(-2, 3):  # -2, -1, 0, 1, 2
        for y_offset in range(-2, 3):
            for z_offset in range(-2, 3):
                # Skip the center point (0, 0, 0)
                if x_offset == 0 and y_offset == 0 and z_offset == 0:
                    continue

                # Format: x_minus_2__y_minus_1__z_0 or x_plus_1__y_0__z_plus_2
                x_str = f"x_minus_{abs(x_offset)}" if x_offset < 0 else f"x_plus_{x_offset}" if x_offset > 0 else "x_0"
                y_str = f"y_minus_{abs(y_offset)}" if y_offset < 0 else f"y_plus_{y_offset}" if y_offset > 0 else "y_0"
                z_str = f"z_minus_{abs(z_offset)}" if z_offset < 0 else f"z_plus_{z_offset}" if z_offset > 0 else "z_0"

                col_name = f"{x_str}__{y_str}__{z_str}"
                columns.append(col_name)

    return columns


def main():
    """Create dataframe with coordinate space mapping."""
    print(f"Processing pickle file: {PICKLE_FILENAME}")
    print(f"DEBUG mode: {DEBUG}")

    if DEBUG:
        print(f"Center coordinates: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")

        # Create the coordinate generator
        # Note: Use DATASET_NAME.pkl for metadata lookup
        coordinator = givenXYZreplyVelocityCube(
            pickle_filename=f"{DATASET_NAME}.pkl",
            x=CENTER_X,
            y=CENTER_Y,
            z=CENTER_Z
        )

        # Get the coordinate triplets
        triplets = coordinator.locateNeighbors()

        # Print the results
        print(f"Generated {len(triplets)} coordinate triplets")
        print("\nFirst 5 triplets:")
        for i, triplet in enumerate(triplets[:5]):
            print(f"  {i + 1}. ({triplet[0]}, {triplet[1]}, {triplet[2]})")
    else:
        # Process all coordinates from the dataset
        print("Processing ALL coordinates from dataset...")

        # Create initial coordinator to get metadata
        coordinator = givenXYZreplyVelocityCube(
            pickle_filename=f"{DATASET_NAME}.pkl",
            x=CENTER_X,
            y=CENTER_Y,
            z=CENTER_Z
        )

        # Get all unique coordinates from enumerated lists
        all_triplets = []
        for x in coordinator.x_enumerated:
            for y in coordinator.y_enumerated:
                for z in coordinator.z_enumerated:
                    all_triplets.append((x, y, z))

        triplets = all_triplets
        print(f"Generated {len(triplets)} coordinate triplets from full dataset")

    # Create column names
    columns = create_column_names()
    print(f"\nCreated {len(columns)} columns (1 center + {len(columns)-1} neighbors)")

    # Initialize dataframe with NaN values
    df_compounding_comparison_coordinate_space_map = pd.DataFrame(
        index=range(len(triplets)),
        columns=columns,
        dtype=object
    )

    # Populate the dataframe
    print("\nPopulating dataframe...")
    for idx, center_triplet in enumerate(triplets):
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(triplets)}...")

        # Store the center coordinate as a tuple string
        df_compounding_comparison_coordinate_space_map.at[idx, 'x_y_z'] = \
            f"{center_triplet[0]}_{center_triplet[1]}_{center_triplet[2]}"

        # Calculate and store all neighbor coordinates from -2 to +2
        for x_offset in range(-2, 3):  # -2, -1, 0, 1, 2
            for y_offset in range(-2, 3):
                for z_offset in range(-2, 3):
                    if x_offset == 0 and y_offset == 0 and z_offset == 0:
                        continue

                    # Calculate neighbor coordinate
                    neighbor_x = center_triplet[0] + x_offset
                    neighbor_y = center_triplet[1] + y_offset
                    neighbor_z = center_triplet[2] + z_offset

                    # Generate column name
                    x_str = f"x_minus_{abs(x_offset)}" if x_offset < 0 else f"x_plus_{x_offset}" if x_offset > 0 else "x_0"
                    y_str = f"y_minus_{abs(y_offset)}" if y_offset < 0 else f"y_plus_{y_offset}" if y_offset > 0 else "y_0"
                    z_str = f"z_minus_{abs(z_offset)}" if z_offset < 0 else f"z_plus_{z_offset}" if z_offset > 0 else "z_0"
                    col_name = f"{x_str}__{y_str}__{z_str}"

                    # Store neighbor coordinate
                    df_compounding_comparison_coordinate_space_map.at[idx, col_name] = \
                        f"{neighbor_x}_{neighbor_y}_{neighbor_z}"

    print(f"Dataframe shape: {df_compounding_comparison_coordinate_space_map.shape}")
    print(f"\nFirst few rows of the dataframe:")
    print(df_compounding_comparison_coordinate_space_map.head())

    # Check row count if not in DEBUG mode
    num_rows = len(df_compounding_comparison_coordinate_space_map)
    if not DEBUG and num_rows < MIN_EXPECTED_ROWS:
        # Print warning in rainbow colors
        colors = [
            '\033[91m',  # Red
            '\033[93m',  # Yellow
            '\033[92m',  # Green
            '\033[96m',  # Cyan
            '\033[94m',  # Blue
            '\033[95m',  # Magenta
        ]
        reset = '\033[0m'

        warning_msg = f"WARNING: Only {num_rows} rows generated, expected at least {MIN_EXPECTED_ROWS}!"
        rainbow_warning = ''.join(colors[i % len(colors)] + char for i, char in enumerate(warning_msg)) + reset
        print(f"\n{rainbow_warning}\n")

    # Save to compressed pickle with time suffix in dataset subdirectory
    output_filename = f"df_compounding_comparison_coordinate_space_map_{TIME:04d}.pkl.gz"
    dataset_output_dir = os.path.join(OUTPUT_DIR, DATASET_NAME)
    output_path = os.path.join(dataset_output_dir, output_filename)

    # Create output directory if it doesn't exist
    os.makedirs(dataset_output_dir, exist_ok=True)

    print(f"\nSaving to {output_path}...")
    df_compounding_comparison_coordinate_space_map.to_pickle(
        output_path,
        compression='gzip'
    )
    print(f"Successfully saved dataframe to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
