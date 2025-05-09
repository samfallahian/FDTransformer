#!/usr/bin/env python3
"""
test_coordinate_space.py - Test the CoordinateSpace class with real data.

CONFIGURATION - UPDATE THESE VALUES:
PICKLE_FILENAME = "3p6.pkl"  # Pickle file to test with
CENTER_X = -113              # X coordinate of the center point
CENTER_Y = 35                # Y coordinate of the center point
CENTER_Z = 3                 # Z coordinate of the center point
PICKLE_PATH = "/path/to/your/data/directory"  # Path to the directory containing pickle files
"""

import sys
import os
import logging
import pandas as pd
from CoordinateSpace import find_neighbors, CoordinateSpace, givenXYZreplyVelocityCube

# REQUIRED INPUT VALUES - UPDATE THESE WITH YOUR ACTUAL DATA
PICKLE_FILENAME = "3p6.pkl"
CENTER_X = -113
CENTER_Y = 35
CENTER_Z = 3
PICKLE_PATH = "/Users/kkreth/PycharmProjects/data/all_data_cleaned_dtype_correct"

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test the CoordinateSpace class with real data."""
    print(f"Testing with pickle file: {PICKLE_FILENAME}")
    print(f"Center coordinates: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")

    # Create the coordinate generator with real data
    coordinator = givenXYZreplyVelocityCube(
        pickle_filename=PICKLE_FILENAME,
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

    print("\nLast 5 triplets:")
    for i, triplet in enumerate(triplets[-5:]):
        print(f"  {i + len(triplets) - 4}. ({triplet[0]}, {triplet[1]}, {triplet[2]})")

    # Print coordinate range
    x_coords = [t[0] for t in triplets]
    y_coords = [t[1] for t in triplets]
    z_coords = [t[2] for t in triplets]

    print("\nCoordinate ranges:")
    print(f"  X: {min(x_coords)} to {max(x_coords)} ({len(set(x_coords))} unique values)")
    print(f"  Y: {min(y_coords)} to {max(y_coords)} ({len(set(y_coords))} unique values)")
    print(f"  Z: {min(z_coords)} to {max(z_coords)} ({len(set(z_coords))} unique values)")

    print("\nTesting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())