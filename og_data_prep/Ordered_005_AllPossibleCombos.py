'''
Ordered_005_AllPossibleCombos.py

This script generates a comprehensive mapping of all interior centroids to their surrounding 5x5x5
coordinate neighborhoods based on the ACTUAL grid found in the Scaled_OG_Data files.

The output is a CSV file used for unit testing and spatial mapping, overwriting 
the existing one that used the old metadata grid.
'''
import os
import sys
import logging
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Any

# Ensure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ordered_001_Initialize import HostPreferences
from CoordinateSpace import CoordinateSpace

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def ensure_int_sorted(values: List[Any]) -> List[int]:
    return sorted(int(v) for v in values)

def trim_edges(values: List[int], trim_each_side: int = 3) -> List[int]:
    """
    Drop first `trim_each_side` and last `trim_each_side` values from a sorted list.
    If the list is too short, return an empty list.
    """
    if len(values) <= trim_each_side * 2:
        return []
    return values[trim_each_side: len(values) - trim_each_side]

def neighbors_for_point(x: int, y: int, z: int,
                        x_enum: List[int], y_enum: List[int], z_enum: List[int]) -> List[Tuple[int, int, int]]:
    """
    Use CoordinateSpace to generate the 5x5x5 neighbors (125 triplets) around a centroid.
    """
    cube = CoordinateSpace.create_from_point(x, y, z, x_enum, y_enum, z_enum)
    combos = cube.combinations
    if len(combos) != 125:
        logger.warning(f"Neighbors returned {len(combos)} instead of 125 for centroid ({x},{y},{z}).")
    # Ensure deterministic ordering: x, then y, then z ascending
    combos = sorted(combos, key=lambda t: (t[0], t[1], t[2]))
    return combos

def build_dataframe(x_enum_raw: List[int], y_enum_raw: List[int], z_enum_raw: List[int]) -> pd.DataFrame:
    """
    Build a DataFrame with all interior permutations and their neighbor cubes.

    Columns:
      - centroid_x, centroid_y, centroid_z: integer columns
      - nbr_dx_{dx}_dy_{dy}_dz_{dz}_{x|y|z}: integer columns for each of the 125 neighbors
    """
    rows = []

    x_enum = trim_edges(x_enum_raw, 3)
    y_enum = trim_edges(y_enum_raw, 3)
    z_enum = trim_edges(z_enum_raw, 3)

    if not x_enum or not y_enum or not z_enum:
        logger.warning("Insufficient interior points to generate centroids.")
        return pd.DataFrame()

    # Pre-generate column names for 125 neighbors based on offsets
    offsets = [-2, -1, 0, 1, 2]
    off_lab = {-2: 'm2', -1: 'm1', 0: '0', 1: 'p1', 2: 'p2'}
    neighbor_cols = []
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                prefix = f"nbr_dx_{off_lab[dx]}_dy_{off_lab[dy]}_dz_{off_lab[dz]}"
                neighbor_cols.extend([f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])

    # Build all centroid permutations (Cartesian product) and order by x,y,z
    centroids = sorted(product(x_enum, y_enum, z_enum), key=lambda t: (t[0], t[1], t[2]))
    logger.info(f"Interior centroids count = {len(centroids)}")

    for cx, cy, cz in centroids:
        nbrs = neighbors_for_point(cx, cy, cz, x_enum_raw, y_enum_raw, z_enum_raw)

        if len(nbrs) != 125:
            # Handle unexpected size
            continue

        row = {
            'centroid_x': int(cx),
            'centroid_y': int(cy),
            'centroid_z': int(cz),
        }

        # Flatten neighbor triplets into the row dict
        # neighbors_for_point returns sorted by x, then y, then z, which matches our column loop order
        for i, (nx, ny, nz) in enumerate(nbrs):
            prefix = neighbor_cols[i*3].rsplit('_', 1)[0]
            row[f"{prefix}_x"] = int(nx)
            row[f"{prefix}_y"] = int(ny)
            row[f"{prefix}_z"] = int(nz)

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        # Reorder columns: centroid first, then neighbors
        ordered_cols = ['centroid_x', 'centroid_y', 'centroid_z'] + neighbor_cols
        df = df[ordered_cols]

    return df

def get_grid_from_sim_data(input_dir: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Read the first .pkl.gz file in the input directory and extract unique X, Y, Z coordinates.
    """
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.pkl.gz') and not f.startswith('.')])
    if not files:
        raise FileNotFoundError(f"No .pkl.gz files found in {input_dir}")
    
    file_path = os.path.join(input_dir, files[0])
    logger.info(f"Extracting grid from {file_path}...")
    
    df = pd.read_pickle(file_path, compression='gzip')
    # Use time step 1 as representative (the grid should be static)
    df_t1 = df[df['time'] == 1]
    
    x_enum = ensure_int_sorted(df_t1['x'].unique())
    y_enum = ensure_int_sorted(df_t1['y'].unique())
    z_enum = ensure_int_sorted(df_t1['z'].unique())
    
    return x_enum, y_enum, z_enum

def main():
    input_dir = "/Users/kkreth/PycharmProjects/data/Scaled_OG_Data"
    output_csv = "/Users/kkreth/PycharmProjects/cgan/cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv"
    
    try:
        x_enum, y_enum, z_enum = get_grid_from_sim_data(input_dir)
        logger.info(f"Grid dimensions: X={len(x_enum)}, Y={len(y_enum)}, Z={len(z_enum)}")
        
        df = build_dataframe(x_enum, y_enum, z_enum)
        
        if df.empty:
            logger.error("Generated DataFrame is empty!")
            return

        logger.info(f"Writing updated mapping CSV to: {output_csv}")
        df.to_csv(output_csv, index=False)
        logger.info(f"Rows written: {len(df)}; Columns: {len(df.columns)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
