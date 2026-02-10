'''
This script generates a comprehensive mapping of all interior centroids to their surrounding 5x5x5 
coordinate neighborhoods. The output is a CSV file used for unit testing and spatial mapping.

### DECODING INSTRUCTIONS (LLM-READABLE)
The output file 'df_all_possible_combinations_with_neighbors.csv' uses a structured column 
schema to represent a 3D coordinate space.

1. COLUMN GROUPS:
   - Centroid (3 columns): 'centroid_x', 'centroid_y', 'centroid_z'. 
     The primary coordinate being mapped.
   - Neighbors (375 columns): 125 unique neighbors, each with 3 coordinate components (x, y, z).

2. NEIGHBOR NAMING CONVENTION:
   Format: nbr_dx_{DX}_dy_{DY}_dz_{DZ}_{COORD}
   - DX, DY, DZ: The relative offset from the centroid in 'position units'.
   - Offset Key:
     - m2 = -2 units
     - m1 = -1 unit
     - 0  = 0 units
     - p1 = +1 unit
     - p2 = +2 units
   - COORD: The specific axis ('x', 'y', or 'z').

   Example: 'nbr_dx_m2_dy_0_dz_p1_x' is the X-coordinate of the neighbor located 
   2 units back on X, 0 units on Y, and 1 unit forward on Z.

3. INTERNAL CONSISTENCY:
   - The centroid is restated within the neighbor grid at offset 0,0,0.
   - Therefore:
     - nbr_dx_0_dy_0_dz_0_x == centroid_x
     - nbr_dx_0_dy_0_dz_0_y == centroid_y
     - nbr_dx_0_dy_0_dz_0_z == centroid_z

4. DATA ORDERING:
   - Rows: Sorted by centroid_x, then centroid_y, then centroid_z (ascending).
   - Columns: Centroid integers appear first, followed by neighbors in lexicographical offset order.
'''
import os
import json
import logging
from itertools import product
from typing import Dict, List, Tuple, Any

import pandas as pd

from Ordered_001_Initialize import HostPreferences
from CoordinateSpace import CoordinateSpace


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load Experiment metadata JSON.
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def ensure_int_sorted(values: List[Any]) -> List[int]:
    return sorted(int(v) for v in values)


def trim_edges(values: List[int], trim_each_side: int = 2) -> List[int]:
    """
    Drop first `trim_each_side` and last `trim_each_side` values from a sorted list.
    If the list is too short, return an empty list.
    """
    if len(values) <= trim_each_side * 2:
        return []
    return values[trim_each_side: len(values) - trim_each_side]


def collect_enumerations(meta: Dict[str, Any]) -> List[Tuple[str, List[int], List[int], List[int]]]:
    """
    Collect (source_key, x_enum, y_enum, z_enum) for all entries that contain enumerations.
    """
    collected = []
    for key, entry in meta.items():
        x_enum = entry.get('x_enumerated')
        y_enum = entry.get('y_enumerated')
        z_enum = entry.get('z_enumerated')
        if x_enum is None or y_enum is None or z_enum is None:
            continue
        collected.append((key, ensure_int_sorted(x_enum), ensure_int_sorted(y_enum), ensure_int_sorted(z_enum)))
    return collected


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


def build_dataframe(meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a DataFrame with all interior permutations and their neighbor cubes.

    Columns:
      - centroid_x, centroid_y, centroid_z: integer columns
      - nbr_dx_{dx}_dy_{dy}_dz_{dz}_{x|y|z}: integer columns for each of the 125 neighbors
    """
    rows = []

    # Process only the first dataset entry to avoid redundant identical rows
    all_enums = collect_enumerations(meta)
    if not all_enums:
        logger.warning("No valid enumerations found in metadata.")
        return pd.DataFrame()

    source, x_enum_raw, y_enum_raw, z_enum_raw = all_enums[0]
    logger.info(f"Processing single representative source: '{source}'")

    x_enum = trim_edges(x_enum_raw, 2)
    y_enum = trim_edges(y_enum_raw, 2)
    z_enum = trim_edges(z_enum_raw, 2)

    if not x_enum or not y_enum or not z_enum:
        logger.info(f"Skipping '{source}' due to insufficient interior points.")
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


def default_output_path() -> str:
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, 'df_all_possible_combinations_with_neighbors.csv')


def main():
    # Discover metadata path via HostPreferences; fall back to standard configs path
    prefs = HostPreferences()
    metadata_path = prefs.metadata_location
    if not metadata_path or not os.path.exists(metadata_path):
        # Fallback
        candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'Experiment_MetaData.json')
        if os.path.exists(candidate):
            metadata_path = candidate
        else:
            raise FileNotFoundError("Could not locate Experiment_MetaData.json via HostPreferences or fallback path.")

    logger.info(f"Loading metadata from: {metadata_path}")
    meta = load_metadata(metadata_path)

    df = build_dataframe(meta)

    out_csv = default_output_path()
    logger.info(f"Writing output CSV to: {out_csv}")
    df.to_csv(out_csv, index=False)

    # Quick summary
    logger.info(f"Rows written: {len(df)}; Columns: {list(df.columns)[:10]} ... total={len(df.columns)}")


if __name__ == '__main__':
    main()
