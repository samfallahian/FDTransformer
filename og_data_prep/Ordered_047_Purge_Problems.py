"""
Ordered_047_Purge_Problems.py

This script identifies and removes rows from the cube centroid mapping CSV 
that contain specific missing coordinates found in the OG simulation data.

Problematic Coordinates:
- (85, -56, -17)
- (14, 75, 11)
- (93, 23, 30)

Action:
1. Backup original file to df_all_possible_combinations_with_neighbors_ForNonOGData.csv
2. Remove all rows containing any of the problematic coordinates in any coordinate column.
3. Overwrite the original mapping file with the cleaned data.
"""

import os
import pandas as pd
import shutil

def main():
    # Paths
    base_dir = "/Users/kkreth/PycharmProjects/cgan/cube_centroid_mapping"
    mapping_file = os.path.join(base_dir, "df_all_possible_combinations_with_neighbors.csv")
    backup_file = os.path.join(base_dir, "df_all_possible_combinations_with_neighbors_ForNonOGData.csv")
    
    missing_coords = {(85, -56, -17), (14, 75, 11), (93, 23, 30)}
    
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found at {mapping_file}")
        return

    print(f"Loading mapping from {mapping_file}...")
    df = pd.read_csv(mapping_file)
    original_row_count = len(df)
    print(f"Original row count: {original_row_count}")

    # Backup the original file
    print(f"Creating backup at {backup_file}...")
    shutil.copy2(mapping_file, backup_file)

    # Identify coordinate column triplets
    x_cols = [c for c in df.columns if c.endswith('_x')]
    
    # Create a mask for rows to KEEP
    # Start with all True
    keep_mask = pd.Series(True, index=df.index)
    
    print("Identifying problematic rows...")
    for x_col in x_cols:
        base = x_col[:-2]
        y_col = base + '_y'
        z_col = base + '_z'
        
        # Check if this triplet matches any missing coord
        for coord in missing_coords:
            # Mask for rows that CONTAIN the problematic coord
            has_problem = (df[x_col] == coord[0]) & (df[y_col] == coord[1]) & (df[z_col] == coord[2])
            # Update keep_mask to False where we found a problem
            keep_mask &= ~has_problem

    df_cleaned = df[keep_mask].copy()
    new_row_count = len(df_cleaned)
    rows_removed = original_row_count - new_row_count

    print(f"Rows removed: {rows_removed}")
    print(f"New row count: {new_row_count}")

    if rows_removed != 200:
        print(f"WARNING: Expected to remove 200 rows, but removed {rows_removed} instead.")
    
    # Save the cleaned file
    print(f"Overwriting original file with cleaned data...")
    df_cleaned.to_csv(mapping_file, index=False)
    print("Successfully purged problems.")

if __name__ == "__main__":
    main()
