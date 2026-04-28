import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    n = 50000
    p_val = 5.2
    params = data[:n, 0, 0, 51]
    p_mask = np.abs(params - p_val) < 0.01
    
    # Coordinates of these samples
    coords = data[:n, 0, 0, 48:50][p_mask]
    u_yz = np.unique(coords, axis=0)
    print(f"Unique (Y, Z) for Param {p_val}: {len(u_yz)}")
    
    # Sort Y and Z
    unique_y = np.sort(np.unique(u_yz[:, 0]))
    unique_z = np.sort(np.unique(u_yz[:, 1]))
    print(f"Unique Y ({len(unique_y)}): {unique_y}")
    print(f"Unique Z ({len(unique_z)}): {unique_z}")
    
    # Check if Y and Z gaps are consistent
    print(f"Y diffs: {np.unique(np.diff(unique_y))}")
    print(f"Z diffs: {np.unique(np.diff(unique_z))}")
