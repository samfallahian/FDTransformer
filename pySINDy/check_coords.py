import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    # Check first 5000 samples for coordinate pattern
    n = 5000
    yz = data[:n, 0, 0, 48:50]
    unique_yz, counts = np.unique(yz, axis=0, return_counts=True)
    print(f"Unique (Y, Z) in first {n} samples: {len(unique_yz)}")
    print(f"Counts of top 5 (Y, Z): {sorted(counts, reverse=True)[:5]}")
    
    # Check X coordinates
    x_coords = data[0, 0, :, 47]
    print(f"X coords: {x_coords}")
    print(f"X diffs: {np.diff(x_coords)}")
