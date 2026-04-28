import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    idx = 0
    # X coords are stored in feature 47
    x_coords = data[idx, 0, :, 47]
    print(f"Sample 0 X coords: {x_coords}")
    
    # Check sample 1 X coords
    idx1 = 1
    x_coords1 = data[idx1, 0, :, 47]
    print(f"Sample 1 X coords: {x_coords1}")
    
    # Are they the same for all samples?
    n = 1000
    all_x = data[:n, 0, :, 47]
    print(f"All X same in first {n}: {np.all(all_x == x_coords)}")
