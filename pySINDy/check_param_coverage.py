import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    # Check first 50,000 samples for unique Y and Z
    n = 50000
    yz_param = data[:n, 0, 0, [48, 49, 51]]
    unique_yz_param, counts = np.unique(yz_param, axis=0, return_counts=True)
    print(f"Total unique (Y, Z, Param): {len(unique_yz_param)}")
    
    # Let's count how many unique (Y, Z) for each Param
    unique_params = np.unique(unique_yz_param[:, 2])
    print(f"Unique Params: {unique_params}")
    
    for p in unique_params:
        p_mask = np.abs(unique_yz_param[:, 2] - p) < 0.01
        num_yz = np.sum(p_mask)
        print(f"Param {p:.1f}: {num_yz} unique (Y, Z) grid points found in first {n} samples")
