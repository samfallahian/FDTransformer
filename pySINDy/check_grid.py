import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    # Let's find some samples with different Y, same Param, same X.
    # We saw Sample 0: Param=5.2, Y=51, Z=-21
    # Sample 4: Param=5.2, Y=19, Z=-9
    # Let's see if we can find more for Param=5.2
    data = f['data']
    n = 10000
    params = data[:n, 0, 0, 51]
    idx_52 = np.where(np.abs(params - 5.2) < 0.01)[0]
    print(f"Found {len(idx_52)} samples with Param 5.2 in first {n}")
    
    yz_52 = data[idx_52, 0, 0, 48:50]
    unique_yz = np.unique(yz_52, axis=0)
    print(f"Unique (Y, Z) for Param 5.2: {len(unique_yz)}")
    
    # Are Y, Z on a grid?
    print(f"Sample unique Y: {np.unique(unique_yz[:, 0])}")
    print(f"Sample unique Z: {np.unique(unique_yz[:, 1])}")
