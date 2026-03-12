import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    # Check if there are multiple samples with same (Y, Z) AND same Param
    n = 10000
    yz_param = data[:n, 0, 0, [48, 49, 51]]
    unique_yz_param, counts = np.unique(yz_param, axis=0, return_counts=True)
    print(f"Unique (Y, Z, Param) in first {n} samples: {len(unique_yz_param)}")
    print(f"Counts of top 5 (Y, Z, Param): {sorted(counts, reverse=True)[:5]}")
    
    # Let's find indices of one set
    target = unique_yz_param[np.argmax(counts)]
    indices = np.where(np.all(yz_param == target, axis=1))[0]
    print(f"Indices for target {target}: {indices}")
    
    # Check X coordinates for these samples
    for idx in indices:
        x0 = data[idx, 0, 0, 47]
        print(f"Sample {idx}, X0={x0}")
