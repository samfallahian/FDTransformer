import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    idx = 677
    print(f"Sample {idx}, X-coords at t=0 to t=7:")
    for t in range(8):
        print(f"t={t}: {data[idx, t, 0, 47]}")
