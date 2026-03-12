import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    idx = [677, 1816, 2689, 4643, 6128, 7127, 9650, 9653]
    for i in idx:
        print(f"Sample {i}, first latent token (0-5):\n", data[i, 0, 0, :6])
