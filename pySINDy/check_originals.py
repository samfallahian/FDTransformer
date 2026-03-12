import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    # Originals is likely (B, 26, 3), so it's a single time step for 26 points.
    # We need to know which time step it corresponds to.
    # The transformer evaluation script says it extracts predictions for the 8th time step.
    # So "originals" is likely the ground truth for the 8th time step.
    
    orig = f['originals'][:1000]
    print("Originals shape:", orig.shape)
    
    # Let's check if "originals" match the last time step latent info?
    # No, they are decoded velocities. 
    # Let's check first 5 samples for param 5.2 and their Y, Z.
    params = data[:1000, 0, 0, 51]
    idx_52 = np.where(np.abs(params - 5.2) < 0.01)[0]
    for i in idx_52[:5]:
        print(f"Sample {i}, Y, Z: {data[i, 0, 0, 48:50]}, Originals[0]: {orig[i, 0]}")
