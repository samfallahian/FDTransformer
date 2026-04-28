import os
import torch
import h5py
import numpy as np
import sys

# Add project root to sys.path to allow imports from other modules
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
sys.path.insert(0, PROJECT_ROOT)

from TransformLatent import FloatConverter

def load_ae():
    ENCODER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
    DEVICE = "cpu" # Using CPU for data preparation
    ae = torch.jit.load(ENCODER_CHECKPOINT, map_location=DEVICE)
    ae.eval()
    return ae

def prepare_encoded_enstrophy():
    h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
    ae = load_ae()
    converter = FloatConverter()
    TRIPLET_IDX = 62
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        n_search = 100000
        p_target = 5.2
        
        params = data[:n_search, 0, 0, 51]
        mask = np.abs(params - p_target) < 0.01
        indices = np.where(mask)[0]
        
        coords = data[indices, 0, 0, 48:50]
        u_yz, u_idx = np.unique(coords, axis=0, return_index=True)
        indices = indices[u_idx]
        
        unique_y = np.sort(np.unique(u_yz[:, 0]))
        unique_z = np.sort(np.unique(u_yz[:, 1]))
        ny, nz = len(unique_y), len(unique_z)
        nx = 26
        
        V_encoded = np.zeros((nx, ny, nz, 3))
        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}
        
        # We need to extract the 8th timestep latents for these samples
        # The latents are in data[idx, 7, :, 0:47]
        for idx in indices:
            y_val = data[idx, 0, 0, 48]
            z_val = data[idx, 0, 0, 49]
            iy, iz = y_map[y_val], z_map[z_val]
            
            latents = data[idx, 7, :, :47] # (26, 47)
            latents_torch = torch.from_numpy(latents).float()
            
            with torch.no_grad():
                decoded = ae.decode(latents_torch) # (26, 375)
                # Reshape to (26, 125, 3) and extract 63rd triplet (TRIPLET_IDX=62)
                v_full = decoded.reshape(26, 125, 3)
                v_63 = v_full[:, TRIPLET_IDX, :].numpy()
                
                # Denormalize
                v_denorm = converter.unconvert(v_63)
                V_encoded[:, iy, iz, :] = v_denorm
        
        # Calculate gradients and enstrophy
        x = data[0, 0, :, 47]
        y = unique_y
        z = unique_z
        
        grad_u = np.gradient(V_encoded[..., 0], x, y, z)
        grad_v = np.gradient(V_encoded[..., 1], x, y, z)
        grad_w = np.gradient(V_encoded[..., 2], x, y, z)
        
        wx = grad_w[1] - grad_v[2] # dw/dy - dv/dz
        wy = grad_u[2] - grad_w[0] # du/dz - dw/dx
        wz = grad_v[0] - grad_u[1] # dv/dx - du/dy
        
        enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
        
        np.savez("pySINDy/encoded_data_grad.npz", V=V_encoded, wx=wx, wy=wy, wz=wz, enstrophy=enstrophy)
        print("Saved pySINDy/encoded_data_grad.npz")

if __name__ == "__main__":
    prepare_encoded_enstrophy()
