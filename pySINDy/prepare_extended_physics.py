import os
import torch
import h5py
import numpy as np
import sys
import scipy.ndimage as ndimage

# Add project root to sys.path to allow imports from other modules
PROJECT_ROOT = "/Users/kkreth/PycharmProjects/cgan"
sys.path.insert(0, PROJECT_ROOT)

from transformer.transformer_model_v1 import OrderedTransformerV1
from TransformLatent import FloatConverter

def load_models():
    TRANSFORMER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/transformer/best_ordered_transformer_v1.pt"
    ENCODER_CHECKPOINT = "/Users/kkreth/PycharmProjects/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
    DEVICE = "cpu"
    
    # Patch for torch._dynamo compatibility
    try:
        import torch._dynamo.convert_frame
        if not hasattr(torch._dynamo.convert_frame, 'ConvertFrameBox'):
            class Dummy: pass
            torch._dynamo.convert_frame.ConvertFrameBox = Dummy
    except: pass

    # Add transformer directory to sys.path
    TRANS_DIR = os.path.join(PROJECT_ROOT, "transformer")
    if TRANS_DIR not in sys.path:
        sys.path.insert(0, TRANS_DIR)
    
    # Define a dummy Config class in __main__ to satisfy the unpickler
    class Config:
        pass
    import __main__
    __main__.Config = Config

    checkpoint = torch.load(TRANSFORMER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        transformer = checkpoint['model']
        if hasattr(transformer, '_orig_mod'): transformer = transformer._orig_mod
    else:
        from types import SimpleNamespace
        cfg = SimpleNamespace(**checkpoint['config'])
        transformer = OrderedTransformerV1(cfg)
        transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.eval()

    ae = torch.jit.load(ENCODER_CHECKPOINT, map_location=DEVICE)
    ae.eval()
    
    return transformer, ae

def calculate_vorticity(V, x, y, z):
    # V shape: (nx, ny, nz, 3)
    grad_u = np.gradient(V[..., 0], x, y, z)
    grad_v = np.gradient(V[..., 1], x, y, z)
    grad_w = np.gradient(V[..., 2], x, y, z)
    
    # wx = dw/dy - dv/dz, wy = du/dz - dw/dx, wz = dv/dx - du/dy
    wx = grad_w[1] - grad_v[2]
    wy = grad_u[2] - grad_w[0]
    wz = grad_v[0] - grad_u[1]
    return wx, wy, wz

def prepare_extended_physics():
    h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
    transformer, ae = load_models()
    converter = FloatConverter()
    TRIPLET_IDX = 62
    
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        originals = f['originals']
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
        
        V_raw = np.zeros((nx, ny, nz, 3))
        V_encoded = np.zeros((nx, ny, nz, 3))
        V_pred = np.zeros((nx, ny, nz, 3))
        
        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}
        
        print(f"Processing {len(indices)} grid points for extended physics...")
        
        for idx in indices:
            y_val = data[idx, 0, 0, 48]
            z_val = data[idx, 0, 0, 49]
            iy, iz = y_map[y_val], z_map[z_val]
            
            # Raw
            V_raw[:, iy, iz, :] = originals[idx]
            
            # Encoded/Decoded (T=7)
            sample_data = data[idx] # (8, 26, 52)
            sample_flat = sample_data.reshape(-1, 52)
            t8_input_latents = torch.from_numpy(sample_flat[182:208, :47]).float()
            
            with torch.no_grad():
                decoded_ae = ae.decode(t8_input_latents) # (26, 375)
                v_full_ae = decoded_ae.reshape(26, 125, 3)
                v_63_ae = v_full_ae[:, TRIPLET_IDX, :].numpy()
                V_encoded[:, iy, iz, :] = converter.unconvert(v_63_ae)
                
                # Predict 8th timestep
                inputs = torch.from_numpy(sample_flat[:182]).float().unsqueeze(0) # (1, 182, 52)
                current_seq = inputs.clone()
                t8_preds = []
                for step in range(182, 208):
                    step_out = transformer(current_seq)
                    next_latent = step_out[:, -1, :]
                    t8_preds.append(next_latent)
                    if step < 207:
                        new_token = torch.from_numpy(sample_flat[step:step+1]).float().unsqueeze(0)
                        new_token[0, 0, :47] = next_latent
                        current_seq = torch.cat([current_seq, new_token], dim=1)
                
                t8_latents_pred = torch.cat(t8_preds, dim=0) # (26, 47)
                decoded_pred = ae.decode(t8_latents_pred) # (26, 375)
                v_full_pred = decoded_pred.reshape(26, 125, 3)
                v_63_pred = v_full_pred[:, TRIPLET_IDX, :].numpy()
                V_pred[:, iy, iz, :] = converter.unconvert(v_63_pred)

        x, y, z = data[0, 0, :, 47], unique_y, unique_z
        
        for name, V in [("raw", V_raw), ("encoded", V_encoded), ("predicted", V_pred)]:
            u, v, w = V[..., 0], V[..., 1], V[..., 2]
            wx, wy, wz = calculate_vorticity(V, x, y, z)
            
            ke = 0.5 * (u**2 + v**2 + w**2)
            helicity = u*wx + v*wy + w*wz
            enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
            
            np.savez(f"pySINDy/{name}_extended.npz", 
                     V=V, wx=wx, wy=wy, wz=wz, 
                     ke=ke, helicity=helicity, enstrophy=enstrophy)
            print(f"Saved pySINDy/{name}_extended.npz")

if __name__ == "__main__":
    prepare_extended_physics()
