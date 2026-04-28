import h5py
import numpy as np

h5_path = "/Users/kkreth/PycharmProjects/data/transformer_evaluation/evaluation_data.h5"
with h5py.File(h5_path, 'r') as f:
    data = f['data']
    originals = f['originals']
    
    n_search = 100000
    p_target = 5.2
    
    # We'll use a larger chunk to find points
    params = data[:n_search, 0, 0, 51]
    mask = np.abs(params - p_target) < 0.01
    indices = np.where(mask)[0]
    
    coords = data[indices, 0, 0, 48:50] # (N, 2)
    u_yz, u_idx = np.unique(coords, axis=0, return_index=True)
    indices = indices[u_idx]
    
    # Grid info
    unique_y = np.sort(np.unique(u_yz[:, 0]))
    unique_z = np.sort(np.unique(u_yz[:, 1]))
    ny, nz = len(unique_y), len(unique_z)
    nx = 26
    
    print(f"Grid: {nx} x {ny} x {nz}")
    
    # Map coordinates to indices
    y_map = {val: i for i, val in enumerate(unique_y)}
    z_map = {val: i for i, val in enumerate(unique_z)}
    
    # Fill velocity grid
    # Velocities: (nx, ny, nz, 3)
    V = np.zeros((nx, ny, nz, 3))
    found_mask = np.zeros((ny, nz), dtype=bool)
    
    for idx in indices:
        y_val = data[idx, 0, 0, 48]
        z_val = data[idx, 0, 0, 49]
        iy, iz = y_map[y_val], z_map[z_val]
        V[:, iy, iz, :] = originals[idx]
        found_mask[iy, iz] = True
    
    print(f"Found {np.sum(found_mask)} / {ny*nz} YZ points")
    
    if np.all(found_mask):
        print("Grid is complete!")
        # Compute derivatives
        dx = np.diff(data[0,0,:,47])
        dy = np.diff(unique_y)
        dz = np.diff(unique_z)
        
        print(f"dx range: {np.min(dx)} to {np.max(dx)}")
        print(f"dy range: {np.min(dy)} to {np.max(dy)}")
        print(f"dz range: {np.min(dz)} to {np.max(dz)}")
        
        # Finite differences (central)
        # Gradient of u, v, w
        # dudx, dudy, dudz, etc.
        
        # Vorticity w = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
        # Using np.gradient with non-uniform grid is possible
        # but for simplicity let's assume average spacing or use the provided x, y, z arrays
        x = data[0,0,:,47]
        y = unique_y
        z = unique_z
        
        # V has shape (nx, ny, nz, 3)
        # np.gradient(V, x, y, z, axis=(0,1,2))
        
        grad_u = np.gradient(V[..., 0], x, y, z) # list of 3 arrays
        grad_v = np.gradient(V[..., 1], x, y, z)
        grad_w = np.gradient(V[..., 2], x, y, z)
        
        dudy = grad_u[1]
        dudz = grad_u[2]
        dvdx = grad_v[0]
        dvdz = grad_v[2]
        dwdx = grad_w[0]
        dwdy = grad_w[1]
        
        wx = dwdy - dvdz
        wy = dudz - dwdx
        wz = dvdx - dudy
        
        enstrophy = 0.5 * (wx**2 + wy**2 + wz**2)
        print(f"Enstrophy mean: {np.mean(enstrophy)}")
        
        # Save some data for SINDy
        np.savez("pySINDy/raw_data_grad.npz", V=V, wx=wx, wy=wy, wz=wz, enstrophy=enstrophy)
        print("Saved pySINDy/raw_data_grad.npz")

