import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, output_path, resolve_path


def prepare_raw_enstrophy(config):
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)
    n_search = config["runtime"]["n_search"]
    p_target = config["runtime"]["p_target"]
    out_path = output_path(config, "raw_grad")

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        originals = f['originals']

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

        print(f"Grid: {nx} x {ny} x {nz}")

        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}

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
            dx = np.diff(data[0, 0, :, 47])
            dy = np.diff(unique_y)
            dz = np.diff(unique_z)

            print(f"dx range: {np.min(dx)} to {np.max(dx)}")
            print(f"dy range: {np.min(dy)} to {np.max(dy)}")
            print(f"dz range: {np.min(dz)} to {np.max(dz)}")

            x = data[0, 0, :, 47]
            y = unique_y
            z = unique_z

            grad_u = np.gradient(V[..., 0], x, y, z)
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

            np.savez(out_path, V=V, wx=wx, wy=wy, wz=wz, enstrophy=enstrophy)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = make_parser("Prepare raw velocity gradients and enstrophy for SINDy.")
    args = parser.parse_args()
    prepare_raw_enstrophy(load_config_from_args(args))
