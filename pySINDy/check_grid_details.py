import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Print detailed grid spacing for one Reynolds parameter.", runtime=False)
    parser.add_argument("--n", type=int, default=50000, help="Number of samples to inspect.")
    parser.add_argument("--p-val", type=float, default=5.2, help="Parameter/Reynolds value to filter.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        params = data[:args.n, 0, 0, 51]
        p_mask = np.abs(params - args.p_val) < 0.01

        coords = data[:args.n, 0, 0, 48:50][p_mask]
        u_yz = np.unique(coords, axis=0)
        print(f"Unique (Y, Z) for Param {args.p_val}: {len(u_yz)}")

        unique_y = np.sort(np.unique(u_yz[:, 0]))
        unique_z = np.sort(np.unique(u_yz[:, 1]))
        print(f"Unique Y ({len(unique_y)}): {unique_y}")
        print(f"Unique Z ({len(unique_z)}): {unique_z}")
        print(f"Y diffs: {np.unique(np.diff(unique_y))}")
        print(f"Z diffs: {np.unique(np.diff(unique_z))}")


if __name__ == "__main__":
    main()
