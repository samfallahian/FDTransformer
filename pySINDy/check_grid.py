import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Check Y/Z grid coverage for one Reynolds parameter.", runtime=False)
    parser.add_argument("--n", type=int, default=10000, help="Number of samples to inspect.")
    parser.add_argument("--p-val", type=float, default=5.2, help="Parameter/Reynolds value to filter.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        params = data[:args.n, 0, 0, 51]
        idx_param = np.where(np.abs(params - args.p_val) < 0.01)[0]
        print(f"Found {len(idx_param)} samples with Param {args.p_val} in first {args.n}")

        yz_param = data[idx_param, 0, 0, 48:50]
        unique_yz = np.unique(yz_param, axis=0)
        print(f"Unique (Y, Z) for Param {args.p_val}: {len(unique_yz)}")
        print(f"Sample unique Y: {np.unique(unique_yz[:, 0])}")
        print(f"Sample unique Z: {np.unique(unique_yz[:, 1])}")


if __name__ == "__main__":
    main()
