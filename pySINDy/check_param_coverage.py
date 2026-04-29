import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Check Y/Z/parameter coverage in the evaluation HDF5 file.", runtime=False)
    parser.add_argument("--n", type=int, default=50000, help="Number of samples to inspect.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        yz_param = data[:args.n, 0, 0, [48, 49, 51]]
        unique_yz_param, counts = np.unique(yz_param, axis=0, return_counts=True)
        print(f"Total unique (Y, Z, Param): {len(unique_yz_param)}")

        unique_params = np.unique(unique_yz_param[:, 2])
        print(f"Unique Params: {unique_params}")

        for p in unique_params:
            p_mask = np.abs(unique_yz_param[:, 2] - p) < 0.01
            num_yz = np.sum(p_mask)
            print(f"Param {p:.1f}: {num_yz} unique (Y, Z) grid points found in first {args.n} samples")


if __name__ == "__main__":
    main()
