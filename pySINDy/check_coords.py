import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Inspect coordinate patterns in the evaluation HDF5 file.", runtime=False)
    parser.add_argument("--n", type=int, default=5000, help="Number of samples to inspect.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        yz = data[:args.n, 0, 0, 48:50]
        unique_yz, counts = np.unique(yz, axis=0, return_counts=True)
        print(f"Unique (Y, Z) in first {args.n} samples: {len(unique_yz)}")
        print(f"Counts of top 5 (Y, Z): {sorted(counts, reverse=True)[:5]}")

        x_coords = data[0, 0, :, 47]
        print(f"X coords: {x_coords}")
        print(f"X diffs: {np.diff(x_coords)}")


if __name__ == "__main__":
    main()
