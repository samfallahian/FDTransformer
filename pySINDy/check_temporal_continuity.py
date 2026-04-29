import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Check repeated Y/Z/parameter samples in the evaluation HDF5 file.", runtime=False)
    parser.add_argument("--n", type=int, default=10000, help="Number of samples to inspect.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        yz_param = data[:args.n, 0, 0, [48, 49, 51]]
        unique_yz_param, counts = np.unique(yz_param, axis=0, return_counts=True)
        print(f"Unique (Y, Z, Param) in first {args.n} samples: {len(unique_yz_param)}")
        print(f"Counts of top 5 (Y, Z, Param): {sorted(counts, reverse=True)[:5]}")

        target = unique_yz_param[np.argmax(counts)]
        indices = np.where(np.all(yz_param == target, axis=1))[0]
        print(f"Indices for target {target}: {indices}")

        for idx in indices:
            x0 = data[idx, 0, 0, 47]
            print(f"Sample {idx}, X0={x0}")


if __name__ == "__main__":
    main()
