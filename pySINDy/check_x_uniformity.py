import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Check whether X coordinates are uniform across samples.", runtime=False)
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to inspect.")
    parser.add_argument("--index", type=int, default=0, help="First sample index to print.")
    parser.add_argument("--compare-index", type=int, default=1, help="Second sample index to print.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        x_coords = data[args.index, 0, :, 47]
        print(f"Sample {args.index} X coords: {x_coords}")

        x_coords_compare = data[args.compare_index, 0, :, 47]
        print(f"Sample {args.compare_index} X coords: {x_coords_compare}")

        all_x = data[:args.n, 0, :, 47]
        print(f"All X same in first {args.n}: {np.all(all_x == x_coords)}")


if __name__ == "__main__":
    main()
