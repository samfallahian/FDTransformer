import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Print selected latent tokens from the evaluation HDF5 file.", runtime=False)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[677, 1816, 2689, 4643, 6128, 7127, 9650, 9653],
        help="Sample indices to inspect.",
    )
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        for i in args.indices:
            print(f"Sample {i}, first latent token (0-5):\n", data[i, 0, 0, :6])


if __name__ == "__main__":
    main()
