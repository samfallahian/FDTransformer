import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Inspect the originals dataset in the evaluation HDF5 file.", runtime=False)
    parser.add_argument("--n", type=int, default=1000, help="Number of original samples to load.")
    parser.add_argument("--p-val", type=float, default=5.2, help="Parameter/Reynolds value to filter.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        orig = f['originals'][:args.n]
        print("Originals shape:", orig.shape)

        params = data[:args.n, 0, 0, 51]
        idx_param = np.where(np.abs(params - args.p_val) < 0.01)[0]
        for i in idx_param[:5]:
            print(f"Sample {i}, Y, Z: {data[i, 0, 0, 48:50]}, Originals[0]: {orig[i, 0]}")


if __name__ == "__main__":
    main()
