import h5py
import numpy as np

from pysindy_config import load_config_from_args, make_parser, resolve_path


def main():
    parser = make_parser("Print X coordinates across timesteps for one sample.", runtime=False)
    parser.add_argument("--index", type=int, default=677, help="Sample index to inspect.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)

    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        print(f"Sample {args.index}, X-coords at t=0 to t=7:")
        for t in range(8):
            print(f"t={t}: {data[args.index, t, 0, 47]}")


if __name__ == "__main__":
    main()
