import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysindy_config import load_config_from_args, make_parser, resolve_path


def investigate_one_experiment(param_val, h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['data']
        # We need to find all samples with the same param_val and same (y, z) to see if they form a time series
        # But wait, each sample already has 8 timesteps.
        # Let's see if those 8 timesteps have different X or same X.
        
        d0 = data[0]
        print("Sample 0, Param:", d0[0, 0, 51])
        print("Sample 0, Y, Z:", d0[0, 0, 48:50])
        print("Sample 0, X coordinates at t=0:\n", d0[0, :, 47])
        print("Sample 0, X coordinates at t=1:\n", d0[1, :, 47])
        
        # If X is the same across timesteps in one sample, then it's a time series at those X locations.
        
        # Let's find samples that have the same (y, z) and same param.
        # This might be hard to search through 1M samples.
        
        # Let's just look at the first few samples and see their Y, Z.
        for i in range(10):
            print(f"Sample {i}: Param={data[i,0,0,51]:.1f}, Y={data[i,0,0,48]}, Z={data[i,0,0,49]}")

if __name__ == "__main__":
    parser = make_parser("Inspect one evaluation experiment in the HDF5 data.", runtime=False)
    parser.add_argument("--param-val", type=float, default=5.2, help="Parameter/Reynolds value to inspect.")
    args = parser.parse_args()
    config = load_config_from_args(args)
    h5_path = resolve_path(config, ("data", "evaluation_h5"), required=True)
    investigate_one_experiment(args.param_val, h5_path)
