# pySINDy

This folder contains SINDy-based physics recovery scripts for velocity fields produced by the FDTransformer pipeline. The scripts prepare raw, autoencoder-decoded, and transformer-predicted velocity grids, compute vorticity/enstrophy and related quantities, then recover algebraic physics relations with PySINDy.

## Setup

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The scripts expect these Python packages: `numpy`, `pandas`, `h5py`, `torch`, `pysindy`, `scipy`, `scikit-learn`, `matplotlib`, and `tqdm`.

## Configure Paths

No script should need a local path edited in source code. Copy the example config and update paths for your machine:

```bash
cd pySINDy
cp config.example.json config.json
```

Important config keys:

- `data.evaluation_h5`: HDF5 file created by the transformer evaluation pipeline.
- `project_root`: repository root used for importing `TransformLatent` and `transformer`.
- `models.encoder_checkpoint`: TorchScript autoencoder checkpoint.
- `models.transformer_checkpoint`: transformer checkpoint.
- `outputs.output_dir`: where generated `.npz`, `.csv`, and `.png` files in this folder are written.
- `outputs.documentation_dir`: where the longer staircase/interpolation experiment results are written.
- Other `outputs.*` keys: filenames for individual generated artifacts.
- `runtime`: shared parameters such as `p_target`, `n_search`, `triplet_idx`, `device`, and Reynolds/context lists.

Every script accepts `--config`. Most data-preparation scripts also accept direct overrides such as:

```bash
python prepare_raw_enstrophy.py --h5-path /path/to/evaluation_data.h5 --output-dir ./outputs
```

Use `python <script>.py --help` for the exact options.

## Quick Data Checks

These scripts inspect the HDF5 evaluation data before running heavier jobs:

```bash
python check_coords.py --config config.json
python check_grid.py --config config.json
python check_grid_details.py --config config.json
python check_param_coverage.py --config config.json
python check_temporal_continuity.py --config config.json
python check_x_uniformity.py --config config.json
python check_originals.py --config config.json
python check_latent_diff.py --config config.json
python check_one_sample_t.py --config config.json
python investigate_data.py --config config.json
```

## Prepare SINDy Inputs

Run these to generate the gradient/enstrophy `.npz` inputs:

```bash
python prepare_raw_enstrophy.py --config config.json
python prepare_encoded_enstrophy.py --config config.json
python prepare_predicted_enstrophy.py --config config.json
```

Default outputs are:

- `raw_data_grad.npz`
- `encoded_data_grad.npz`
- `predicted_data_grad.npz`

To generate extended physics inputs containing kinetic energy, helicity, and enstrophy:

```bash
python prepare_extended_physics.py --config config.json
```

Default outputs are `raw_extended.npz`, `encoded_extended.npz`, and `predicted_extended.npz`.

## Run Recovery And Evaluation

Recover enstrophy from one source at a time:

```bash
python recover_enstrophy_raw.py --config config.json
python recover_enstrophy_encoded.py --config config.json
python recover_enstrophy_predicted.py --config config.json
```

Run the combined evaluation suite and generate a summary table/figure:

```bash
python run_all_evaluations.py --config config.json
```

Run cross-source recovery:

```bash
python cross_source_sindy.py --config config.json
```

Run noise robustness and smoothing checks:

```bash
python stress_test_sindy.py --config config.json
python test_smoothing.py --config config.json
```

Recover kinetic energy, helicity, and enstrophy from the extended physics files:

```bash
python recover_physics_extended.py --config config.json
```

## Larger Experiments

These jobs load the HDF5 data and model checkpoints, then run over multiple Reynolds numbers or temporal prediction settings. They can take much longer than the single-parameter preparation scripts.

```bash
python reproduce_all_params.py --config config.json
python staircase_physics_recovery.py --config config.json
python interpolation_physics_recovery.py --config config.json
```

The staircase and interpolation scripts resume from their existing CSV files when available. Their default outputs go to `outputs.documentation_dir`.

## Useful Overrides

Examples:

```bash
python run_all_evaluations.py --raw-input ./raw_data_grad.npz --output ./evaluation_results.csv
python recover_physics_extended.py --encoded-input ./encoded_extended.npz
python reproduce_all_params.py --device auto --batch-size 32
python staircase_physics_recovery.py --h5-path /path/to/evaluation_data.h5 --documentation-dir ../Documentation
```

Generated result files are intentionally configurable through `config.json` or command-line arguments, so moving the data directory or result directory should not require editing Python source files.
