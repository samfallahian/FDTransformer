# OG Data Preparation

This directory contains the preprocessing pipeline that turns original simulation
dataframes into cube-shaped velocity features, optional autoencoder datasets, and
latent-enriched files for downstream modeling.

The scripts are intentionally small and file-oriented. Each stage reads `.pkl.gz`
dataframes from one configured location, writes the next representation to another
configured location, and can be run independently for restartable jobs.

## Configuration

All data and model paths are read from `pipeline_config.json` by default. You can
use another config with `--config path/to/config.json`, or override individual
paths through the CLI flags on each script.

Important config keys:

```json
{
  "unmodified_data_dir": "/data/Unmodified_OG_Data",
  "corrected_data_dir": "/data/Corrected_OG_Data",
  "scaled_data_dir": "/data/Scaled_OG_Data",
  "cubed_data_dir": "/data/Cubed_OG_Data",
  "final_cubed_data_dir": "/data/Final_Cubed_OG_Data",
  "final_latent_data_dir": "/data/Final_Cubed_OG_Data_wLatent",
  "autoencoder_dataset_dir": "/data",
  "cube_mapping_csv": "../cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv",
  "autoencoder_model_path": "../encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best.pt"
}
```

Relative config paths are resolved relative to the config file. Host-specific
paths should live in a local copy of `pipeline_config.json` rather than in source
code.

## Inputs

Raw input files are expected to be pandas pickle files with gzip compression:

- extension: `.pkl.gz`
- coordinate columns: `x`, `y`, `z`
- time column: `time` or `t`
- velocity columns: `vx`, `vy`, `vz`

## Pipeline

Diagnostics on raw data:

```bash
python Ordered_010_DetermineHealth.py --config pipeline_config.json
python Ordered_020_Find_Extremes.py --config pipeline_config.json
```

Main preprocessing:

```bash
python Ordered_030_dType_Corrections.py --config pipeline_config.json
python Ordered_040_CorrectScale.py --config pipeline_config.json
python Ordered_005_AllPossibleCombos.py --config pipeline_config.json
python Ordered_050_RowFilter_TimeSeperate.py --config pipeline_config.json
python Ordered_060_Create_Cubes_Per_Line.py --config pipeline_config.json
```

Model-oriented outputs:

```bash
python Ordered_100_build_autoencoder_dataset.py --config pipeline_config.json
python Ordered_200_precomputeAllLatent.py --config pipeline_config.json
python Ordered_250_ValidateOneFile.py --config pipeline_config.json
```

`Ordered_200_precomputeAllLatent.py` and `Ordered_250_ValidateOneFile.py` use
`--device auto` by default. They choose CUDA first, then Apple MPS, then CPU. Use
`--device cpu` to force CPU or `--device cuda` to fail fast if CUDA is expected
but unavailable. On GPU/MPS, latent precompute defaults to one worker so a single
accelerator is not filled with multiple model copies; override with `--workers`
only when the server has enough accelerator memory or multiple devices.

## Stage Summary

- `Ordered_010_DetermineHealth.py` checks time-step coverage and row-count consistency.
- `Ordered_020_Find_Extremes.py` reports velocity min/max values and writes a distribution plot.
- `Ordered_030_dType_Corrections.py` normalizes dtypes and renames velocity columns to `original_vx`, `original_vy`, and `original_vz`.
- `Ordered_040_CorrectScale.py` scales original velocity columns back into `vx`, `vy`, and `vz`.
- `Ordered_005_AllPossibleCombos.py` builds the centroid-to-5x5x5-neighbor mapping CSV from the scaled grid.
- `Ordered_050_RowFilter_TimeSeperate.py` filters scaled files to all coordinates needed for cubes and writes one file per time step.
- `Ordered_060_Create_Cubes_Per_Line.py` joins each centroid with neighbor velocities and writes 375 velocity feature columns.
- `Ordered_100_build_autoencoder_dataset.py` samples fixed training and validation arrays for autoencoder training.
- `Ordered_200_precomputeAllLatent.py` loads the configured autoencoder checkpoint and appends `latent_1` through `latent_47`.
- `Ordered_250_ValidateOneFile.py` decodes one latent-enriched file and writes reconstruction/error columns to CSV.

## Common Overrides

Every script accepts `--config`. Frequently used per-script overrides include:

- `--dir`, `--input_dir`, `--input_root`, or `--input_file` for input data.
- `--output_dir`, `--output_root`, `--output_csv`, or `--output` for outputs.
- `--map_path` or `--filter_csv` for the centroid-neighbor mapping CSV.
- `--model_path` and `--model_index` for autoencoder stages.
- `--device auto|cuda|mps|cpu` for autoencoder inference stages.
- `--workers`, `--batch_size`, and `--first_only` for larger parallel stages.

Use `python <script>.py --help` for the exact flags on a stage.

## Dependencies

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The autoencoder and dataset stages also import code from the parent project,
including `EfficientDataLoader`, `TransformLatent`, and
`encoder.autoencoderGEN3.models`.
