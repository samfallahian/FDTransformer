# FDTransformer

FDTransformer is a physics-modeling pipeline for turning original flow simulation
data into cube features, compressing those cubes with an autoencoder, learning
latent spatio-temporal dynamics with a transformer, and recovering/interpreting/predicting
physics.

The repository is organized as restartable stages. Each major folder has its own
README with detailed command-line options; this file is the clean project map.

## Repository Map

| Path | Role |
| --- | --- |
| `og_data_prep/` | Preprocesses original pandas `.pkl.gz` simulation files into corrected, scaled, cubed, and latent-enriched datasets. See `og_data_prep/README.md`. |
| `encoder/autoencoderGEN3/` | Trains and validates the GEN3 autoencoder that compresses 375 velocity cube features into 47 latent features. See `encoder/autoencoderGEN3/README.md`. |
| `transformer/` | Builds transformer HDF5 datasets, trains `OrderedTransformerV1`, evaluates decoded predictions, and draws metrics/robustness plots. See `transformer/README.md`. |
| `pySINDy/` | Prepares raw, encoded, and predicted fields for SINDy physics recovery and runs recovery/evaluation experiments. See `pySINDy/README.md`. |
| `helpers/` | Shared data loading, latent conversion, validation, and utility code used by the preprocessing and modeling stages. |
| `configs/` | Shared project metadata. Pipeline path configuration now lives in each stage's JSON config file. |
| `main.py` | Project-level runner for the transformer pipeline. |

## Setup

Create an environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA machines, install the PyTorch build that matches your CUDA runtime if
the default `torch` package does not provide GPU support. The training and
evaluation scripts choose CUDA first, then Apple MPS, then CPU when `device` is
set to `auto`.

## Configuration

Each pipeline area uses a local JSON config:

| Area | Example config | Local config to create |
| --- | --- | --- |
| OG data prep | `og_data_prep/pipeline_config.json` | Edit directly or copy per machine. |
| Autoencoder | `encoder/autoencoderGEN3/config.example.json` | `encoder/autoencoderGEN3/config.json` |
| Transformer | `transformer/config.example.json` | `transformer/transformer_config.json` |
| pySINDy | `pySINDy/config.example.json` | `pySINDy/config.json` |

Host-specific paths should be kept in those JSON files or passed with CLI
overrides. The old host-keyed preferences file has been removed.

## Downloadable Data

Ready-made transformer datasets are available on Google Drive:

| File | Use | Suggested config key |
| --- | --- | --- |
| [`training_data.h5`](https://www.kaggle.com/datasets/lfmpaper/training-dataset-training-data-h5) | Transformer training HDF5. | `paths.training_h5` |
| [`validation_data.h5`](https://www.kaggle.com/datasets/lfmpaper/validation-data-validation-data-h5) | Transformer validation HDF5. | `paths.validation_h5` |
| [`evaluation_data.h5`](https://www.kaggle.com/datasets/lfmpaper/evaluation-dataset-evaluation-data-h5) | Prepared transformer evaluation HDF5 with original velocity metadata. | `paths.evaluation_h5` |
| [`Original-data.zip`](https://www.kaggle.com/datasets/lfmpaper/original-data-for-evaluation) | Original evaluation source data. Includes one `7.452e-02` m/s flow-speed case for evaluation. | `paths.evaluation_input_root` after extraction |

After downloading, place the HDF5 files wherever your local
`transformer/transformer_config.json` points, or update the matching `paths.*`
entries. Extract `Original-data.zip` before using it as `evaluation_input_root`.

## Typical Workflow

1. Run `og_data_prep/` stages to produce corrected/scaled cube files and
   latent-enriched `.pkl.gz` files.
2. Train or validate the GEN3 autoencoder in `encoder/autoencoderGEN3/`.
3. Build transformer HDF5 files, train the transformer, and evaluate it.
4. Use `pySINDy/` to compare raw, encoded, and transformer-predicted physics.

## Transformer Runner

The root `main.py` provides a clean way to run transformer stages from the
repository root:

```bash
python main.py transformer --list
python main.py transformer data --config transformer/transformer_config.json
python main.py transformer train --config transformer/transformer_config.json
python main.py transformer eval --config transformer/transformer_config.json
python main.py transformer all --config transformer/transformer_config.json
```

Available transformer steps are:

```text
prepare-data  validate-data  train  prepare-eval  evaluate  plots  corruption
```

Useful groups are:

```text
data = prepare-data + validate-data
eval = prepare-eval + evaluate + plots
all  = every transformer stage in order
```

For a small smoke-test dataset:

```bash
python main.py transformer data --config transformer/transformer_config.json --test-run
```

Common overrides are forwarded to the stages that support them:

```bash
python main.py transformer train evaluate \
  --config transformer/transformer_config.json \
  --device auto \
  --batch-size 8 \
  --limit-samples 1000
```

Preview commands without running them:

```bash
python main.py transformer all --config transformer/transformer_config.json --dry-run
```

For detailed per-stage options, run the stage script directly with `--help` or
see `transformer/README.md`.

## Requirements

`requirements.txt` lists the direct runtime packages used by the project:
PyTorch, NumPy/Pandas/SciPy, HDF5 support, plotting/reporting utilities, WandB,
and PySINDy/scikit-learn for physics recovery. It intentionally avoids storing a
full environment freeze so transitive dependencies can resolve cleanly for each
platform.
