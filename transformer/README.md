# FDTransformer Transformer Pipeline

This directory contains the transformer-side pipeline for:

1. preparing latent HDF5 datasets from precomputed `.pkl.gz` cube files,
2. validating the generated HDF5 files,
3. training `OrderedTransformerV1`,
4. evaluating predictions in latent and decoded velocity space,
5. generating summary plots and optional corruption-robustness metrics.

The code no longer assumes machine-specific input or output paths. Put paths in a config file, or pass them on the command line.

## Files

| File | Purpose |
| --- | --- |
| `transformer_config.py` | Shared config loader, path resolver, and defaults. |
| `config.example.json` | Copy this to `transformer_config.json` and edit paths for your machine. |
| `Ordered_010_Prepare_Dataset.py` | Builds `training_data.h5` and `validation_data.h5` from latent cube files. |
| `Ordered_020_DataSet_Validations.py` | Checks generated HDF5 samples for coordinate/time consistency. |
| `Ordered_100_TrainTransformer_v1.py` | Trains the ordered transformer. |
| `Ordered_150_Prepare_Evaluation_Dataset.py` | Builds an evaluation HDF5 with final-step original velocities. |
| `Ordered_200_EvaluateTransformer_v1.py` | Evaluates the trained transformer and exports JSON plus pointwise pickle output. |
| `Ordered_240_EvaluateTransformer_v1_drawplots.py` | Generates plots from `evaluation_results.json`. |
| `Ordered_300_EvaluateTransformer_v1_with_datacorruption.py` | Runs corruption sweep evaluation. |
| `transformer_model_v1.py` | Transformer model definition. |
| `dataset.py` | Simple HDF5 dataset wrapper. |

## Configuration

Create a local config:

```bash
cp config.example.json transformer_config.json
```

Edit the paths in `transformer_config.json`. Relative paths are resolved from this directory; absolute paths also work.

The pipeline scripts accept:

```bash
--config /path/to/transformer_config.json
```

If `--config` is omitted, scripts use `TRANSFORMER_CONFIG` when set, then `./transformer_config.json` when it exists, then built-in safe defaults. Command-line arguments override the config file.

## GPU Behavior

Device selection defaults to `auto`.

Priority is:

1. CUDA GPU, when `torch.cuda.is_available()` is true.
2. Apple MPS, when available.
3. CPU.

You can force a device with `--device cuda`, `--device mps`, or `--device cpu`. Training and evaluation defaults use smaller batches (`training.batch_size=32`, `evaluation.batch_size=8`, `evaluation.micro_batch_size=4`) so they are more likely to run on general GPUs. Increase these values only after confirming memory headroom.

## Data Layout

Training, validation, and evaluation HDF5 files store samples as:

```text
data: (N, NUM_TIME, 26, 52) float32
```

Default `NUM_TIME` is `80`, so the default flattened transformer sequence length is `80 * 26 = 2080` tokens.

Feature mapping:

| Indices | Name | Description |
| --- | --- | --- |
| `0-46` | Latents | 47-dimensional autoencoder latent vector. |
| `47` | X | Physical X coordinate. |
| `48` | Y | Physical Y coordinate. |
| `49` | Z | Physical Z coordinate. |
| `50` | Relative time | Time index inside the sampled window. |
| `51` | Parameter | Experiment parameter value, parsed from the parameter-set directory name. |

Evaluation HDF5 files also include:

```text
originals:  (N, 26, 3) final-step original vx, vy, vz
start_t:    (N,) source frame index
start_time: (N,) source time value when available
```

## Full Pipeline

From the repository root, the transformer stages can also be run through the
project runner:

```bash
python main.py transformer --list
python main.py transformer data --config transformer/transformer_config.json
python main.py transformer train --config transformer/transformer_config.json
python main.py transformer eval --config transformer/transformer_config.json
```

Use `python main.py transformer all --config transformer/transformer_config.json`
to run every transformer stage in order, or add `--dry-run` to preview the exact
commands. The individual scripts below remain useful when you need stage-specific
flags.

### 1. Prepare train and validation data

```bash
python Ordered_010_Prepare_Dataset.py --config transformer_config.json
```

Useful overrides:

```bash
python Ordered_010_Prepare_Dataset.py \
  --input-root /path/to/Final_Cubed_OG_Data_wLatent \
  --output-dir /path/to/transformer_input \
  --num-samples 250000
```

Quick smoke test:

```bash
python Ordered_010_Prepare_Dataset.py --config transformer_config.json --test_run
```

### 2. Validate generated HDF5 files

```bash
python Ordered_020_DataSet_Validations.py --config transformer_config.json
```

Or validate specific files:

```bash
python Ordered_020_DataSet_Validations.py --h5 /path/to/training_data.h5 --h5 /path/to/validation_data.h5
```

### 3. Train the transformer

```bash
python Ordered_100_TrainTransformer_v1.py --config transformer_config.json
```

Common overrides for smaller GPUs:

```bash
python Ordered_100_TrainTransformer_v1.py \
  --config transformer_config.json \
  --batch-size 16 \
  --limit-samples 10000 \
  --num-workers 2
```

Use `--limit-samples none` or omit it to train on the full HDF5 files. Checkpoints are written to `paths.checkpoint_dir` using `training.checkpoint_base_name`.

### 4. Prepare evaluation data

```bash
python Ordered_150_Prepare_Evaluation_Dataset.py --config transformer_config.json
```

By default the evaluation sampler uses `data.evaluation_start_t`. To sample random windows instead:

```bash
python Ordered_150_Prepare_Evaluation_Dataset.py --config transformer_config.json --start_t none
```

If your shell passes `none` as text, use the config file and set `"evaluation_start_t": null`.

### 5. Evaluate

```bash
python Ordered_200_EvaluateTransformer_v1.py --config transformer_config.json
```

Outputs:

| Output | Config key |
| --- | --- |
| Summary metrics JSON | `paths.evaluation_results_json` |
| Pointwise original/predicted pickle | `paths.pred_gt_pickle` |

For a faster evaluation:

```bash
python Ordered_200_EvaluateTransformer_v1.py \
  --config transformer_config.json \
  --limit-samples 1000 \
  --batch-size 4 \
  --micro-batch-size 2
```

### 6. Draw evaluation plots

```bash
python Ordered_240_EvaluateTransformer_v1_drawplots.py --config transformer_config.json
```

Plots are written to `paths.plots_dir`.

### 7. Optional corruption sweep

```bash
python Ordered_300_EvaluateTransformer_v1_with_datacorruption.py --config transformer_config.json
```

For a quick run:

```bash
python Ordered_300_EvaluateTransformer_v1_with_datacorruption.py \
  --config transformer_config.json \
  --limit-samples 25 \
  --levels 21 \
  --batch-size 4
```

## Notes

- Keep `data.num_time` consistent with the HDF5 files and the checkpoint architecture. The default pipeline writes and trains on 80 time steps.
- If you load an older checkpoint trained with 8 time steps, set `"data": {"num_time": 8}` or pass `--num-time 8`.
- The evaluation scripts prefer `paths.evaluation_h5` when it exists and fall back to `paths.validation_h5`.
- `wandb_mode` can be set to `offline` or `disabled` in the config when running without Weights & Biases login.
