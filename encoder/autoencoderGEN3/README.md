# GEN3 Autoencoder

This folder trains and validates a PyTorch autoencoder that compresses 375-dimensional velocity cubes into a 47-dimensional latent representation.

The core model definitions live in `models.py`. `BaseAE` defines the common forward pass and loss, while `get_model_by_index()` selects one of ten GEN3 architectures. Model index `4`, `Model_GEN3_05_AttentionSE`, is the production model used by the production training and validation scripts.

## Files

- `models.py`: ten autoencoder architectures plus shared residual/attention blocks.
- `train_gen3.py`: comparison training for one or all ten architectures.
- `train_model_05_production.py`: long production training for the AttentionSE model.
- `validate_model_05_production.py`: reconstructs validation files from stored latent columns and writes CSV/PNG metrics.
- `validate_model_05_production_ROM_PCA.py`: compares the production autoencoder against PCA with the same latent dimension.
- `ablation.py`: PCA latent-dimension ablation against the fixed 47-dimensional autoencoder.
- `gradients.py`: vorticity/gradient fidelity check.
- `validation_divergence.py`: divergence consistency check.
- `config.py`: shared JSON config and path resolution helpers.
- `config.example.json`: copyable config template.

## Data Expectations

Training expects two pickle files containing numeric arrays with shape `(n_samples, 375)`:

- `training_auto_encoder.pkl`
- `validation_auto_encoder.pkl`

Validation utilities expect recursive `.pkl` or `.pkl.gz` pandas DataFrames. They look for:

- `latent_1` through `latent_47` when decoding with the trained autoencoder.
- 375 velocity columns, preferably named with the `velocity_` prefix.

## Setup

Run from the repository root or from `encoder`:

```bash
cd /Users/deeplearning/Sources/FDTransformer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For NVIDIA CUDA, install a CUDA-enabled PyTorch build if the plain requirements install gives you CPU-only PyTorch. The scripts automatically select CUDA first, then Apple MPS, then CPU.

Check the active accelerator:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
PY
```

On this machine, `system_profiler` reports an Apple M5 Max GPU with Metal support and `nvidia-smi` is not present, so no NVIDIA CUDA GPU is visible from this shell. The current `python3` also does not have `torch` installed, so PyTorch MPS/CUDA availability should be rechecked after installing dependencies in the active environment.

## Configure Paths

Copy the example config and edit the paths:

```bash
cd /Users/deeplearning/Sources/FDTransformer/encoder
cp autoencoderGEN3/config.example.json autoencoderGEN3/config.json
```

Relative paths inside `config.json` are resolved from the repository root (`/Users/deeplearning/Sources/FDTransformer`). Relative paths passed as CLI args are resolved from your current working directory.

The important config keys are:

- `data.train_path` and `data.val_path`: explicit training and validation pickle paths.
- `data.data_root`: directory containing the training and validation pickle files.
- `data.validation_data_dir`: directory of per-file validation DataFrames.
- `paths.comparison_save_dir`: checkpoints from `train_gen3.py`.
- `paths.production_save_dir`: checkpoints from `train_model_05_production.py`.
- `paths.production_model_path`: trained AttentionSE checkpoint used by validation scripts.
- `paths.results_dir`: CSV/PNG validation output directory.

Every data/result path can also be passed on the command line, for example `--train_path`, `--val_path`, `--data_dir`, `--save_dir`, `--model_path`, or `--output_dir`.

## Train

Train the production AttentionSE model:

```bash
cd /Users/deeplearning/Sources/FDTransformer/encoder
python autoencoderGEN3/train_model_05_production.py \
  --config autoencoderGEN3/config.json \
  --epochs 1000 \
  --batch_size 4096 \
  --no_wandb
```

Train a quick smoke test:

```bash
python autoencoderGEN3/train_model_05_production.py \
  --config autoencoderGEN3/config.json \
  --epochs 1 \
  --data_percentage 1 \
  --batch_size 128 \
  --no_wandb
```

Compare all ten GEN3 architectures:

```bash
python autoencoderGEN3/train_gen3.py \
  --config autoencoderGEN3/config.json \
  --epochs 100 \
  --data_percentage 10 \
  --no_wandb
```

Train only one comparison model:

```bash
python autoencoderGEN3/train_gen3.py \
  --config autoencoderGEN3/config.json \
  --model_idx 4 \
  --epochs 100 \
  --no_wandb
```

`model_idx` is zero-based, so `4` selects `Model_GEN3_05_AttentionSE`.

## Validate

Validate the production model on random validation files:

```bash
python autoencoderGEN3/validate_model_05_production.py \
  --config autoencoderGEN3/config.json \
  --n_files 100 \
  --seed 42
```

Compare AE vs PCA:

```bash
python autoencoderGEN3/validate_model_05_production_ROM_PCA.py \
  --config autoencoderGEN3/config.json \
  --n_files 20 \
  --latent_dim 47
```

Run the physical consistency checks:

```bash
python autoencoderGEN3/ablation.py --config autoencoderGEN3/config.json
python autoencoderGEN3/gradients.py --config autoencoderGEN3/config.json
python autoencoderGEN3/validation_divergence.py --config autoencoderGEN3/config.json
```

## Outputs

Production training writes rolling epoch checkpoints and scripted models to `paths.production_save_dir`, keeping the last five epoch checkpoints. It also maintains `Model_GEN3_05_AttentionSE_absolute_best.pt`.

Comparison training writes each model's best checkpoint and TorchScript export to `paths.comparison_save_dir`.

Validation scripts write summaries, detailed CSVs, and plots to `paths.results_dir` unless overridden with `--output_dir` or a more specific output argument.
