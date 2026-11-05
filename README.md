# FDTransformer

A causal transformer that forecasts the evolution of a turbulent wake in the 47-dim latent space of a frozen GEN3 AttentionSE autoencoder. It's scored on how much its 68-frame autoregressive rollout beats a persistence baseline ("freeze the last frame"), measured on decoded centroid velocity (`improvement_pct`).

For background, see [`docs/DATA_PIPELINE_WALKTHROUGH.md`](docs/DATA_PIPELINE_WALKTHROUGH.md) for the data format, and [`docs/OVERVIEW.md`](docs/OVERVIEW.md) §81–83 for the handoff notes and current research status. OVERVIEW.md is the full lab notebook. Its file names and paths are historical: they refer to the old `transformer_neurIPS/` layout.

## Repository layout

| Path | What it is |
|---|---|
| `run_train.sh` | **Entry point.** Launches one training run on one GPU you choose. |
| `train_production_transformer_deep_dive.py` | The trainer: config, arms, data loading, training loop, eval, checkpointing, wandb. |
| `model_variants.py` | Model architectures (base / swiglu / mqa / conv, token- or frame-tokenized, optional MoE). |
| `pyproject.toml`, `uv.lock` | uv environment definition and pinned lock file (PyTorch CUDA 13.0 wheels on Linux). |
| `assets/` | Frozen decoder and warm-start checkpoint. Copy these in by hand, see [`assets/README.md`](assets/README.md). |
| `data_prep/` | Scripts that **rebuild** `train_80.h5`/`val_80.h5` from the raw simulation tree, plus `h5_to_raw.py`. Not needed for training. |
| `tests/` | `pytest` checks: data-file shape/content, wandb project naming. |
| `docs/` | `OVERVIEW.md` (lab notebook) and `DATA_PIPELINE_WALKTHROUGH.md`. |
| `saved_models/` | *(created at runtime, git-ignored)*: checkpoints `r<round>_<arm>_{best,rollout_best,train_best,latest}.pt`, `_status.json`, `_alerts.jsonl`, `ridge_frame_map.pt`. |
| `runs/` | *(created at runtime, git-ignored)*: one folder per launch with `train.log`, `run_info.txt` and the trainer's `<arm>.json` result. |

## 1. Set up the environment (server)

Target machine: 2× H200 NVL, CUDA 13.1 driver, Linux.

```bash
# once: install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd FDTransformer
uv sync --python 3.12         # builds .venv/ from the committed uv.lock
# uv sync --extra data-prep   # only if you need to rebuild datasets (pandas/pyarrow/tqdm)

uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
# expected on the server: 2.14.1+cu130 13.0 2
```

On Linux, `torch` comes from the PyTorch `cu130` wheel index configured in `pyproject.toml`. `uv.lock` pins it to `2.14.1+cu130`, which runs on the CUDA 13.1 driver. On macOS it resolves to the regular PyPI build (MPS/CPU), so the same files work for local debugging.

Don't copy your Mac's `.venv/` to the server. It's git-ignored, so `git pull` won't carry it, but with `rsync`, add `--exclude .venv`. Build the server's own venv with `uv sync`.

## 2. Put the inputs in place

1. **Data**: `train_80.h5` and `val_80.h5` in `/data/ptv-092026` (the launcher's default `--data-dir`).
2. **Decoder** (always required): `assets/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`.
3. **Warm-start checkpoint** (optional, only for `--warm-start`): `assets/r2_h11_ridge_distill_rollout_best.pt`.

You can extract both asset files from the `LFM` git branch. [`assets/README.md`](assets/README.md) has the exact commands.

wandb is on by default: run `uv run wandb login` once, or set `WANDB_API_KEY`, or pass `--no-wandb`. The project is `Persistence_Linear_v1` (`Config.WANDB_PROJECT`).

## 3. Run

```bash
./run_train.sh --dry-run          # pre-flight checks + the exact command; launches nothing
./run_train.sh --smoke            # ~1 min synthetic check: model builds, is causal, trains
./run_train.sh                    # reference recipe on GPU 0
./run_train.sh --gpu 1            # same, on GPU 1
./run_train.sh --gpu 1 --detach   # background (survives logout); prints the log path + stop command
```

By default, `./run_train.sh` runs arm `h11_ridge_distill` with preset `trial112` (`WEIGHT_DECAY=0.0966…`, `DROPOUT=0.0735…`, `EMA_DECAY=0.98`, `AR_LOSS_WEIGHT=0.683…`) from a **cold start** (random init), with the 2026-11-05 fixes on: the latent-consistency loss, the standardised fp32 output head, and EMA-correct checkpoints. The budget is 20000 steps, with an eval every 500 steps on 256 param-stratified rollout sequences. Promotion uses the per-experiment mean improvement, and early stopping fires after 5000 steps without a new promoted checkpoint, but never before the curricula finish. See [`docs/CHANGES_2026-11-05.md`](docs/CHANGES_2026-11-05.md) for what changed and why, and for the `--set` line that reproduces the old recipe.

Warm starting is opt-in. `--warm-start` uses `assets/r2_h11_ridge_distill_rollout_best.pt`, and `--warm-start PATH` uses any checkpoint. When warm starting, the launcher also applies a fine-tune schedule so the curricula don't restart (`AR_FRAMES_START=None AR_WEIGHT_WARMUP_FRAC=0 RIDGE_DISTILL_WARMUP_FRAC=0 LEARNING_RATE=3e-4 WARMUP_FRAC=0.01`). Your own `--set` values override it, and `--warm-start-keep-schedule` turns it off.

If `saved_models/ridge_frame_map.pt` doesn't exist yet, the launcher first runs the trainer's `--diagnostics-only` pass on the same GPU to fit it. The `h11_ridge_distill` distillation loss needs that map.

More examples:

```bash
# The #1 open research question (OVERVIEW.md §83): drop the (x,y,z,t,param) input columns.
./run_train.sh --gpu 0 --round 500 --wandb-group metaAB --set USE_META_COLS=True
./run_train.sh --gpu 1 --round 501 --wandb-group metaAB --set USE_META_COLS=False

# Continue training from a checkpoint (opt-in warm start)
./run_train.sh --gpu 0 --round 502 --warm-start saved_models/r500_h11_ridge_distill_rollout_best.pt

# Any other arm, without the trial-112 preset
./run_train.sh --arm s7_h9_scaled --preset none --round 5
./run_train.sh --list-arms

# Extra trainer flags go after --
./run_train.sh -- --batch-size 32
```

You can run two launches at the same time on different GPUs, as in the A/B example above. Each one is an independent process. Give each run its own `--round`, or a different arm: two live runs with the same `r<round>_<arm>` are refused by the trainer's run lock.

### Launcher options

| Option | Default | Meaning |
|---|---|---|
| `--gpu N` | `0` | GPU index. Exported as `CUDA_VISIBLE_DEVICES=N`. |
| `--arm NAME` | `h11_ridge_distill` | Which arm (named set of Config overrides) to train. |
| `--preset NAME` | `trial112` for h11, else `none` | Hyper-parameter preset applied before your `--set` values. |
| `--round N` | `1` | Run tag used in checkpoint and wandb names. Re-using a round **resumes** from `r<N>_<arm>_latest.pt`. |
| `--fresh` | off | Ignore an existing `_latest.pt` and start over. |
| `--data-dir DIR` | `/data/ptv-092026` | Folder containing `train_80.h5` / `val_80.h5`. |
| `--decoder PATH` | `assets/Model_GEN3_05_…_scripted.pt` | Frozen decoder. |
| `--warm-start [PATH]` | off (cold start) | Opt-in initial weights. With no PATH, uses `assets/r2_h11_ridge_distill_rollout_best.pt`. Also applies the warm-start fine-tune schedule. |
| `--warm-start-keep-schedule` | off | Warm start without the fine-tune schedule. |
| `--no-warm-start` | (default) | Cold start. Kept so old command lines still parse. |
| `--max-steps N` | `20000` | Optimizer steps. This also sizes the LR warmup/cosine schedule and the curricula. |
| `--max-hours H` | `6.0` | Wall-clock kill switch only. |
| `--val-every N` | `500` | Eval cadence (steps). |
| `--rollout-seqs N` | `256` | Validation rollout sequences per eval, stratified evenly across the val experiments. |
| `--patience N` | `5000` | Early-stop patience in steps, never applied before the curricula finish (`EARLY_STOP_MIN_STEPS`). `0` disables it. |
| `--set KEY=VALUE` | none | Any `Config` field (repeatable). |
| `--wandb-group NAME` / `--no-wandb` | none | wandb grouping / disable wandb. |
| `--smoke`, `--diagnostics`, `--list-arms` | none | Alternative modes. |
| `--detach`, `--dry-run` | off | Run in the background with nohup / only print checks and the command. |

These environment variables pass straight through to the trainer: `PFD_DATA_LOAD_THREADS` (HDF5 read threads, default 8), `PFD_CUDA_MICRO_BATCH` (default 64 on H200), `PFD_COMPILE_MODEL=1` (torch.compile, off by default), `PFD_NO_COLOR=1`. `PYTHON` overrides the interpreter; the default is `.venv/bin/python`, then `uv run python`.

### Outputs

The best checkpoint is `saved_models/r<round>_<arm>_rollout_best.pt`. It's promoted when the per-experiment mean improvement over persistence (`PROMOTE_METRIC`) beats the previous best and is above 0. When EMA is on, its `model_state_dict` and scripted twin are the **EMA weights** that earned the score; the live weights are in `raw_model_state_dict`, and `weights_source` says which. The trainer also writes `_best.pt` (best teacher-forced `val_tf_loss`), `_train_best.pt`, and `_latest.pt` (live weights plus `ema_state_dict`, about every 10 minutes, with up to 5 archived copies).

Each eval prints the pooled improvement, a per-experiment line with the ridge map's score on the same sequences and the "ceiling captured", and `val latent nMSE` (it should approach the copy-previous-frame reference printed at start-up). Each kind also gets a TorchScript `_scripted.pt` twin. Live status is in `saved_models/r<round>_<arm>_status.json`, and the console log is in `runs/<run_id>/train.log`.

### Optional: faster startup

The HDF5 files store one row per chunk. `data_prep/h5_to_raw.py` writes a flat `.raw` copy next to each `.h5`, and the trainer picks it up automatically. It needs about 14 GB extra disk space and write access to the data folder:

```bash
PFD_DATA_DIR=/data/ptv-092026 uv run python data_prep/h5_to_raw.py
```

## 4. Tests

```bash
PFD_DATA_DIR=/data/ptv-092026 uv run pytest -q
```

## Files the code expects from outside this folder

| What | Status |
|---|---|
| GEN3 decoder `Model_GEN3_05_AttentionSE_absolute_best_scripted.pt` | Was loaded from `../encoder/autoencoderGEN3/saved_models_production/`, **outside** the repo. Now loaded from `assets/`, and you need to copy it there. |
| Warm-start `r2_h11_ridge_distill_rollout_best.pt` | Was in `saved_models/` (removed in the cleanup). Now expected in `assets/`. |
| `/data/ptv-092026/{train,val}_80.h5` | Configurable with `--data-dir` / `PFD_DATA_DIR`. |
| Raw simulation trees (`/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data[_wLatent]`) | Only used by `data_prep/prepare_data.py` and `data_prep/build_wake_atlas.py` to rebuild the datasets. Override with `PFD_SOURCE_ROOT` / `PFD_RAW_SOURCE_ROOT` (or `--source-root`). |
| `saved_models/old/r1_a3b_delta_ar_rollout_best.pt` (v1.0 winner) | Only the trainer's built-in default `--warm-start` when it's run directly without the launcher. The launcher always passes an explicit path or `--no-warm-start`. |

## Changelog

### 2026-11-05: review fixes, cold start by default

Details and rationale are in [`docs/CHANGES_2026-11-05.md`](docs/CHANGES_2026-11-05.md).

- Trainer and model:
  - added a latent-consistency loss (`LATENT_LOSS_WEIGHT=1.0`);
  - the output head is now standardised (`OUTPUT_NORMALIZED`, with exact conversion of old checkpoints) and runs in fp32 under autocast (`FP32_OUTPUT_HEAD`);
  - EMA-promoted checkpoints now store the EMA weights, and the EMA is restored on resume;
  - evals use 256 param-stratified sequences, report per-experiment and ridge scores, and promote on the per-experiment mean (`PROMOTE_METRIC`);
  - `_best.pt` is selected on `val_tf_loss`;
  - early stopping waits for the curricula to finish (`EARLY_STOP_MIN_STEPS`).
- Warm start is opt-in in both the trainer and the launcher. The launcher applies a fine-tune schedule when warm starting, and its defaults are now 20000 steps / 6 h / eval every 500 / 256 sequences / patience 5000.
- New `tests/test_review_fixes.py`.

### 2026-11-05: code & research review

- Added [`docs/REVIEW_2026-11-05.md`](docs/REVIEW_2026-11-05.md): findings on the code and recipe (off-manifold latent predictions, EMA checkpoint bug, val-param extrapolation, 16-sequence eval, warm-start curriculum reset) and a warm-start vs. cold-start recommendation. No code changed.

### 2026-11-05: server cleanup (branch `LFM-updated`)

- Added `run_train.sh` (single GPU, selectable with `--gpu`), `pyproject.toml` + `uv.lock` (uv; Linux resolves `torch 2.14.1+cu130`), this README and `assets/README.md`.
- Trainer: the data directory can now be set with `PFD_DATA_DIR`, and the decoder default moved from `../encoder/...` (outside the repo) to `assets/`. File names stay pinned, and no training or eval logic changed.
- Removed the 10 `run_sweep_*.sh` launchers, `sweep_deep_dive.py`, the RunPod tooling (`singleshot/`), the legacy trainer (`train_production_transformer.py`, `model.py`, `train_search.py`, `run_diagnostics.py`) and its tests, the broken v1.0 eval harness (`persistence_formal_documentation.py`, `tests/test_model_vs_baseline.py`), and one-off scripts (`analyze_windows.py`, `decompress_h5.py`, `identify_training_wake.py`, `make_sweep_sample_data.py`, `model_soup_eval.py`, `diagnose_rollout_noise_sensitivity.py`, `requirements_sweep.txt`). Also unstaged the old-run `saved_models/*` status files from the git index. All of these are still in git history.
- Moved the data-prep scripts to `data_prep/` and the docs to `docs/`. Made the data-prep paths configurable with env vars. `tests/test_data_quality.py` now checks the 80-frame files instead of the stale `train_40.h5`.
