# transformer_neurIPS/ — Folder Overview

This document pins **v1.0 (40-frame)** — the model already trained, checkpointed under
`saved_models/`, and evaluated in `tests/reports/` — and scopes **v2.0 (80-frame)**
as the successor. The `encoder_neurIPS/` autoencoder is **retired from downstream use**
starting with v2.0; the transformer stays in latent space for loss, and the reporting-only
decode path is repointed at the scripted `autoencoderGEN3` model (see §Versioning).

---

## 1. Layout

Top-level files under `transformer_neurIPS/`:

| Path | Role |
|---|---|
| `prepare_data.py` | Builds `data/train_<N>.h5` / `data/val_<N>.h5` from the raw pickled per-frame dataframes under `/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent`. Parameterized by `--num-time`. |
| `model.py` | Baseline transformer implementation (pre-variant sweep). |
| `model_variants.py` | The four architectural variants (`base`, `swiglu`, `mqa`, `conv`) + shared RoPE / MQA / SwiGLU building blocks. Only the positional-embedding capacity depends on `NUM_TIME`. |
| `train_production_transformer.py` | Original single-run production trainer. |
| `train_production_transformer_deep_dive.py` | Trainer of record for v1.0's Round-1 arms sweep; produces the `r1_*` checkpoints. |
| `train_search.py` | Small hyperparameter search driver. |
| `sweep_deep_dive.py` | Sweep runner used to launch the Round-1 arms. |
| `analyze_windows.py` | Ad-hoc window-analysis helper. |
| `identify_training_wake.py` | Script that produced the 24 hand-curated `WAKE_COORDS`. |
| `run_diagnostics.py` | Post-hoc diagnostic runner. |
| `persistence_formal_documentation.py` | Apples-to-apples model-vs-persistence evaluation harness with ANSI/rainbow console + bootstrap CIs; writes `Documentation/persistence_formal/`. |
| `tests/` | Unit tests (data quality, metrics, checkpointing) + leaderboard/deep-dive report generator (`test_model_vs_baseline.py`). |
| `tests/reports/` | `model_leaderboard.{md,csv}` + `r1_a3b_delta_ar_deep_dive.{md,csv}` — the only published numeric artifacts for v1.0 checkpoints. |
| `saved_models/` | v1.0 checkpoints (see §5). |
| `data/` | v1.0 (`train_40.h5`, `val_40.h5`) + v2.0 (`train_80.h5`, `val_80.h5`) HDF5 files. |
| `wandb/` | Local W&B run directories from v1.0. |

---

## 2. Data pipeline

`prepare_data.py` writes each sequence as a `float32` block of shape
`(NUM_TIME, NUM_X=26, 52)`. The 52 feature columns are (from
`prepare_data.py` lines 76–85):

```
0:47  latents (47)        47  x        48  y (int32)        49  z (int32)
50  t_index (0..NUM_TIME−1)                                  51  param (float32)
```

Every sequence is split into a **12-frame context** and an **(N − 12) forecast**
tail; at 40 frames that is 12 + 28 (233.3 ms forecast @ 120 Hz), at 80 frames
it is 12 + 68 (566.7 ms forecast).

### Non-overlapping window tiling — the answer to "why does N=80 have ~half the sequences?"

From `prepare_data.py` line 29:

```python
WINDOWS_PER_COORD = TOTAL_TIMESTAMPS // NUM_TIME_NEW  # 30 at N=40, 15 at N=80
```

For each `(param, wake_coord)` pair the wake cohort emits
`WINDOWS_PER_COORD` **disjoint** non-overlapping windows tiling all 1200
frames exactly once (line 110: `start_step = w_idx * NUM_TIME_NEW + 1`).
The random cohort is then generated **1-for-1** with the wake cohort (line 148:
`while len(random_plans) < num_wake_plans`). So doubling `NUM_TIME`
exactly halves the sequence count in both cohorts. **v1.0 was not double-sampling
at N=40**; both cohorts tile 1200 frames exactly once at every `NUM_TIME`.

The underlying source-frame coverage is conserved by construction, but the
retained files are not required to have exactly equal sequence-frame counts:
`3730 × 80 = 298,400` for N=80 versus `7464 × 40 = 298,560` for N=40. The
160 sequence-frame-slot difference (about 0.054%) comes from valid-sequence
drop-outs at line 210 and the fact that missing/all-zero frames can invalidate
different windows after the two window sizes are applied; it is not evidence
that N=40 was sampled twice.

### 50 / 50 wake vs. random sampling — verbatim from `prepare_data.py`

The split is on `(y, z)`, **not on `x`**. `X_COORDS` (line 31) is the same
26 fixed streamwise stations for every sequence in either cohort.

- **Wake cohort** (lines 106–112). Every one of the 24 `WAKE_COORDS`
  `(y, z)` tuples × the 10 training parameter sets × `WINDOWS_PER_COORD`
  disjoint temporal windows. The 24 coordinates are hand-curated at lines
  34–39:
  ```python
  WAKE_COORDS = [
      (-71, -1), (-67, -1), (-63, -21), (-59, -17), (-55, 2), (-47, -21),
      (-43, 22), (-31, -21), (-16, 22), (-12, 22), (-8, 18), (0, -1),
      (3, 10), (11, 22), (15, 22), (23, 22), (27, 22), (39, 10),
      (47, -21), (55, 2), (59, 2), (67, 10), (71, -13), (75, -5),
  ]
  ```
- **Random cohort** (lines 126–156). Uniform sampling over
  `y_range = np.arange(y_min, y_max+1, 4)` /
  `z_range = np.arange(z_min, z_max+1, 4)` — the extents come from the source
  file (fallback `-80..80`). One random plan is generated per wake plan
  (line 148), **excluding** the 24 wake tuples (line 152:
  `if (y, z) in WAKE_COORDS: continue`). Seeded with `random.seed(42)`.
- **Parameter split** (lines 247–248). `train_params = ["3p6", "4p4",
  "4p6", "5p2", "6p6", "7p2", "7p8", "8p4", "10p4", "11p4"]` (10 sets),
  `val_params = ["6p4"]` (1 set held out). So `6p4` is the only parameter
  set that never appears in training; `4p4` **is** in training (the
  user's recollection that both were held out is incorrect per code).

### Current on-disk state of `data/`

Byte counts are read directly off `ls -la transformer_neurIPS/data/`:

| file | sequences | shape | bytes on disk |
|---|---:|---|---:|
| `train_40.h5` | 7464 | `(7464, 40, 26, 52)` | 1,397,935,569 B (1.398 GB) |
| `val_40.h5`   | 829  | `(829, 40, 26, 52)`  | 154,979,724 B (155 MB) |
| `train_80.h5` | 3730 | `(3730, 80, 26, 52)` | 1,393,563,365 B (1.394 GB) |
| `val_80.h5`   | 419  | `(419, 80, 26, 52)`  | 156,683,337 B (157 MB) |

### Why 80-frame files are *not* larger than 40-frame files

This is by design, not a sign of truncation. `prepare_data.py` writes the
dataset with HDF5 `compression='gzip'`; both frame lengths use the same
gzip-compressed writer. Because non-overlapping tiling conserves nearly all
of the underlying latent-frame content (halving sequences while doubling
frames), the compressed files land at near-parity rather than one being twice
as large. The tiny 0.3 % skew — `train_80` is ≈4 MB *smaller* than `train_40`,
while `val_80` is ≈1.7 MB *larger* than `val_40` — reflects metadata, chunking,
and compression-overhead effects, not evidence of truncation. **This invariant
is baked into CI** via `tests/test_data_files_size_parity.py`, which hard-fails
if the two files diverge beyond the measured parity budget.

---

## 3. Model

`model_variants.py` implements four transformer variants sharing an
`EMBED_SIZE=256` / `N_LAYERS=6` scaffold:

- `base` — plain causal transformer with learned positional embeddings.
- `swiglu` — SwiGLU FFN in place of the standard MLP.
- `mqa` — multi-query attention.
- `conv` — depthwise-conv positional mixing arm.

Only the positional-embedding module's capacity depends on `NUM_TIME`
(v1.0: `40 * 26 = 1040`; v2.0: `80 * 26 = 2080`). Every other shape is
independent of the sequence length.

The v1.0 promoted checkpoint (`r1_a3b_delta_ar_rollout_best.pt`) is a
**`base`** variant with `PREDICT_DELTA=True` and the AR schedule enabled.

---

## 4. Training driver

`train_production_transformer_deep_dive.py` is the trainer of record.
`Config` defaults (line 105 onward), verbatim:

| Field | v1.0 value | Notes |
|---|---|---|
| `LATENT_DIM` | 47 | latents per (t, x) |
| `NUM_X` | 26 | x-locations per frame |
| `NUM_TIME` | 40 | frames per sequence (v2.0 pins this to 80) |
| `SEQ_LEN` | 1040 | `NUM_X * NUM_TIME` (v2.0: 2080) |
| `EMBED_SIZE` | 256 | |
| `N_HEADS` | 8 | |
| `N_LAYERS` | 6 | |
| `DROPOUT` | 0.01 | |
| `ATTN_IMPL` | `sdpa` | causal by construction |
| `PREDICT_DELTA` | `False` on control; `True` on `r1_a3b_delta_ar` | |
| `BATCH_SIZE` | 64 | physical micro-batch |
| `ACCUMULATION_STEPS` | 8 | effective batch = 512 |
| `LEARNING_RATE` | 1e-3 | peak LR |
| `WARMUP_FRAC` | 0.03 | |
| `LR_FINAL_FRAC` | 0.02 | cosine floor |
| `GRAD_CLIP` | 1.0 | |
| `ADAM_BETAS` | (0.9, 0.95) | |
| `LOSS` | `l2norm` | l2norm \| mse \| huber |
| `MAX_STEPS` | 6000 | optimizer steps (primary clock) |
| `MAX_HOURS` | 12.0 | wall-clock safety net |
| `NOISE_STD` | 5e-4 | gaussian noise on fed-in latents |
| `AR_MODE` | `none` on control; `sched` on `r1_a3b_delta_ar` | |
| `WANDB_PROJECT` | `"runpod_b300_deepdive"` | v2.0 flips this to `"NI_Review"` |

**DataLoader design (the v1.0 lesson).** `TransformerDataset` (line 486)
reads the whole HDF5 once and holds it as one `torch.Tensor`;
`InMemoryBatcher` (line 512) yields shuffled slices of that resident
tensor directly (no `torch.utils.data.DataLoader`, no workers, no
`collate`, no pinned-memory staging); `_preload_to_device` (line 546)
moves the whole tensor once to the training device. Per-batch host↔device
copies are zero. This eliminated the ≈90 % epoch wall-time cost the
standard `DataLoader` pipeline incurred at N=40. At N=80 the resident
combined tensor is ≈1.79 GB float32 (train 1.61 GB + val 0.18 GB), which
comfortably fits M-series unified memory.

**Round-1 arms sweep (v1.0).** Five arms in the `r1_*` prefix; the
promoted arm is `r1_a3b_delta_ar` (base + `PREDICT_DELTA=True` + AR
schedule). See `sweep_deep_dive.py` for the full arm definitions.

---

## 5. v1.0 saved-models inventory

`saved_models/` currently holds **four training runs**, each in a
`_train_best` / `_latest` (± `_rollout_best`) triple, with matching
`*_scripted.pt` TorchScript twins:

| Run | Variant | Kinds present | Ckpt size (best) | Scripted twin | Notes |
|---|---|---|---|---|---|
| `production_base_E256_L6_1785204475` | `base` (E=256, L=6) | `train_best`, `latest` | 22 MB | ✅ | Earliest of the four (Unix-timestamp filename). |
| `production_base_E256_L6` | `base` (E=256, L=6) | `train_best`, `latest` | 22 MB | ✅ | Same architecture, no timestamp in name. |
| `production_mqa_E256_L6` | `mqa` (E=256, L=6) | `train_best`, `latest` | 26 MB | ✅ | Multi-query attention arm. |
| `production_swiglu_E256_L6` | `swiglu` (E=256, L=6) | `train_best`, `latest` | 28 MB | ✅ | SwiGLU FFN arm. |
| **`r1_a3b_delta_ar`** | `base` + `PREDICT_DELTA=True` + AR schedule | `best`, `rollout_best`, `latest` | 55 MB | ❌ | **Current winner** — targeted by `persistence_formal_documentation.py`. |

---

## 6. v1.0 results

Numbers below are quoted verbatim from `tests/reports/model_leaderboard.md`
and `tests/reports/r1_a3b_delta_ar_deep_dive.md`. All measured on `device: cuda`,
metric space **centroid velocity (m/s, AE-decoded)**.

### Leaderboard (`model_leaderboard.md`)

- val data: `val_40.h5`, `subset_ratio=0.1`, **82 sequences**, 3 rollout samples,
  rollout horizon **104 tokens = 4 frames** (leaderboard mode).
- Winner: `r1_a3b_delta_ar_rollout_best.pt`, causal ✅, **4.78 M params**, epoch 2400.
- Single-step centroid MAE: **model 2.47e-4 vs persistence 3.29e-4 → +25.08 %**.
- Rollout (4-frame) centroid MAE: **model 6.08e-4 vs persistence 6.61e-4 → +7.95 %**.
- Checkpoint-recorded at save time: `train_L2=0.005273`, `val_L2=0.003439`,
  `rollout_MSE=3.9e-5` (re-measured on this run: `1.5e-5`).

### Deep dive (`r1_a3b_delta_ar_deep_dive.md`)

- val data: `val_40.h5`, `subset_ratio=0.2`, **8/165 rollout samples**,
  rollout horizon **728 tokens = 28 frames**.
- **Horizon-avg centroid MAE**: **model 9.96e-4 vs persistence 1.363e-3 → +26.92 %**.
- **Horizon-avg latent MSE**: **model 4.105e-5 vs persistence 9.598e-5 → +57.22 %**.
- **Gap accumulation**: frame 1 = **+11.97 %**, frame 28 = **+32.57 %**,
  gap = **−20.60 pts** (the model *widens* its lead as persistence decays).
- **Per component at frame 28**: `vx +30.62 %`, `vy +31.94 %`, `vz +36.94 %` —
  all three velocity components beat persistence, `vz` most.
- Throughput: **513.2 tokens/s**, ≈1.418 s/rollout-sample avg.

These numbers are the floor v2.0 must at least match at the shared horizons 1–28.

---

## 7. Provenance

v1.0 was trained on a **RunPod B300** GPU. Evidence: `Config.WANDB_PROJECT
= "runpod_b300_deepdive"` (trainer line 175) and the deep-dive report's
`device: cuda` line. There is no B200 evidence in the repo.

`encoder_neurIPS/saved_models/simultaneous_training/model_28_best.pt` is
the AE-side variant-28 sweep checkpoint and was **not** promoted for
transformer use; `tests/test_model_vs_baseline.py::load_autoencoder`
picks `model_04_best.pt` in v1.0 (repointed at the scripted GEN3 model
in v2.0 — see §8).

---

## 8. Versioning

**v1.0 (frozen).** Reproducible from `data/train_40.h5` + `data/val_40.h5`
and the checkpoints under `saved_models/`. No further changes land in v1.0.

**v2.0 (planned, 80-frame successor).** The 233.3 ms forecast window of v1.0
cannot substantiate the paper's headline narrative — 300 ms post-reversal
sustain, long-horizon persistence gap, inter-reversal walking. v2.0 doubles
`NUM_TIME` to 80 (566.7 ms forecast @ 120 Hz), warm-starts from
`saved_models/r1_a3b_delta_ar_rollout_best.pt`, and extends
`persistence_formal_documentation.py` to a 68-frame forecast horizon. Concretely:

1. **Trainer refactor** — hard-set `Config.NUM_TIME = 80` (no CLI toggle,
   no 40 fallback), grow the positional-embedding module's capacity to
   `80 * 26 = 2080` tokens, hard-code `train_80.h5` / `val_80.h5`.
2. **Warm-start** — `load_state_dict(state_dict, strict=False)` from
   `r1_a3b_delta_ar_rollout_best.pt` with an explicit missing-keys
   allowlist for length-dependent tensors only.
3. **Device-adaptive regime** — MPS/CPU → `micro_batch=1`,
   `virtual_batch=32`, rainbow console banner (rationale: MPS caching
   allocator + AR-rollout shape churn caused the 88 GB OOM in v1.0).
   CUDA → micro-batching disabled, `torch.bfloat16` AMP,
   `torch.compile(model)`, `cudnn.benchmark=True`, bright-green banner.
4. **AE decoder swap** — `load_autoencoder(device)` is repointed at
   `encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`
   and loaded via `torch.jit.load`. No `create_model_variant(4)`
   construction, no silent `latents[:, :3]` fallback.
5. **Reporting** — `Config.WANDB_PROJECT = "NI_Review"`,
   per-epoch persistence delta printed in green (model wins) / red
   (persistence wins), regenerated `Documentation/persistence_formal/`
   at the 68-frame horizon.

**`encoder_neurIPS/` retirement.** The `encoder_neurIPS/` autoencoder is
**not** used downstream in v2.0. Its `OVERVIEW.md` remains in place as a
historical marker for v1.0's decoder provenance, but v2.0 does not decode
through it, does not train against it, and does not depend on it.
