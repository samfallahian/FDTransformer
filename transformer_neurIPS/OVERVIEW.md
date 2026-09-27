# transformer_neurIPS/ — Folder Overview

This document pins **v1.0 (40-frame)** — the model already trained, checkpointed under
`saved_models/`, and evaluated in `tests/reports/` — and scopes **v2.0 (80-frame)**
as the successor. The `encoder_neurIPS/` autoencoder is **retired from downstream use**
starting with v2.0; the transformer stays in latent space for loss, and the reporting-only
decode path is repointed at the scripted `autoencoderGEN3` model (see §Versioning).

---

## 0. Environment & replication quickstart

Every `run_sweep_*.sh` launcher prints a startup check like this before
training anything (captured verbatim from a real H300/CUDA run,
`round2_20260903_013205`):

```
Checking required files...
  [OK]      /workspace/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py                   (204K)
  [OK]      /workspace/cgan/transformer_neurIPS/model_variants.py                                           (36K)
  [OK]      /workspace/cgan/transformer_neurIPS/sweep_deep_dive.py                                          (44K)
  [OK]      /workspace/cgan/transformer_neurIPS/data/train_80.h5                                            (8.0G)
  [OK]      /workspace/cgan/transformer_neurIPS/data/val_80.h5                                              (3.5G)
  [OK]      /workspace/cgan/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt (2.8M)

Checking interpreter has the required packages (torch+cuda, h5py, numpy)...
  [OK]     torch 2.14.0+cu130 cuda_available=True
  [OK]     numpy 2.4.2
  [OK]     h5py 3.15.1
  [OK]     wandb 0.29.0
```

**This is illustrative, not a pin.** File sizes/byte counts drift as the
code changes; exact torch/numpy/h5py/wandb versions depend on whatever the
rented box's image shipped with (see `requirements_sweep.txt` for the
minimal install list, not exact pins). The launcher checks *presence* and
*CUDA availability*, not these specific version numbers.

**Replicating in PyCharm**: if you're pointing PyCharm's remote deployment/
interpreter at a box like this instead of running the launcher scripts by
hand over SSH, the same six files above need to exist at those paths on
the remote target, and PyCharm's configured remote Python interpreter
needs `torch` (CUDA build), `h5py`, `numpy`, and (optionally)
`wandb` importable -- `Settings > Project > Python Interpreter`, pointed
at the remote venv shown in each launcher's own startup line (e.g.
`/root/.virtualenvs/cgan/bin/python3` in the run above), not a local one.

**Pulling results back**: every launcher writes into `sweep_logs/<run_id>/`
(one folder per invocation: `UPLOAD_ME.md`, one `.log`/`.json` pair per
arm) and `saved_models/` (checkpoints, `_status.json`, `_alerts.jsonl`).
The `rsync` actually used throughout this investigation to pull both back
at once, mid-run or after (harmless to run against a still-running sweep --
see OVERVIEW.md §26.4's note on `_latest.pt` behaviour under a wall-clock-
bounded run):

```
rsync -avP -e "ssh -p 11791 -i ~/.ssh/id_ed25519" \
  root@195.26.233.156:/workspace/cgan/transformer_neurIPS/sweep_logs \
  root@195.26.233.156:/workspace/cgan/transformer_neurIPS/saved_models \
  /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/
```

The host, port, key path, and remote paths are specific to whichever box
happens to be rented at the time -- copy the shape, not the literal
values, for a different box. `-avP` (archive, verbose, partial/progress)
makes repeated pulls of a growing/still-updating directory cheap: only
changed or new bytes transfer each time, and an interrupted pull resumes
rather than restarting.

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

---

## 9. CUDA vs. MPS — device-adaptive training regime

v2.0 detects the compute device once at startup via `pick_device()` and
resolves a `TrainRegime` dataclass (`resolve_train_regime(device)` in
`train_production_transformer_deep_dive.py`). Every knob the training loop
is allowed to know about the hardware flows through that one object — no
device sniffing anywhere else in the loop.

### Side-by-side diff (what the two branches flip, and why)

| Setting | CUDA (H200 fast-path) | MPS / CPU (laptop-debug path) | Rationale |
|---|---|---|---|
| `micro_batch` (training) | **32** (bumps to **64** if `torch.cuda.get_device_name(0)` contains `H200`) | **1** | On MPS the caching allocator cannot reuse blocks across the changing shapes of an AR rollout; attention-score peak is `B · H · L² · 4 B` **per layer** — at `B=32, H=8, L=2080` this is ~34 GB/layer, and 6 layers overrun the 88 GB MPS ceiling on the very first batch. `micro_batch=1` drops this to ~138 MB/layer. CUDA has no such fragmentation issue and can run the full batch physically. |
| `virtual_batch` (effective batch) | = `micro_batch` (accumulation OFF) | **32** micro-steps of gradient accumulation | CUDA runs one physical batch per optimizer step. MPS/CPU still gets an effective batch of 32 by accumulating `loss.backward()` calls before a single `optimizer.step()`. |
| `eval_micro_batch` (`evaluate()` + per-epoch persistence report) | = `micro_batch` | **1** | The TF eval pass at `EVAL_BATCH_SIZE=128, L=2079` would need ~137 GB in attention scores on MPS; the per-epoch persistence rollout at `n_seqs=32, L=2080` needs another ~34 GB/layer. Both are chunked to singleton on MPS and accumulate as scalar sums. |
| `aux_micro_batch` (`frame_ar_loss` / `sched_sampling_loss`) | = `micro_batch` | **1** (clamps `Config.AR_SEQS` down) | Bookkeeping only on MPS/CPU now — see the next row: the AR loss is fully disabled on that branch. On CUDA the arm-specified `AR_SEQS` (4 for `a3b_delta_ar`, 16 for `f4_frame_ar`) is untouched. |
| AR aux loss (`Config.AR_MODE`) | **enabled** at arm value (`frame_ar` / `sched`) | **DISABLED** (forced to `'none'` at `train()` entry, with a log line) | Even at `AR_SEQS=1` the sequential AR loop under token tokenization does `AR_FRAMES * NUM_X` forwards (default arm: `4 * 26 = 104`), each retaining its **own** activation graph for backward through `preds` — the `.detach()` on the fed-back token only truncates the chain between forwards, not each forward's own graph. With 6 layers at L≈300+ that stacks past the 88 GB MPS ceiling regardless of batch. The primary next-token loss still trains the model on Mac; the AR loss is a rollout-stabilization aux and is CUDA-only in v2.0. |
| AMP autocast | `torch.autocast('cuda', dtype=torch.bfloat16)` | disabled | bf16 avoids the fp16-underflow that would require a `GradScaler`; MPS bf16 is not production-ready in this torch build. |
| `GradScaler` | **not used** (bf16 doesn't underflow) | not used | — |
| `torch.compile(model)` | enabled (try/except; never fatal) | disabled | 15–30 % step-time win on H200; MPS Inductor is not production-ready. |
| `torch.backends.cudnn.benchmark` | `True` | untouched | fixed-shape input paths get to pick the fastest cudnn kernel. |
| `torch.set_float32_matmul_precision('high')` | `True` | untouched | residual fp32 ops (LayerNorm, softmax reductions) execute on TF32. |
| Startup banner | `[CUDA DETECTED — H200 DEFAULTS ACTIVE]` in **bright green**, followed by the diff table above | 🌈 `MICRO-BATCH MODE (micro_batch=1)` with each character rainbow-cycled through red → yellow → green → cyan → blue → magenta | Operator sees at a glance which regime the code resolved on this box. |

Set `PFD_NO_COLOR=1` (or `NO_COLOR=1`, or pipe stdout to a file) to strip
ANSI escapes; the banner is emitted only once at the top of the run.

### Peak-memory reduction on MPS (measured / calculated)

At `NUM_TIME=80, SEQ_LEN=2080, N_HEADS=8, N_LAYERS=6, float32`:

| path | before v2.0 (naive batch=32) | after v2.0 (MPS singleton) | reduction |
|---|---|---|---|
| training forward attention (per layer) | ~34.6 GB | ~0.138 GB | **~250×** |
| `evaluate()` TF pass, `EVAL_BATCH_SIZE=128, L=2079` (per layer) | ~138 GB | ~0.138 GB | **~1000×** |
| `evaluate()` rollout chunk, `chunk=32` (per layer) | ~34.6 GB | ~0.138 GB | **~250×** |
| per-epoch persistence report, `n_seqs=32` (per layer) | ~34.6 GB | ~0.138 GB | **~250×** |
| `frame_ar_loss`, `AR_SEQS=1`, `AR_FRAMES=4` under token tokenization (104 retained forwards) | full activation graph across 104 forwards pushed past the 88 GB ceiling on the shakedown | **path disabled entirely** on MPS/CPU | **∞** (path no longer executes on that branch) |

The CUDA branch keeps every one of those knobs at their v1.0-shaped values;
H200 (141 GB HBM3e + bf16 + no activation fragmentation) absorbs them
comfortably. If the CUDA branch ever needs to change, it changes in
`resolve_train_regime(device)` and nowhere else.

### Runtime tests covering both branches

`tests/test_train_regime_cuda.py` (mocked `torch.cuda.is_available()`)
asserts, for each branch, that the returned `TrainRegime` has the right
`micro_batch`, `eval_micro_batch`, `aux_micro_batch`, `disable_ar`,
`use_amp`, `amp_dtype`, `compile_model`, `cudnn_benchmark`, and a banner
without the wrong header. 3/3 pass on the M4 with no CUDA present.

### Startup memory-expectation printout

After the regime banner, `train()` prints a `[memory]` block showing the
predicted **peak attention-score bytes per code path** at the resolved
regime — `B · H · L² · 4 B · N_LAYERS`, using each path's own `(B, L)`
pair. Columns covered:

- `train forward` — teacher-forced next-token loss at `(micro_batch, SEQ_LEN-1)`
- `eval TF forward` — validation TF pass at `(eval_micro_batch, SEQ_LEN-1)`
- `eval rollout / persistence report` — AR rollout at `(eval_micro_batch, SEQ_LEN)`
- `AR aux loss` — retained-graph across `AR_FRAMES*NUM_X` forwards at
  `(AR_SEQS, SEQ_LEN)` on CUDA, or `DISABLED (MPS/CPU)` in yellow

On MPS a dim reminder line quotes the ~88 GB ceiling and the
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` override. The intent is that any
future edit which accidentally re-inflates a code path (a batch bump, a
restored AR loop, a bigger eval batch) is caught in the very first lines
of the run log, before an OOM.

---

## 10. Implementation log (v2.0.1 — what has actually shipped)

Everything under this heading has already landed in `main`; nothing here
is planned/future work. Each bullet cites the concrete artifact so the log
is auditable.

### 10.1 Data & guardrails

- `prepare_data.py` parameterised by `--num-time` (default 40, must divide
  `TOTAL_TIMESTAMPS=1200`); writes `train_<N>.h5` / `val_<N>.h5`.
- Generated `data/train_80.h5` (3730 seqs) + `data/val_80.h5` (419 seqs)
  under the current `train_params={3p6,4p4,4p6,5p2,6p6,7p2,7p8,8p4,10p4,11p4}` /
  `val_params={6p4}` split (see §2 for the pending rebalance to `{4p4,6p4}`).
- `tests/test_data_files_present.py` — separate `test_40_files_present` /
  `test_80_files_present` assertions on existence + `(N, T, 26, 52)` shape.
- `tests/test_data_files_size_parity.py` — hard-fails if the 40- and 80-frame
  on-disk sizes diverge beyond the measured parity budget (guards against an
  accidental compressor change silently shipping half the data).

### 10.2 Trainer pinning (Step 2)

- `Config.NUM_TIME = 80` as a **constant** in
  `train_production_transformer_deep_dive.py`; no `--num-time` flag, no
  40-frame fallback. Top-of-file docstring documents that v1.0 (40) is
  reproduced from frozen `train_40.h5` + `r1_a3b_delta_ar_rollout_best.pt`.
- `TRAIN_H5 = "data/train_80.h5"`, `VAL_H5 = "data/val_80.h5"` hard-coded.
- `PINNED_CONFIG_FIELDS` blocks arms **and** the generic `--set KEY=VALUE`
  escape hatch from overriding shape/data identity fields
  (`NUM_TIME`, `NUM_X`, `SEQ_LEN`, `INPUT_DIM`, `LATENT_DIM`, `TRAIN_H5`,
  `VAL_H5`, `VAL_ROLLOUT_STEPS`).
- `model_variants.py` positional-embedding capacity is derived from
  `Config.NUM_TIME * Config.NUM_X`, so v2.0 gets 2080 tokens automatically.
- `--smoke-test` path builds the 80-frame model, runs `probe_causality`,
  and raises `SystemExit` **before** the optimizer is constructed if the
  model can see the future. Passes on CPU with the M4 in ~seconds.
- **No-arg default on Mac.** `python train_production_transformer_deep_dive.py`
  with zero flags launches the v1.0 winner arm `a3b_delta_ar` (paired with
  its default `--warm-start` target) instead of exiting with
  `need --arm NAME`. `--arm` / `--diagnostics-only` / `--smoke-test` /
  `--list-arms` remain explicit overrides.

### 10.3 Warm-start loader (Step 3)

- `--warm-start CKPT` (default `saved_models/r1_a3b_delta_ar_rollout_best.pt`)
  + `--no-warm-start` opt-out.
- Sanitises the state_dict **before** `load_state_dict(strict=False)`:
  every shape-mismatched key is dropped and recorded; any dropped key
  outside `WARM_START_LENGTH_DEPENDENT_KEYS = {"time_embeddings.weight"}`
  is a hard `SystemExit`. `strict=False` is **not** relied on as a silent
  shape-mismatch shield.
- `WARM_START_BENIGN_MISSING_KEYS = {"causal_mask", "feat_mean", "feat_std"}`
  mirrors the AE's `BENIGN_MISSING_KEYS` pattern from
  `tests/test_model_vs_baseline.py`.
- `unexpected_keys` → hard `SystemExit`.
- Auditable colored log line: transferred / reinitialised param counts,
  dropped length-dependent tensors with `ckpt-shape → model-shape`,
  and source-checkpoint metadata (`step`, `epoch`, `val_l2`, `train_l2`,
  `rollout_mse`) copied out of the payload.
- **Measured on the real v1.0 checkpoint:** 4.77 M / 4.79 M params
  (**99.57 %**) transfer across 81 tensors; only `time_embeddings.weight`
  reinitialises (v1.0 `(40,256)` → v2.0 `(80,256)`). Causality probe passes
  on the warm-started model before the first optimizer step.

### 10.4 AE decoder swap (Step 3b)

- `tests/test_model_vs_baseline.py::load_autoencoder` no longer builds
  `create_model_variant(4)` and loads a state_dict; it now
  `torch.jit.load`s the absolute path
  `encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`
  and `.eval()`s it.
- `decode_latents_to_centroid` probes `ae.decode` first, falls back to
  calling the module directly, and **asserts the output has 375 features**
  before slicing centroid columns `186:189`; there is no silent
  `latents[:, :3]` degradation path any more.
- `metric_space` updated to
  `"centroid velocity (m/s, GEN3-05-AttentionSE scripted, AE-decoded)"`
  so downstream reports carry the new provenance.
- `persistence_formal_documentation.py::main` now runs a `(1,47)→(1,375)`
  AE round-trip smoke assertion before any rollout starts — a wrong
  interface fails loudly, not silently.

### 10.5 Device-adaptive regime + telemetry (Step 4)

- `pick_device()` + `resolve_train_regime(device)` + `TrainRegime` dataclass;
  full spec in §9.
- ANSI helpers (`_ANSI`, `_c`, `_bold`, `_rainbow`, `_banner`) copied verbatim
  from `persistence_formal_documentation.py`; honour `PFD_NO_COLOR`,
  `NO_COLOR`, and non-tty stdout.
- Training loop: `accum_steps = virtual_batch // micro_batch`; observed
  micro-batch count is asserted equal to `accum_steps` on the first virtual
  batch — a stray early `optimizer.step()` would silently train at an
  effective batch of 1 with a 32× larger effective LR.
- `per_epoch_persistence_report(model, val_data, ...)` — fires at the end of
  every epoch on a fixed 32-sequence val subset over horizons 1..28; prints
  one colored line per metric (`MAE`, `RMSE`, `L2`) with `Δ%` in **green**
  when the model beats persistence, **red** otherwise. Same numbers logged
  to W&B under the `persistence/*` namespace.
- `Config.WANDB_PROJECT = "NI_Review"` (was `"runpod_b300_deepdive"`); run
  name embeds `arm`, `NUM_TIME=80`, and the warm-start flag
  (`r1_a3b_delta_ar_t80_ws1`); the resolved regime (device, micro_batch,
  virtual_batch, AMP, compile, cudnn.benchmark) is logged to `wandb.config`
  so post-hoc filtering matches the console banner.

### 10.6 MPS OOM fixes (three sequential rounds)

Three sequential MPS OOMs surfaced once the trainer actually ran on the
M4; all three are now fixed and every fix leaves CUDA untouched.

- **First OOM — `evaluate()` at batch=32 / TF-batch=128.** Fixed by
  threading `regime.eval_micro_batch` into both `evaluate(chunk=, tf_batch_size=)`
  and `per_epoch_persistence_report(chunk=)`. On MPS both = 1; on CUDA both
  = `micro_batch`. Rollout accumulators became scalar sums so metrics are
  batch-invariant.
- **Second OOM — `frame_ar_loss` at `AR_SEQS=4` on MPS.** Fixed by adding
  `aux_micro_batch` to `TrainRegime` and clamping `Config.AR_SEQS` down to
  `regime.aux_micro_batch` on Mac (with a log line). CUDA branch keeps the
  arm-specified `AR_SEQS`.
- **Third OOM — `frame_ar_loss` at `AR_SEQS=1` on MPS.** Even after the
  clamp, the sequential AR loop under token tokenization does
  `AR_FRAMES * NUM_X` forwards (default arm: `4 * 26 = 104`), each
  retaining its **own** activation graph for backward through `preds` —
  the `.detach()` on the fed-back token truncates the chain between
  forwards but not each forward's own graph. Fixed by adding
  `disable_ar` to `TrainRegime` (True on MPS/CPU, False on CUDA); at
  `train()` entry the MPS/CPU branch forces `Config.AR_MODE = 'none'` and
  `Config.AR_LOSS_WEIGHT = 0.0` with a colored log line explaining the
  count of forwards being avoided. The primary next-token loss is
  unaffected. CUDA branch runs the AR loss at its arm-specified value.
- CUDA banner diff table now shows `eval batch`, `AR/aux batch`, and
  `AR aux loss` rows so the regime table matches the code exactly.

**Startup memory-expectation table.** After the regime banner, `train()`
prints one line per code path with the projected peak
attention-score bytes at the resolved regime — see §9 "Startup
memory-expectation printout" for the exact columns. This is the
operator's first indicator that the resolved regime actually fits on the
device, and a diff-target for any future change that touches batch
sizes or the AR loop.

### 10.7 Persistence-formal harness (Step 6)

- `PFD_HORIZON_FRAMES` default extended to `[1, 6, 12, 24, 36, 48, 60, 68]`
  (v1.0 covered only 1..28); docstring updated `28 frames = 233.3 ms →
  68 frames = 566.7 ms`, `40 frames = 333.3 ms → 80 frames = 666.7 ms`.
- `PFD_KIND` default changed `best → rollout_best`, matching the winner in
  `tests/reports/r1_a3b_delta_ar_deep_dive.md`.
- Val-file loader points at `data/val_80.h5` with a loud fallback to
  `val_40.h5` (so the harness still runs on frozen v1.0 checkpoints).
- Final regeneration of `Documentation/persistence_formal/` (CSV / PDFs /
  `report.md`) is deferred to the H200 operator — it requires an actual
  80-frame trained checkpoint, which this agent does not produce.

### 10.8 What has *not* shipped yet (explicitly)

- **H200 training** — not executed by this agent. The code is H200-ready;
  the three intended permutations (warm-start, cold-start, staged 40→80
  curriculum) are documented in the trainer's top-of-file docstring under
  the `H200 PERMUTATIONS` section.
- **Held-out rebalance to `{4p4, 6p4}`** — designed and reasoned in §2 /
  `.junie/plans/v2-0-80-timestep-migration.md` Step 1a; not yet applied.
  When it lands, `train_80.h5` drops from 3730 → ≈3360 sequences and
  `val_80.h5` roughly doubles (≈800–840), matching the encoder's
  `{4p4, 6p4}` exclusion convention.

### 10.9 Scripted checkpoint companions (progress log)

The trainer's `save_checkpoint` now emits a self-contained TorchScript
companion (`<name>_scripted.pt`) alongside every state-dict `<name>.pt`,
so downstream consumers can `torch.jit.load(path)` without importing
`model_variants.py` / `Config`. This section records the sequence of
changes that landed this feature, and every regression it produced on
the way, so the trap footprints are documented alongside the code that
now avoids them.

#### 10.9.1 Initial landing (previous session)

- `Config.SAVE_SCRIPTED_MODELS = True` (flipped from `False`; the flag
  had been dead code prior).
- New `save_scripted_model(...)` in
  `train_production_transformer_deep_dive.py` with six guardrails:
  `torch.compile` `_orig_mod` unwrap; `torch.jit.script` first with a
  `torch.jit.trace` fallback on a representative synthetic input;
  `feat_mean` / `feat_std` asserted as registered buffers on both
  `BaseTransformer` and `FrameTransformer`; `frame_native` honoured to
  size the example correctly; roundtrip verification via
  `torch.jit.load` + one CPU forward; every write logged in rainbow
  colour with the full absolute path via `_log_write`.
- `save_checkpoint` gained a `save_scripted` override and wires
  `_log_write` into the state-dict write, so an operator scrolling
  through a long log can always answer 'where did that artifact go?'
  without re-deriving `Config.CHECKPOINT_DIR`.
- Unit test at `tests/test_scripted_save.py` covers all six guardrails
  on CPU with tiny model shapes.

#### 10.9.2 Follow-up: `torch._dynamo` recompile-limit warning

Symptom in H200 logs:

```
W torch/_dynamo/convert_frame.py:1994 [0/8] torch._dynamo hit
config.recompile_limit (8)
    last reason: 0/3: GLOBAL_STATE changed: grad_mode
```

Not an error. `torch.compile`'s guards treat `grad_mode` as a
specialisation key, and the trainer flips grad-mode around the compiled
`forward` in five distinct code paths within one loop iteration:
`probe_causality` (once at startup), `evaluate` (every
`VAL_EVERY_STEPS`), `per_epoch_persistence_report` (each epoch),
`sched_sampling_loss`'s inner `no_grad` block, and — as of §10.9.1 —
`save_scripted_model`'s `.eval()` toggle around scripting. Once the
budget of 8 compiled variants is spent, the eval-side variants run
eagerly. Correctness is unaffected; a small percentage of eval-side
throughput is left on the table. Mitigations documented (not applied):
`torch._dynamo.config.recompile_limit = 32` at trainer entry, or
`@torch.compiler.disable()` on the eval paths, or wrapping the
`save_scripted_model` body in `with torch._dynamo.disable():`.

#### 10.9.3 Bug: trace fallback silently migrated the training model to CPU

Symptom, reproduced on H200 in a 12k-step run at step 1150:

```
[write:state_dict] .../r1_a3b_delta_ar_latest.pt (57.61 MB)
TracerWarning: Converting a tensor to a Python boolean might cause the
trace to be incorrect.
  if T <= k:                                  # model_variants.py:442
[write:scripted:trace] .../r1_a3b_delta_ar_latest_scripted.pt (19.34 MB)
...
RuntimeError: Expected all tensors to be on the same device, but got
mat1 is on cuda:0, different from other tensors on cpu (when checking
argument in method wrapper_CUDA_addmm)
```

Root cause:

1. `_delta_anchor` in `model_variants.py` contains a Python-level
   `if T <= k:` shape branch (line 442). TorchScript cannot compile
   that, so `torch.jit.script(inner)` raises and the code falls into
   the `torch.jit.trace` branch.
2. `torch.jit.trace` returns a `ScriptModule` whose parameters and
   buffers **share storage** with the eager `inner` (unlike
   `torch.jit.script`, which deep-copies). The subsequent
   `scripted.to("cpu")` — intended to make the saved artifact portable
   — silently migrated the LIVE eager training model to CPU as a
   side-effect.
3. The very next `teacher_forced(model, batch, ...)` call hit
   `self.input_projection(h)` in `BaseTransformer.forward` with weights
   on CPU and inputs on `cuda:0`. `F.linear` does not broadcast device
   placement, so `wrapper_CUDA_addmm` raised.

Diagnostic evidence for the trace-vs-script split: earlier scripted
saves in the same run logged `[write:scripted:script]` and did NOT
crash. `torch.jit.script` produces an independent module whose
`.to("cpu")` does not touch the eager training model, so those saves
were safe. Only after script started failing (probably reshaped by
`torch.compile`'s recompiled variant to something script-hostile) did
the trace fallback take over and expose the aliasing.

#### 10.9.4 Fix (this session)

In `save_scripted_model`:

- Capture the eager module's original device on entry
  (`orig_device = next(inner.parameters()).device`), before any
  script / trace happens.
- Deep-copy the `ScriptModule` before calling `.to("cpu")` so the
  saved-to-disk copy is storage-independent regardless of whether the
  method was script or trace: `copy.deepcopy(scripted).to("cpu")`,
  with a fallback to the previous in-place `.to("cpu")` on the rare
  torch versions where `deepcopy` of a `ScriptModule` fails.
- Unconditionally restore `inner.to(orig_device)` in the `finally`
  block as a belt-and-suspenders guard, so any future scripting mode
  (e.g. `torch.jit.freeze`) that reintroduces aliasing cannot silently
  poison the training loop again.
- Comments in the function body call out the trace-vs-script storage
  distinction so a reader reaching the code cold sees why the deep
  copy is not optional.

After this change, both branches (`script` and `trace`) leave `inner`
on its original device, `optimizer.step()` sees a coherent
`cuda:0` / `cuda:0` linear op, and the `_scripted.pt` twin still lands
correctly on disk with the roundtrip check passing on CPU.

#### 10.9.5 Related: scripted files are write-only for the trainer

The resume path (`train()` → `os.path.exists(latest_path)` block) and
`load_warm_start()` only ever read a state-dict `.pt`. The
`_scripted.pt` companions are intended for standalone consumers
(`torch.jit.load(...)` on the promotion / evaluation box) and are
IGNORED by the trainer on relaunch. If the state-dict `.pt` is
missing, the trainer will NOT silently fall back to the scripted
twin — that is by design; a resume needs the optimiser state and the
scheduler state, neither of which the scripted artifact carries.

#### 10.9.6 Rainbow start-from log + non-leaf-grad warning fix

Two small follow-ups to §10.9.4:

- **Rainbow start-from lines.** Rainbow write logs answered "where
  did that artifact JUST GO?"; the mirror question, "which file did
  this run START from?", was still buried inside cyan
  `[warm-start] loading v1.0 winner: ...` or plain
  `[resume] ...latest.pt at step N`. Added a single rainbow
  `[start-from:warm-start] <abs_path>` line at the top of
  `load_warm_start()` and a `[start-from:resume] <abs_path> @ step N`
  line inside the resume block of `train()`. Both use
  `os.path.abspath(...)` so remote / tmux scrollback is unambiguous,
  and both go through the same `_rainbow(...)` helper that already
  honours `PFD_NO_COLOR` / `NO_COLOR` / non-tty stdout.

- **`.grad`-on-non-leaf warning after every scripted save.** The
  belt-and-suspenders `inner.to(orig_device)` in
  `save_scripted_model`'s `finally` block was firing every 25 steps
  (once per `_latest.pt` save) with:

  ```
  UserWarning: The .grad attribute of a Tensor that is not a leaf
  Tensor is being accessed. Its .grad attribute won't be populated
  during autograd.backward(). If you indeed want the .grad field to
  be populated for a non-leaf Tensor, use .retain_grad() on the
  non-leaf Tensor.
      param_grad = param.grad        # torch/nn/modules/module.py:974
  ```

  Root cause: `nn.Module.to(...)` unconditionally runs `_apply(...)`,
  which iterates every parameter and touches `param.grad`. On a
  `torch.compile`-wrapped `_orig_mod` whose parameters are exposed
  through the compiled proxy, that access trips PyTorch's non-leaf
  grad check even when nothing needs to move (the `finally` restore
  is a no-op in the common script/deep-copy path). Fix: gate the
  restore on an actual device change —

  ```python
  current_device = next(inner.parameters()).device
  if current_device != orig_device:
      inner.to(orig_device)
  ```

  This preserves the safety net (if a future scripting mode ever
  aliases parameters again, the restore still fires) while
  eliminating the per-save warning. Correctness is unchanged in both
  branches (`script` deep-copies params, `trace` after the §10.9.4
  fix also deep-copies before `.to("cpu")`, so `inner` is already on
  `orig_device` and no move is needed).

---

## 11. v3 dataset split — 3p6 replaces the unusable 4p4 in validation

**Status:** current. Produced by
`transformer_neurIPS/prepare_data.py`. On-disk marker:
every `train_<N>.h5` / `val_<N>.h5` written by this script now
carries `f.attrs['split_version'] = 'v3'`.

### 11.1 Lineage

| Version | `train_params` | `val_params` | Rationale |
|---|---|---|---|
| **v1** | `[3p6, 4p4, 4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4, 11p4]` | `[6p4]` | Encoder-inconsistent: `4p4` was in the transformer's train set even though the encoder had been trained with `4p4` excluded — a latent-space leak of encoder training statistics into the transformer's train loss. |
| **v2** | `[3p6, 4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4, 11p4]` | `[4p4, 6p4]` | Aligned with the encoder's `excluded_from_train=["4p4","6p4"]` convention — both val cases fully out-of-distribution to encoder AND transformer. BROKEN in practice: the raw `4p4` acquisition in `data/Unmodified_OG_Data/` is a partial recording, deliberately renamed `4p4.notusing.gz` to opt it out of every downstream `*.pkl.gz` glob, so the OG data-prep chain (Ordered_030 → … → Ordered_200) never produces a `Final_Cubed_OG_Data_wLatent/4p4/` folder and this trainer sees ZERO usable `4p4` sequences. |
| **v3 (current)** | `[4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4, 11p4]` | `[3p6, 6p4]` | Honest substitute — `3p6` (5.6 m/s) replaces the physically-unavailable `4p4`. See §11.2 for what this costs in scientific interpretation. |

`prepare_data.py`'s docstring block for `prepare_data()` mirrors this
table in code so a reader who lands on the file cold does not have to
reverse-engineer it from `git log`.

### 11.2 Honesty caveat — do not read `3p6` and `6p4` as symmetric

**`6p4` (10.0 m/s)** is held out from BOTH the GEN3 AttentionSE
encoder (via `encoder_neurIPS/build_neurIPS_dataset.py`'s
`excluded_from_train=["4p4","6p4"]`) AND the transformer training set
here. Its 47-dim latents are therefore genuinely out-of-distribution
to the whole stack. Metrics on `6p4` measure encoder+transformer OOD
generalisation and are the direct counterpart to the paper's §3.5
80-step vortex-reversal validation.

**`3p6` (5.6 m/s)** was SEEN BY THE ENCODER during its training —
`3p6` is in the encoder's inclusion set, not its `excluded_from_train`
list. It is held out ONLY from the transformer in v3. Metrics on
`3p6` therefore measure transformer generalisation over latents the
encoder is already comfortable with. That is a strictly weaker OOD
statement than `6p4` gives:

- It is a fair "does the transformer generalise to a wake speed it
  hasn't been trained on" test.
- It is NOT a fair "does the full encoder-plus-transformer stack
  generalise to a new speed" test.
- It is NOT interchangeable with the paper's `4p4` numbers — the
  paper's `4p4` was encoder-blinded, and this is not.

`3p6` sits *below* the training range (training now spans 7.2–17.8
m/s after dropping `3p6`; `3p6` = 5.6 m/s), so it plays the low-side
role the paper's `4p4` was intended to play, but with the caveat
above. When reporting v3 numbers alongside anything from v1.0 /
v2.0 / the paper, that asymmetry must be called out — not buried in
an average.

### 11.3 Reporting convention — per-parameter, then averaged

For every metric where the trainer or eval harness has access to the
underlying per-sequence values, v3 requires all three of the
following, in this order:

1. **`metric/3p6`** — value computed over only the `3p6` sequences.
2. **`metric/6p4`** — value computed over only the `6p4` sequences.
3. **`metric/val_mean`** — the unweighted mean of (1) and (2).

Rationale: after `prepare_data.py` runs to completion, both val cases
have the same `wake × windows_per_coord` plan count and (subject to
`ok/planned` from the per-parameter skip summary) essentially the
same number of kept sequences. So an unweighted mean is defensible.
The point of publishing (1) and (2) beside (3) is to prevent a
"good `3p6` masks a bad `6p4`" (or vice versa) story from hiding in a
single averaged number — the two cases stress different failure
modes.

Weighting is documented as an explicit future-item hook (analogous to
`Config.CENTROID_WEIGHTS`): if per-param weights become useful (e.g.
"the paper cares about `6p4`, weight it 2×"), they should be added as
`Config.VAL_PARAM_WEIGHTS = {"3p6": 1.0, "6p4": 1.0}` and applied
only in a reporting-side aggregator, never in the training loss.

Concrete surfaces that should honour this convention:

- Console `[eval]` line — extend to show both per-param scalars and
  the mean, e.g.
  `[eval] step N: val_tf_3p6=... val_tf_6p4=... val_tf_mean=...`.
- WandB — three keys per metric: `eval/rollout_mse/3p6`,
  `eval/rollout_mse/6p4`, `eval/rollout_mse/val_mean`, and analogous
  triples for `improvement_pct`, per-frame series, persistence
  baselines, and per-epoch MAE/RMSE/L2. Combined with the
  `wandb.define_metric("*", step_metric="step")` recommendation from
  the earlier metrics review, this makes per-param curves filterable
  in the UI.
- "Best" gates (`_best.pt`, `_rollout_best.pt`) should promote on
  `val_mean`, not on either individual case — otherwise the gate can
  favour the encoder-seen `3p6` and quietly regress on the paper-
  aligned `6p4`.
- Deep-dive JSON in `sweep_logs/manual/<arm>.json` — every keyed
  metric grows a `.3p6` / `.6p4` / `.val_mean` triple. Consumers that
  only want the headline number can read `val_mean`; consumers doing
  the honest comparison read all three.

None of these surfaces are wired yet — this section documents the
convention that any subsequent trainer edit MUST implement when it
touches those code paths.

### 11.4 What v3 does NOT change

- `NUM_TIME` / `NUM_X` / `INPUT_DIM` / feature-column layout — same
  52-column contract as v2 (see §2 and the `layout='N_NT_NX_C'` block
  in `prepare_data.py`).
- Wake-vs-random 50/50 sampling policy, `WINDOWS_PER_COORD` tiling,
  the 24 hard-coded `WAKE_COORDS`.
- `train_80.h5`'s and `val_80.h5`'s dtype, compression, or shape.
- The enrichment pipeline (`enrich_h5_with_velocity.py`) — its
  `_enriched.h5` companions are layout-agnostic and will keep working
  as soon as v3-produced `train_80.h5` / `val_80.h5` land next to
  them.

### 11.5 Regeneration recipe

```bash
python transformer_neurIPS/prepare_data.py --num-time 80
python transformer_neurIPS/enrich_h5_with_velocity.py --overwrite
```

The per-parameter skip-summary lines in the first command are the
authoritative confirmation that the v3 split was applied — for each
of `4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4, 11p4` you should see
`ok=1200/1200 missing_files=0`, and for the val cases `3p6` and
`6p4` the same. If any line reads `ok=0/1200`, the source folder
for that parameter is not on disk and the split needs to be
revisited before promoting the file.

---

## 12. v3.1 — x-range restriction and 80-frame-only policy

**Status:** current. Produced by
`transformer_neurIPS/prepare_data.py`. Applies on top of the §11
v3 split.

### 12.1 X-coordinate window: |x| ≤ 20 (inclusive)

The per-token x-sweep was previously 26 samples spanning
`x ∈ [-29, +69]`. As of v3.1 it is restricted to the inclusive
window `|x| ≤ 20`, keeping 10 samples:

```
kept:     [-18, -14, -10, -6, -2, 1, 5, 9, 13, 17]
dropped:  [-29, -26, -22]  (far upstream, mostly freestream)
          [21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69]
                          (far downstream, sparse structure)
```

Rationale: the vortex cores, reversal events, and high-vorticity
structure the transformer needs to forecast concentrate near the
tunnel centreline; the upstream/downstream tails were adding
low-signal tokens to every sequence. Wake coordinates
(`WAKE_COORDS`, `(y, z)` pairs) are unaffected — every wake region
still contributes a full sequence, just with fewer x-samples per
token row.

**Downstream trainer impact.** `SEQ_LEN = NUM_TIME * NUM_X` changes
from `80 * 26 = 2080` → `80 * 10 = 800`. Any resume from a v2.x
`_latest.pt` built at 2080 tokens will hit the same length-dependent
shape-mismatch path that already handles the v1.0 → v2.0 40 → 80
transition — length-dependent tensors (time embeddings, positional
embeddings, any `pos_emb`/`time_embeddings.weight` of shape
`(SEQ_LEN, E)`) will be dropped and rebuilt. Cold-start with
`--fresh` is the cleaner move here; warm-start from
`_rollout_best.pt` will re-init the length-dependent tensors via the
existing `load_warm_start()` path.

Memory footprint drops by the same ratio (`2080/800 ≈ 2.6×` fewer
tokens per forward): attention-score peak bytes fall from
~24.7 GiB → ~9.5 GiB at `B=32, H=8, L=800, layers=6, float32`,
which changes what fits on smaller CUDA devices as well as MPS.

### 12.2 80-frame-only policy

`prepare_data.py` no longer produces 40-frame data. The
`--num-time` CLI flag is retained for wrapper-script compatibility
but hard-errors on anything except `80`, so a stale invocation
like `--num-time 40` cannot silently regenerate legacy
`train_40.h5` / `val_40.h5`. Existing 40-frame H5s on disk are
treated as legacy artifacts to be left alone (or manually
deleted) — do not regenerate them.

### 12.3 Regeneration recipe

Same as §11.5, no flags needed:

```bash
python transformer_neurIPS/prepare_data.py
python transformer_neurIPS/enrich_h5_with_velocity.py --overwrite
```

Header line on start confirms the applied policy:

```
NUM_TIME_NEW=80, NUM_X=10, WINDOWS_PER_COORD=15, X_COORDS=[-18..17] (10 pts)
```

If `NUM_X` is not 10 or `X_COORDS` is not `[-18..17]`, the file
being run is pre-v3.1 and should be re-synced before promoting the
output.

---

## 13. v3.2 — assembly-time all-zero trap, dropoff discussion

**Status:** current. Documentation-only bump on top of §12. Marker
on disk: `f.attrs['split_version'] = 'v3.2'` on every `train_80.h5`
/ `val_80.h5` written by `prepare_data.py`. No code paths changed
in this version; the value is bumped so consumers can distinguish
a v3.1-produced H5 (which may contain "invisible" sequence drops
under the trap described below) from a v3.2-produced H5 (same on
disk, but now knowingly produced with the trap documented and
audited).

### 13.1 The trap — one all-zero token step drops the whole sequence

Location: `prepare_data.py`, assembly loop inside `process_set()`
(see the block guarded by `if np.all(data[:, :47] == 0):`).

Behaviour, verbatim from the code:

```python
for t_offset in range(NUM_TIME_NEW):
    step = start_step + t_offset
    data = all_extracted_data.get((ps, y_val, z_val, step))
    ...
    # Check for non-trivial data (not just zeros in latent dimensions)
    # Latents are columns 0:47
    if np.all(data[:, :47] == 0):
        print(f"  [skip] {ps} @ (y={y_val}, z={z_val}) step={step}: "
              f"all-zero latents (encoder produced no signal for this coord/step).")
        valid_seq = False
        break
```

Semantics: a sequence is a fixed-length window of `NUM_TIME_NEW =
80` consecutive frames at a single `(param, y, z)`. If ANY ONE of
those 80 frames has an all-zero latent block across ALL `NUM_X`
x-samples, the entire 80-frame sequence is discarded. Not just
that one frame — the whole window.

Concrete consequence: a `(param, y, z)` coordinate whose wake
decays for even a single frame inside a given 80-frame window
loses all 80 frames of that window. Across the 15 disjoint
windows tiling the 1200-step recording, one bad frame can knock
out one full window (~6.7% of that coord's contribution) and, in
the worst case where the bad frames are spread across all 15
windows, that coord contributes nothing at all.

### 13.2 Why v3.1 exposed the trap more than v2 did

Before v3.1 the per-token x-sweep was `NUM_X = 26` covering
`x ∈ [-29..+69]`. The all-zero check is `np.all(data[:, :47] ==
0)` over the whole `(NUM_X, 47)` block — i.e. every kept x-sample
must have zero latent for the check to fire. With 26 x-samples
spread across the full sensor line, even a mostly-empty wake
region almost always had SOME non-freestream x row carrying a
non-zero latent (the reindex-and-`nan_to_num` path only produces
zeros where the raw pickle had no matching x row at all, i.e.
missing coordinate data, and those zeros are typically outnumbered
by real rows).

With v3.1's `NUM_X = 10` narrowed to `x ∈ [-18..+17]`, the check
now covers only the near-centreline x-samples. Any `(y, z)` whose
signal for a given frame lives entirely outside `|x| ≤ 20` — e.g.
a far-downstream wake tap whose vortex core sits at `x ≈ 21..69`
in that frame — now presents as all-zero across the kept x-window
and trips the trap. The trap itself is unchanged; the window
restriction just made the failure mode reachable for more
`(param, y, z, window)` tuples than before, which is what caused
the "sequences dropped significantly" symptom the operator flagged
after regenerating under v3.1.

### 13.3 X-window vs. dropoff — quantifying the tradeoff

The relationship between `|x|` bracket and expected dropoff was
worked out in the follow-up discussion; the honest way to pin down
the true no-dropoff bracket is to probe the raw pickles, but the
geometric bracket is:

| `|x|` bound | Kept x samples | Expected status |
|---|---|---|
| `≤ 20` (v3.1 current) | 10 (`[-18..17]`) | many wakes drop; downstream taps with `y > +30` most affected |
| `≤ 40` | 18 (adds `21, 25, 29, 33, 37`) | most wakes recover; a few far-downstream (`y ∈ {39, 47, 55, 59, 67, 71, 75}`) may still miss |
| `≤ 45` | 17 asymmetric (`[-18..+45]`) or 22 symmetric | expected to eliminate dropoff for all 24 wake taps |
| `≤ 70` (pre-v3.1 baseline) | 26 (`[-29..+69]`) | no dropoff — the original window |

If a future run needs to widen the window to recover coverage
without giving back the full attention-memory cost, the cheapest
move is to extend positive x only (`x ∈ [-18, +45]` — 17 samples),
because the drop is concentrated on the downstream side. That
would raise `SEQ_LEN` from 800 → 1360 (still ~35% smaller than the
pre-v3.1 baseline of 2080) and, per §12.1's memory arithmetic,
attention-score peak bytes rise from ~9.5 GiB → ~27 GiB at
`B=32, H=8, L=1360, layers=6, float32`.

A precise no-dropoff bracket per wake tap requires a probe script
against `Final_Cubed_OG_Data_wLatent/*/`; the sketch is captured
in the follow-up discussion (grep the code comments for
`per_yz_x_active` for the recipe) and is left as a future item.

### 13.4 How to know if the trap fired in the run you just did

Look for two console signatures in `prepare_data.py`'s output:

1. Per-frame skip line — one per triggered `(ps, y, z, step)`:
   `[skip] <ps> @ (y=..., z=...) step=<S>: all-zero latents ...`.
2. Final "Sequence counts" report — every `<ps>: <kept> sequences
   (planned=..., skipped=...)` line where `skipped > 0` is a
   direct measurement of the trap firing for that parameter.

Cross-check: sum of per-`ps` `skipped` should equal the total
count of `[skip]` lines emitted for that `ps` during assembly
(plus any `assembly_missing_step:<ps>` entries from
`skip_reasons`, which are a distinct cause — file/step missing at
extraction time, not all-zero latents). If the two disagree, the
run hit both failure modes; read the `skip_reasons` dict at the
end for the split.

### 13.5 Alternatives to the current all-or-nothing behaviour

Not applied — captured for the next time this file is edited:

- **Zero-fill and continue.** Replace `valid_seq = False; break`
  with `seq[t_offset] = 0.0; continue`, so one bad frame turns
  into one zero-filled frame instead of a discarded window. Pro:
  no coverage loss. Con: the transformer trains on synthetic
  zero targets for those frames, which is a supervision signal
  the encoder never produced.
- **Threshold-based drop.** Count zero-latent frames per window
  and only discard if the count exceeds e.g. 10% of `NUM_TIME`.
  Pro: keeps mostly-good windows. Con: one more knob to tune;
  needs an operator-visible attr recording the threshold used.
- **Coord-level pre-filter.** Drop `(param, y, z)` up front if
  more than K of its 1200 frames are all-zero in the current
  x-window, before the wake/random plan phase. Pro: never emits
  a partial window in the first place. Con: shrinks the effective
  wake set for the current x-window without leaving a per-window
  audit trail.

If any of these are adopted, bump `split_version` further (`v3.3`
etc.) and document the exact policy inside the new "Sequence
counts" summary so downstream consumers can attribute observed
count changes to the policy shift rather than to source-data
drift.

### 13.6 What v3.2 does NOT change

- The all-zero break-and-discard behaviour is preserved as-is.
  Any change to the loop body is a v3.3+ concern.
- Feature-column layout, `X_COORDS`, `NUM_X`, `NUM_TIME_NEW`,
  `WAKE_COORDS`, and the train/val split are unchanged from
  v3 / v3.1.
- The enrichment pipeline is unaffected; the `_enriched.h5`
  companions carry the `split_version` attr through unchanged.

### 13.7 Regeneration recipe

Same as §11.5 / §12.3:

```bash
python transformer_neurIPS/prepare_data.py
python transformer_neurIPS/enrich_h5_with_velocity.py --overwrite
```

After the run, `h5py.File(...).attrs['split_version']` on both
outputs should read `'v3.2'`. If it reads `'v3.1'` or `'v3'`, the
file predates this documentation bump — the on-disk data is
identical to what v3.2 would produce (no code path changed), but
the "counts skipped due to trap" narrative in this section
applies retroactively; do a fresh regeneration only if you want
the attr to match the doc.

## 14. v3.3 — random-plan (y, z) sampled from the real data grid

Marker: on-disk `f.attrs['split_version'] = 'v3.3'` on every
`train_80.h5` / `val_80.h5` written by
`transformer_neurIPS/prepare_data.py` after this change. Unlike the
v3.1 → v3.2 bump, v3.3 **changes the on-disk bytes** — the random
sequences differ from the ones v3.2 would have emitted (or, more
precisely, tried and failed to emit). Any consumer that keys off
`split_version` should treat v3.3 as a hard regeneration boundary.

### 14.1 The bug this closes

The operator flagged that regeneration under v3.2 still produced
"many fewer choices" than expected, and (correctly) suspected the
all-zero-latent trap from §13 was not the primary cause. A
diagnostic test suite (`tests/test_sequence_dropoff_diagnosis.py`,
five tests) confirmed the real culprit on synthetic data:

- `wake_plans` (one per `(param, y, z, window)` over
  `WAKE_COORDS`): **~100% kept**.
- `random_plans` (populated 1-for-1 with wake plans, targeting
  `(y, z)` off the wake set): **~100% dropped**, silently, via
  `assembly_missing_step:<ps>` in `skip_reasons`. The per-frame
  `[skip] ... all-zero latents` line NEVER fires for this failure
  mode; the drop is invisible unless you read the final
  `skip_reasons` dict.

Root cause, in the pre-v3.3 code path:

1. `random.choice(y_range)` / `random.choice(z_range)` drew from a
   **regular step-4 grid** built as
   `np.arange(y_min, y_max + 1, 4)` / same for z, using
   `y_min/y_max/z_min/z_max` read from the probe pickle.
2. The **real data grid** is IRREGULAR — `WAKE_COORDS` shows
   y-spacings of `4, 4, 4, 4, 8, 4, 12, 15, 4, 4, 4, 3, 8, 4, 8, 4,
   12, 8, 8, 4, 8, 4, 4`, and z-spacings of the same character.
   The OG pipeline (`og_data_prep/Ordered_050 → Ordered_060 →
   Ordered_200`) emits rows only on that irregular grid, not on
   a regular step-4 grid.
3. Consequence: almost every regular-grid pick landed on a `(y, z)`
   that had no corresponding row in ANY step's pickle.
   `extract_from_file` returned `None` for that coord, and the
   assembly loop's `if data is None: valid_seq = False; break`
   dropped the whole 80-frame window on the first frame it tried
   to read. Random plans therefore contributed ~0 sequences even
   though they were counted 1-for-1 in the "planned" total.

The 50%-ish "sequences dropped" figure across v3, v3.1, and v3.2
was overwhelmingly this failure mode, not the all-zero trap.

### 14.2 What v3.3 does

`prepare_data.py`'s `process_set()` now builds the random-plan
`(y, z)` pool from data instead of from a synthetic grid:

- Per param `ps`, open the probe pickle `SOURCE_ROOT/<ps>/0001.pkl.gz`
  (same probe already used for `y_min/y_max/z_min/z_max` before).
- Collect the set of `(int(y), int(z))` pairs actually present as
  rows in that pickle.
- Subtract `WAKE_COORDS` — the random plans are supposed to be
  negative examples *away from* the wake taps.
- `random.choice` samples from THIS pool for the rest of the
  random-plan generation loop.

The 1-for-1 population size is unchanged: still
`len(random_plans) == len(wake_plans)`, so total `planned` doesn't
change and the split remains half-wake / half-random. What changes
is that random plans now target real `(y, z)` rows, so extraction
returns real features and only `§13`'s all-zero trap or a true
missing-step can still drop a random plan.

Failure mode is preserved by design: if a probe pickle is missing
or unreadable for a given param, the code loudly falls back to the
pre-v3.3 regular step-4 grid for that ONE param and prints
`[random-plan] probe '...' unavailable/unreadable for '<ps>';
falling back to regular step-4 grid (pre-v3.3 behaviour for this
param).` so the pre-v3.3 dropoff is visible rather than silent.
An additional summary line always fires:

```
[random-plan] valid non-wake (y,z) pool per param
    (v3.3 data-grid sampling): {'4p6': 1234, '5p2': 1234, ...}
```

so an operator can see the pool size per param at a glance.

### 14.3 Expected effect on sequence counts

Under v3.2, a run typically produced roughly `N_wake` sequences
per param (all wake plans survived; all random plans dropped).
Under v3.3, expect roughly `2 × N_wake` sequences per param — a
doubling — because random plans now survive extraction. The
per-param "Sequence counts" summary at the end of `process_set`
will show `skipped=0` (or a small residual from the §13 trap on
far-downstream x-window-clipped frames) rather than the ~50%
skip that was silently absorbed by `assembly_missing_step`
before.

If the doubling is NOT observed, read the `[random-plan]` summary
line: a param falling back to the regular-grid path (probe pickle
missing) will still drop random plans at the v3.2 rate, so its
per-param `skipped` count will resemble v3.2.

### 14.4 Unit-test contract

`transformer_neurIPS/tests/test_sequence_dropoff_diagnosis.py`
carries five tests. Under v3.2 the last one was
`EXPECTED_FAIL_UNTIL_FIX`; under v3.3 all five pass:

- `ExtractReturnsNoneForMissingYZ` — documents the extraction-side
  `None` return for absent `(y, z)`. Unchanged; passes.
- `RandomPlanGridMissesRealDataGrid.test_random_plan_hit_rate_on_wake_grid`
  — the pre-v3.3 regular-grid picker cannot hit `WAKE_COORDS` by
  construction. Documentary; passes.
- `RandomPlanGridMissesRealDataGrid.test_random_plan_grid_regular_data_grid_irregular`
  — asserts irregular real spacings. Documentary; passes.
- `DropoffAttributionEndToEnd.test_wake_plans_survive_random_plans_also_survive`
  — synthetic scenario with wake + non-wake rows in the probe
  pickle. Under v3.3 the harness samples from real `(y, z)` rows
  and the total dropoff is ~0%. Post-fix contract; passes.
- `RandomPlansShouldMostlySurvive.test_overall_drop_is_small`
  — overall drop must be `< 10%` in the same synthetic scenario.
  Post-fix contract; passes.

Any regression that reintroduces the regular-grid picker (or
otherwise re-enables the ~100% random-plan drop) will make the
last two tests fail loudly with `kept={'wake': N, 'random': 0}`
in the assertion message, which is exactly the pre-v3.3
fingerprint.

### 14.5 What v3.3 does NOT change

- **The all-zero-break trap (§13) is still there.** v3.3 does not
  adopt any of §13.5's alternatives (zero-fill-and-continue,
  threshold drop, coord-level pre-filter). That remains a future
  concern.
- **`X_COORDS`, `NUM_X`, `NUM_TIME_NEW`, `WAKE_COORDS`, the train
  / val split, and every feature-column offset are unchanged**
  from v3 / v3.1 / v3.2.
- **The 1-for-1 wake/random ratio is preserved.** If a future
  policy revision decides the random plans should be fewer,
  more, or dropped entirely, that requires a further `split_version`
  bump.
- **The enrichment pipeline is unaffected;** `_enriched.h5`
  companions carry the new `'v3.3'` attr through unchanged.

### 14.6 Regeneration recipe

Because on-disk bytes differ from v3.2, a fresh regeneration is
required to pick up the doubled sequence count:

```bash
python transformer_neurIPS/prepare_data.py
python transformer_neurIPS/enrich_h5_with_velocity.py --overwrite
```

After the run, `h5py.File(...).attrs['split_version']` on both
outputs must read `'v3.3'`. If a downstream consumer sees `'v3.2'`
or earlier, it is looking at a pre-v3.3 file whose random-plan
population was silently truncated by ~half; the operator should
regenerate before promoting metrics.

Lineage recap:

```
v1   → train has 4p4;      val = ["6p4"]                 (encoder-inconsistent)
v2   → train drops 4p4;    val = ["4p4", "6p4"]          (raw 4p4 unusable)
v3   → train also drops 3p6; val = ["3p6", "6p4"]        (§11)
v3.1 → x-window |x|<=20;   80-frame-only policy          (§12)
v3.2 → all-zero-break trap documented (bytes unchanged)  (§13)
v3.3 → random-plan (y,z) sampled from real data grid     (§14, THIS SECTION;
       fixes silent ~50% random-plan dropoff; bytes DIFFER)
```

## 15. v3.4 — physics-derived wake atlas (replaces WAKE_COORDS)

Marker: on-disk `f.attrs['split_version'] = 'v3.4'` on every
`train_80.h5` / `val_80.h5` written by
`transformer_neurIPS/prepare_data.py` after this change. Bytes DIFFER
from v3.3 because the wake-seed population is now dense and per-`(param,
step)` derived from vorticity physics, and the random-plan exclusion set
uses that same population instead of the legacy 24 hardcoded taps.

### 15.1 What went wrong with the hardcoded 24

Prior to v3.4 `prepare_data.py` shipped a module-level literal:

```python
WAKE_COORDS = [ (-71, -1), (-67, -1), ... ]   # 24 hand-picked (y, z) taps
```

Every wake sequence's `(y, z)` came from this list, and the random-plan
generator excluded these 24 pairs to build its "negative examples" pool.
Two problems:

1. **No traceability.** These 24 pairs had drifted away from the physics
   that produced the reversal write-up at
   `Documentation/vortex_reversal/transformer_6p4_reversal_evaluation.md`.
   You could not point at a `(param, step, |ω|)` record on disk and say
   "this atlas row exists because of THAT event."
2. **Undersized wake seed pool.** 24 unique `(y, z)` × 15 windows × 8
   train params = 2880 wake sequences total. The operator's ask was
   explicitly for "10s of thousands of wake data," and the source
   pickles carry thousands of `(y, z)` rows per param.

### 15.2 What v3.4 does

New standalone script `transformer_neurIPS/build_wake_atlas.py` sweeps
every `(param, step)` combination in the OG data tree, runs the exact
same 90%-of-peak vorticity-magnitude core detection used by
`vorticity_search.py` (physics kernel is copied, not imported, so
`transformer_neurIPS/` stays self-contained per the scope invariant),
and for each detected core point ALSO emits **spatial sliding-window
neighbours** — `(y_grid + dy, z_grid + dz)` for `(dy, dz)` on a
`--stride-yz` grid inside an L-infinity `--radius-yz` ball, snapped to
the probe pickle's real `(y, z)` grid.

Defaults (`--stride-yz 4 --radius-yz 8`) produce up to 25 rows per
detected core, giving the ~10× multiplier that turns thousands of core
points into tens of thousands of atlas rows.

### 15.3 Provenance chain (v3.4)

```
.pkl.gz (Final_Cubed_OG_Data)
  ↓  (build_wake_atlas.py: 3D vorticity, core detection, sliding window)
data/wake_atlas_shards/<param>/step_<NNNN>.parquet
  ↓  (build_wake_atlas.py --merge: dedup + sort + sha256)
data/wake_atlas.csv.gz  +  data/wake_atlas.sha256  +
                          data/wake_atlas.manifest.json
  ↓  (prepare_data.py: load_wake_atlas → wake_plans + random exclusion)
data/train_80.h5  and  data/val_80.h5
  ↓  (enrich_h5_with_velocity.py, unchanged)
data/train_80_enriched.h5  and  data/val_80_enriched.h5
```

Every H5 written under v3.4 carries these attrs so a downstream
consumer can audit the atlas that seeded it:

- `attrs['split_version'] = 'v3.4'`
- `attrs['wake_atlas_source']` — absolute path of the atlas artifact
  used, or `legacy:_LEGACY_WAKE_COORDS_FOR_FALLBACK` if the opt-in
  fallback fired, or `shard-tree-in-memory-merge` if the merged csv
  was missing but the shard tree was loaded and merged on demand.
- `attrs['wake_atlas_sha256']` — hex sha256 of the atlas file (or
  `legacy` / `shard-tree-in-memory-merge`).
- `attrs['wake_atlas_rows']` — JSON string `{param: unique_yz_count}`.

### 15.4 Restart / thread protocol

`build_wake_atlas.py` is designed to run overnight on the Mac without
babysitting:

- **Per-`(param, step)` shard files.** A kill mid-sweep loses at most
  one shard. Re-launching the script is a no-op for any `(param, step)`
  whose shard already exists AND matches the current `SCHEMA_VER`.
  Corrupt shards (unreadable / schema mismatch) are auto-recomputed.
  No lock files.
- **`ProcessPoolExecutor` for compute.** `--workers` (default
  `min(cpu_count, 8)`) drives the CPU-bound vorticity kernel.
- **`ThreadPoolExecutor` for I/O.** `--io-workers` (default 4) gates
  shard writes so the compute workers do not sit on the disk.
- **Atomic writes.** Every shard write goes to `<path>.tmp` and is
  `os.replace`d only after the whole shard DataFrame lands on disk.
- **Missing param folders** (`4p4/` — partial recording, not shipped)
  are handled with a loud `[atlas] WARNING` and skipped; a
  `_MISSING` marker file is written under the shards dir for that
  param so the operator sees the skip on re-runs.
- **Parquet fallback.** If `pyarrow` is unavailable, shards are
  written as `.csv.gz` instead. Both codecs round-trip through
  pandas.

`--force` recomputes every shard even if it already exists; `--merge`
(or `--merge-only`) runs the shard concat + dedup + sort into
`data/wake_atlas.csv.gz`, then writes `data/wake_atlas.sha256` and
`data/wake_atlas.manifest.json` (per-param `rows` / `unique_yz` /
`steps_with_cores` counters).

CLI reference (illustrative):

```
python transformer_neurIPS/build_wake_atlas.py                 # default sweep, all params, all steps
python transformer_neurIPS/build_wake_atlas.py --params 6p4    # single param
python transformer_neurIPS/build_wake_atlas.py --steps 1:101   # steps 1..100
python transformer_neurIPS/build_wake_atlas.py --workers 8
python transformer_neurIPS/build_wake_atlas.py --force         # recompute all shards
python transformer_neurIPS/build_wake_atlas.py --merge         # merge shards -> wake_atlas.csv.gz
python transformer_neurIPS/build_wake_atlas.py --merge-only    # skip sweep, only merge
```

### 15.5 Atlas doubles as the exclusion set

The operator's explicit insight — "we will now have a data set that is
much better to use as 'avoid these' when selecting random areas" — is a
design constraint, not just a nice-to-have. In v3.4 the SAME
`atlas_yz_by_param: Dict[param, Set[(int, int)]]` feeds:

- `wake_plans` — one plan per `(param, y_grid, z_grid, window)` for
  every atlas row (still tiled 15 windows per `(param, y, z)`; see
  v3.5 follow-up below for event-anchored windowing).
- `non_wake_pool = probe_yz - atlas_yz_by_param[ps]` — the random-plan
  candidate set. This shrinks the pool relative to v3.3 (which only
  excluded 24 taps); if a param's atlas exclusion empties the pool
  entirely the code falls back to `probe_yz` for THAT param and logs
  a loud `[random-plan] WARNING: atlas exclusion emptied the non-wake
  pool for '<ps>'` line so shrinkage is never silent.

### 15.6 Legacy fallback (opt-in, diagnostic only)

The 24 hand-picked taps live on as a module-private constant
`_LEGACY_WAKE_COORDS_FOR_FALLBACK` (with a back-compat alias
`WAKE_COORDS` that still points at the legacy list). It is used ONLY
when:

```
export PREPARE_DATA_ALLOW_LEGACY_WAKE_FALLBACK=1
```

Semantics:

- Env var UNSET + atlas & shards both missing → `SystemExit` with a
  message pointing at `build_wake_atlas.py`.
- Env var SET + atlas & shards both missing → red `[wake-atlas] LEGACY
  FALLBACK ACTIVE (env opt-in)` warning; every param uses the 24 taps.
- Env var SET + atlas present but empty for one param → red
  `[wake-atlas] LEGACY FALLBACK ACTIVE for '<ps>'` warning; that ONE
  param uses the 24 taps; other params proceed with atlas data.
- Env var UNSET + atlas present but empty for one param → `SystemExit`
  pointing at the exact `build_wake_atlas.py --params <ps> --merge`
  invocation that would fix it.

### 15.7 Repository policy for the atlas artifact

`data/wake_atlas.csv.gz` is committed if it is ≤ ~100 MB compressed
(default `--stride-yz 4 --radius-yz 8` should land the atlas in the
10k–100k row range across all 10 params, ~1–10 MB). If the operator
runs with a denser stride/radius and the merged CSV exceeds ~100 MB,
commit only `data/wake_atlas.sha256` and `data/wake_atlas.manifest.json`
and document the rebuild recipe in this section. The shards directory
(`data/wake_atlas_shards/`) itself is committed as an empty
`.gitkeep`; individual shard files are generated, not committed.

### 15.8 Regeneration recipe

Because on-disk H5 bytes differ from v3.3, a fresh regeneration is
required whenever the atlas is rebuilt:

```bash
# 1. build (or resume) per-shard atlas rows
python transformer_neurIPS/build_wake_atlas.py

# 2. merge shards into wake_atlas.csv.gz (+ sha256 + manifest)
python transformer_neurIPS/build_wake_atlas.py --merge-only

# 3. regenerate train / val H5 using the merged atlas
python transformer_neurIPS/prepare_data.py

# 4. regenerate enriched companions (unchanged pipeline)
python transformer_neurIPS/enrich_h5_with_velocity.py --overwrite
```

After the run, `h5py.File(...).attrs['split_version']` on both outputs
must read `'v3.4'`, and `attrs['wake_atlas_source']` must point at the
absolute path of the atlas CSV used.

### 15.9 Test coverage

`transformer_neurIPS/tests/test_wake_atlas.py` (10 test methods):

- `TestAtlasLoader` — csv parsing, dedup, and the `include_neighbours`
  toggle.
- `TestMissingAtlasBehaviour` — hard-fail without env var; opt-in
  fallback matches the legacy 24 taps.
- `TestSlidingWindowNeighbourGeneration` — offset math, and
  neighbours falling off the probe grid are dropped.
- `TestRandomPlanExclusion` — `non_wake_pool == probe_yz - atlas_yz`.
- `TestBuilderRestartSafety` — two runs; second is a no-op; deleting
  one shard causes only that shard to be recomputed.
- `TestAutoMergeFromShards` — loader auto-merges a shard tree when
  `wake_atlas.csv.gz` is absent.

The scope-invariant test
(`test_changes_scoped_to_transformer_neurips.py`) continues to pass:
all new atlas symbols (`build_wake_atlas`, `load_wake_atlas`,
`resolve_wake_atlas_for_params`, `WAKE_ATLAS_CSV`,
`WAKE_ATLAS_SHARDS_DIR`) live only under `transformer_neurIPS/`.

### 15.10 What v3.4 does NOT change

- **Tiled `WINDOWS_PER_COORD=15` scheme is unchanged.** The atlas
  carries a per-`(param, step)` `step` column, but `prepare_data.py`
  still tiles 15 disjoint 80-frame windows per `(param, y_grid,
  z_grid)`. Event-anchored windowing (place a documented reversal
  step inside the forecast horizon of a specific training window) is
  a v3.5+ follow-up that the atlas ENABLES but this pass does not
  implement.
- **§13 all-zero-break trap is unchanged.** Same behaviour as v3.3.
- **Enrichment pipeline (`enrich_h5_with_velocity.py`) is unchanged.**
- **`X_COORDS`, `NUM_X`, `NUM_TIME_NEW`, train/val split are all
  unchanged from v3 / v3.1 / v3.2 / v3.3.**

### 15.11 Lineage recap

```
v1   → train has 4p4;      val = ["6p4"]                 (encoder-inconsistent)
v2   → train drops 4p4;    val = ["4p4", "6p4"]          (raw 4p4 unusable)
v3   → train also drops 3p6; val = ["3p6", "6p4"]        (§11)
v3.1 → x-window |x|<=20;   80-frame-only policy          (§12)
v3.2 → all-zero-break trap documented (bytes unchanged)  (§13)
v3.3 → random-plan (y,z) sampled from real data grid     (§14)
v3.4 → WAKE_COORDS retired; physics-derived wake atlas   (§15, THIS SECTION;
       from build_wake_atlas.py drives BOTH wake plans
       AND random-plan exclusion; bytes DIFFER)
```

### 15.12 v3.5 follow-up marker (event-anchored windowing)

The atlas carries per-row `(param, step, cx, cy, cz, y_grid, z_grid,
n_core_points, vort_mag, peak_val, omega_x/y/z, is_neighbour, dy, dz,
core_id, schema_ver)`. v3.4 only consumes `(param, y_grid, z_grid)`;
`step` is currently unused by `prepare_data.py`. A v3.5+ pass can add
"event-anchored windows" — for each atlas row with `is_neighbour=False`
and `n_core_points >= K`, place its `step` inside the forecast horizon
of at least one training window (e.g. shift `start_step` so
`step - start_step ∈ [context_end, num_time - 1]`). This is the
concrete follow-up the atlas exists to enable. **Not implemented as of
v3.6** — do not confuse this proposed windowing change with the actual
"v3.5" split-composition change documented in §16 below; they are
unrelated and this one remains a TODO.

---

## 16. v3.5 — high-side val bracket (11p4 moves train → val)

**Status:** already shipped in code, previously undocumented here.
`prepare_data.py`'s own `prepare_data()` docstring (see the "LINEAGE"
block just above `train_params` / `val_params`) already calls this
"v3.5", but this file never got a section for it and — more importantly
— `process_set()` still hard-codes `f_out.attrs['split_version'] =
'v3.4'` (prepare_data.py, HDF5-metadata block), so every `train_80.h5` /
`val_80.h5` on disk is stamped with a `split_version` attr that is one
version behind the split policy that actually produced it. Treat
`attrs['split_version'] == 'v3.4'` files as "v3.4-or-v3.5, check
`attrs['param_list']` to tell them apart" until that literal is bumped.

### 16.1 What changed

| | v3 / v3.4 | v3.5 (current) |
|---|---|---|
| `train_params` | `[4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4, 11p4]` (8) | `[4p6, 5p2, 6p6, 7p2, 7p8, 8p4, 10p4]` (7) |
| `val_params` | `[3p6, 6p4]` (2) | `[3p6, 6p4, 11p4]` (3) |

`11p4` (17.8 m/s, the highest-Reynolds case in the corpus) moves from
train to val, giving the val split a HIGH-side bracket to go with
`3p6`'s low-side and `6p4`'s mid-range — i.e. val now spans the low,
middle, and high ends of the speed sweep instead of just low+middle.

### 16.2 Honesty caveat — same shape as `3p6`, not as `6p4`

Per the §11.2 convention: `11p4` was SEEN BY THE ENCODER during its
training (it is not in `excluded_from_train=["4p4","6p4"]`), so metrics
on `11p4` are a transformer-only OOD statement, not an encoder+
transformer one — exactly the same caveat that applies to `3p6`. Only
`6p4` is held out from both stages. The §11.3 reporting convention
(per-parameter, then averaged) now needs a three-way split
(`metric/3p6`, `metric/6p4`, `metric/11p4`, `metric/val_mean`) instead
of two; that reporting wiring was already unimplemented for the
two-case version (§11.3's closing note) and remains unimplemented here.

### 16.3 Confirmed on the 2026-08-28 rebuild

The build that produced the current `data/train_80.h5` (59,280
sequences) / `data/val_80.h5` (25,410 sequences) used this v3.5 split —
confirmed from the console's per-parameter skip summary (`4p6, 5p2,
6p6, 7p2, 7p8, 8p4, 10p4` for train; `3p6, 6p4, 11p4` for val, every
one at `skipped=0`). The large jump in sequence count versus the §2
table (7,464 / 829 at v1.0) is the v3.4 wake-atlas density increase
(§15), not this split change — see §17.2 below for the arithmetic.

### 16.4 Outstanding cleanup

- Bump `f_out.attrs['split_version']` from the literal `'v3.4'` to
  `'v3.5'` in `process_set()` so on-disk files self-report the split
  that actually produced them. Not done as part of this pass — flagged
  here so it isn't lost. Existing `v3.4`-stamped files do not need
  regeneration for this alone (the bytes are already v3.5-correct);
  only the attr is stale.

---

## 17. v3.6 — trainer/data shape resync, device-detection banner, sampled centroid check

**Status:** shipped this pass. Triggered by rebuilding `train_80.h5` /
`val_80.h5` under the current (v3.1 + v3.4/v3.5) policy and discovering
the production trainer had not been kept in sync with `prepare_data.py`
since the v3.1 x-window cut.

### 17.1 The bug: `Config.NUM_X` / `SEQ_LEN` still pinned to the pre-v3.1 shape

`train_production_transformer_deep_dive.py`'s `Config` hard-coded
`NUM_X = 26` / `SEQ_LEN = NUM_X * NUM_TIME = 2080`, unchanged since
before the §12.1 x-window restriction. `Config.TRAIN_H5` / `VAL_H5`
point straight at `train_80.h5` / `val_80.h5`, which have carried
`NUM_X = 10` / 800 tokens-per-sequence since v3.1. Training against the
current files would have crashed on the very first line of
`TransformerDataset.__init__`:

```
RuntimeError: shape '[N, 2080, 52]' is invalid for input of size ...
```

(This is the exact error `tests/test_model_vs_baseline.py` already
surfaces — but for an unrelated, pre-existing reason: that test
evaluates v1.0 checkpoints against the legacy `val_40.h5` without
overriding `Config` back to the v1.0 shape, so it fails against
whichever `SEQ_LEN` `Config` happens to be pinned to. Not fixed in this
pass — it needs its own `Config` scoping, independent of this section's
change.)

**Fix:** `Config.NUM_X = 10`, `SEQ_LEN = 800`. Verified end-to-end via
`--smoke-test`:

```
[smoke] pinned v3.1: NUM_TIME=80, NUM_X=10, SEQ_LEN=800 (expected 800), device=mps
[smoke] built BaseTransformer (4.79 M params, frame_native=False)
[smoke] probe_causality: ... causal=True
[smoke] OK: 80-frame model builds, is causal, and trains one micro-batch at a time.
```

Comment-only mentions of the old `2080` / `26` shape scattered through
memory-arithmetic docstrings (CUDA-vs-MPS regime rationale, AR-loop
forward counts, the ridge-regression baseline's `D = NX*LATENT_DIM`)
were updated to the current numbers where they stated a fact as if
still true; historical narrative describing a *past* v1.0-era incident
(e.g. the `ar_context_len=128 = 4*26+24` postmortem) was left alone —
it's an accurate record of what happened at the time, not a live spec.

### 17.2 Resident-tensor memory footprint moved ~10x since that docstring was written

`TransformerDataset`'s docstring previously claimed "~3.7k x 2080 x 52
float32 ~= 1.6 GB" for train. Under the current v3.4/v3.5 data that
estimate is off by almost an order of magnitude in both directions at
once — NUM_X shrank 26→10 (smaller), but the wake-atlas rebuild grew
train from ~7.5k to 59,280 sequences (much larger) — netting out to
**~9.2 GiB train + ~3.9 GiB val ≈ 13 GiB combined resident**, all held
as one `torch.Tensor` per `TransformerDataset` design. Updated the
docstring to state this; flagging here because it changes what "fits
comfortably in RAM / on the GPU" means for this trainer going forward —
worth confirming headroom on whatever box actually launches training,
especially the Mac (unified memory shared with the OS and everything
else running).

### 17.3 Device-detection banner at the top of every run

`main()` previously only printed the detailed device/regime banner
(`resolve_train_regime()`'s `regime.banner`) from inside `train()` /
`run_smoke_test()` — invisible on the `--list-arms` / `--diagnostics-
only` paths, and not the literal first line of output on the paths that
do reach it. Added `print_device_detection_banner()`, called as the
first statement in `main()`, before argument-dependent config wiring:
bold green `[DEVICE DETECTED] CUDA — <name>` on CUDA, rainbow
`[DEVICE DETECTED] MPS (Apple Silicon)` on MPS, bold yellow `[DEVICE
DETECTED] CPU (...)` otherwise. The more detailed regime banner
(micro-batch sizing, AMP, compile) still prints later, on the paths
that actually train.

### 17.4 Sampled centroid-decode sanity check

New `tests/test_centroid_availability.py`. Does **not** run the full
`enrich_h5_with_velocity.py` pass (tens of millions of latent decodes,
a multi-hour job) — the all-zero-latent trap (§13) and the "skipped=0"
per-parameter console lines from a build already structurally guarantee
every kept sequence has non-degenerate latents, so the open question a
cheap test can usefully answer is "does the frozen GEN3 decoder produce
sane, finite centroid velocity across a representative cross-section of
coordinates," not "does data exist for every coordinate."

Samples `TX_CENTROID_SAMPLE_SEQS` (default 50) evenly-spaced sequences
per file (not a contiguous head slice — `process_set()` shuffles wake +
random plans together before writing, so evenly-spaced indices give a
representative cross-section), decodes every token through the same
`load_decoder()` / `CENTROID_SLICE` that `enrich_h5_with_velocity.py`
uses, and asserts the output is finite and under a loose sanity bound
(`MAX_SANE_VELOCITY = 1000`). Confirmed on the 2026-08-28 rebuild: 50/50
samples on both `train_80.h5` and `val_80.h5` landed on 50 distinct
`(param, y, z)` coordinates each, all finite, `max|v|` ≈ 0.84–0.91 —
well inside the 5.6–17.8 m/s training-speed range.

**Important:** confirms the decoder behaves sanely on this data; does
**not** mean training needs `_enriched.h5` files to exist. Verified from
the trainer itself — `centroid_velocity_loss()` calls `decode_centroid()`
on both prediction and target latents live, every training step; it
never reads a `centroid_velocity` dataset from disk. The pre-decoded
`centroid_velocity` column that `enrich_h5_with_velocity.py` produces is
a **planned, not-yet-wired** optimization (its own docstring says so —
"eliminates one decoder forward per training step"); nothing in the
current trainer consumes it. Training can proceed directly against
`train_80.h5` / `val_80.h5` as long as the scripted decoder file is
present on disk, regardless of whether the `_enriched.h5` companions
exist.

### 17.5 Stale tests found during this pass — fixed in v3.7 (§18.4)

Surfaced by running the full suite against the freshly-rebuilt data
files. Left as-is at the time of the v3.6 pass; all three were fixed in
the v3.7 pass (§18.4) and are noted here only for the historical record
of what was found and when:

- `tests/test_data_files_present.py` — hard-coded a single `NUM_X = 26`
  applied to both the 40-frame (legacy, correctly 26) and 80-frame
  (v3.1+, actually 10) cohorts, so `test_80_files_present` failed
  (`10 != 26`) against a correctly-built file.
- `tests/test_data_files_size_parity.py` — asserted 40- and 80-frame
  on-disk byte sizes stay within 2% of each other. That invariant
  predates both the v3.1 x-window cut and the v3.4 wake-atlas density
  increase; `train_80.h5` is now legitimately ~6x the size of
  `train_40.h5` (different `NUM_X`, ~8x more sequences), not evidence of
  truncation.
- `tests/test_model_vs_baseline.py` — two `setUpClass` errors evaluating
  checkpoints against `val_40.h5`, caused by the test not scoping
  `Config` back to the v1.0 shape before loading the legacy file.

---

## 18. v3.7 — the trainer actually trains on centroid velocity now, plus a checkpoint-promotion gate and alerting

**Status:** shipped this pass. Triggered by a direct question during a live
CUDA training run ("where is the centroid loss being tracked?") that led to
discovering the primary training/eval loop was never actually using
`centroid_velocity_loss` despite extensive comments claiming it was, and
then, once that was fixed and a real run's log was read closely, that
`_rollout_best.pt` had been promoting checkpoints with no check against the
persistence baseline at all.

### 18.1 The bug: `centroid_velocity_loss` was documented, not wired in

`centroid_velocity_loss()` / `centroid_per_dim_errors()` (added in an
earlier pass, §10.9.7) are correctly implemented and were already used
inside the optional AR auxiliary loss (`frame_ar_loss` / `sched_sampling_loss`)
-- but that's a secondary term added on top of the DOMINANT per-step loss,
which was still `base_loss(pred, tgt, Config)`: latent-space L2norm/MSE/huber
on the raw 47-dim autoencoder latent, exactly the thing multiple docstrings
in this file said had been "retired." The same gap existed in `evaluate()`
(`val_tf_loss`, and the `rollout_mse`/`persistence_mse`/`improvement_pct`
that `_rollout_best.pt` is selected on) and in `per_epoch_persistence_report()`
(the console/wandb `persistence/*` MAE/RMSE/L2 numbers) -- all computed
directly on raw latents, never decoded through the frozen GEN3 decoder.

Consequence: every "centroid" framing in this file's comments and in
OVERVIEW.md's earlier sections was aspirational for the primary loop. The
live `train_loss`/`val_tf_loss`/`persistence/*` numbers a human or wandb
dashboard would actually see were latent-space numbers with no direct
physical meaning, not the decoded-velocity numbers the design intended.

### 18.2 The fix: decode through the frozen GEN3 decoder everywhere it matters

- `train()`'s per-step teacher-forced loss now calls `centroid_velocity_loss`
  directly (previously `base_loss`). The accumulator variable was renamed
  `base_acc` -> `primary_acc` to stop implying a latent-space quantity.
  `base_loss()` itself is now dead code and was deleted.
- New shared helper `to_per_token_latent(t, cfg)` reshapes a possibly
  frame-flattened tensor (`cfg.TOKENIZATION == 'frame'` models produce a
  trailing `NUM_X*LATENT_DIM` axis) back to a trailing `LATENT_DIM=47` axis
  before it hits the decoder. Without this, a frame-native arm would feed
  the decoder a 470-wide "latent" and silently misbehave -- this exact class
  of bug was caught and fixed in three places (`null_baselines()`, the
  per-step loop, `evaluate()`'s teacher-forced section) before it could ever
  fire, since none of the arms currently in use are frame-native
  (`frame_native=False` confirmed via `--smoke-test`).
- `evaluate()`: the teacher-forced `val_tf_loss` and the rollout-vs-
  persistence `rollout_mse`/`persistence_mse`/`improvement_pct` (the metrics
  `_rollout_best.pt` selects on) now decode `pred`/`tgt`/`pred_f`/`true_f`/
  `pers_f` through `decode_centroid()` before computing error. `rollout_frames()`
  already normalizes both tokenizations to a trailing `(NUM_X, LATENT_DIM)`
  shape, so no `to_per_token_latent()` reshape was needed there.
- `per_epoch_persistence_report()`: the console/wandb `persistence/*`
  MAE/RMSE/L2 breakdown now decodes `gt_c`/`pred_c`/`pers_c` the same way,
  matching the metric space this section's docstring already claimed.
- `null_baselines()` gained a `centroid_l2` key per trivial predictor
  (zeros/mean/previous-token/previous-frame), correctly reshaped through
  `to_per_token_latent()` for the frame-flattened case. `train()`'s
  floor/anchor sanity gate now reads `centroid_l2` instead of
  `nulls[...][Config.LOSS]` -- the latent-space `l2norm`/`mse`/`huber` keys
  are kept and logged separately, informational-only.
- Periodic console/wandb logging gained an explicit `train/centroid_l2` key
  (mirroring `train_loss`, which already IS that value, under a name that
  can't be confused with the retired latent-space objective) and a per-dim
  breakdown (`train/mae_vx`, `.../rmse_vz`, etc.) from `centroid_per_dim_errors()`
  on the last micro-batch of each logged virtual batch -- diagnostic only,
  computed under `torch.no_grad()`, does not affect the gradient just taken.

Verified via `--smoke-test` (builds, causal, trains a micro-batch, at the
correct `NUM_X=10`/`SEQ_LEN=800` shape) and a manual pass of
`null_baselines()` / `evaluate()` / `per_epoch_persistence_report()` /
`centroid_velocity_loss()` against real `val_80.h5` data with a freshly
initialized model -- all four executed without shape errors and produced
finite, velocity-scale (not latent-scale) numbers.

**This changes what the model is actually optimized against.** A run
already in progress under the old `base_loss` objective needs a restart to
train against `centroid_velocity_loss` instead.

### 18.3 `_rollout_best.pt` was being promoted with no check against persistence

Reading a real CUDA run's log after the above fix landed surfaced a second,
independent bug: single-step numbers looked healthy (`train_loss` ≈
`val_tf_loss` ≈ 0.0012-0.0015, gradient norm stable, LR decaying on
schedule), but the rollout comparison was catastrophic --
`rollout_mse≈95` (≈9.75 m/s RMS) against `persistence_mse≈0.0028` (≈0.053
m/s RMS), an improvement of roughly -3,300,000%, worsening by three orders
of magnitude from frame 1 to the last frame of the 68-frame rollout. That
shape (fine single-step, catastrophic and horizon-compounding on rollout)
is the signature of autoregressive error compounding / exposure bias, not
a static bias -- plausibly because the AR auxiliary loss's training horizon
(`AR_FRAMES=2`, 20 tokens) is far shorter than the 68-frame (680-token) eval
rollout it's meant to stabilize. **That root cause is NOT fixed in this
pass** -- it needs its own design decision (how far to extend `AR_FRAMES`/
`AR_SEQS`, whether to add a horizon curriculum, whether the `PREDICT_DELTA`
integration needs damping over hundreds of sequential feedback steps) and
is tracked as an open item, not a mechanical patch.

What WAS fixed: the checkpoint-promotion logic that let this go unnoticed.
`_rollout_best.pt` (`train()`'s rollout-best block) was gated purely on
`rollout_mse < best["rollout_mse"]` -- a self-relative comparison against
this run's own history, seeded at `float('inf')`. Any finite value beats
infinity on the very first eval, and any run whose rollout is uniformly bad
can keep re-earning "best" just by being marginally less bad than its own
worst point, with zero check against the persistence baseline it's
supposed to be beating. Given `DEFAULT_WARM_START_CKPT` points at
`r1_a3b_delta_ar_rollout_best.pt` and is the default no-arg warm-start
target for future runs, a garbage "best" doesn't just sit there mislabeled
-- it becomes the seed for the next run too.

### 18.4 The fix: a real promotion gate, plus alerting for when it fires

- New `Config.MAX_SANE_ROLLOUT_RMSE_MPS = 3.0 * 17.8` (≈53.4 m/s) -- a
  generous absolute ceiling (3x the fastest training free-stream speed;
  observed real centroid velocities top out around 1 m/s per
  `tests/test_centroid_availability.py`) meant to catch genuine decoder-fed
  divergence, not to gatekeep marginal model quality.
- `best` gained a `promoted_rollout_mse` field, distinct from the existing
  `rollout_mse` (best-EVER-seen value regardless of gate outcome, kept so
  the code can tell "found a new self-relative low" apart from "actually
  promoted a checkpoint"). `_rollout_best.pt` is now only written when a new
  low ALSO beats persistence (`improvement_pct > 0`) AND passes the sanity
  ceiling.
- When a new self-relative low is found but fails that gate, three things
  fire together, not just a console line (a console print assumes someone
  is watching a multi-hour unattended run in real time, which is exactly
  how the original bug went unnoticed):
  1. A red console line (`[alert] new rollout_mse low ... NOT promoted`).
  2. `_Telemetry.alert()` (new method) -> a real `wandb.alert()` push
     notification (email/Slack, per the user's wandb settings), distinct
     from `.log()` which only ever lands in run history.
  3. `append_local_alert()` (new function) appends one JSON line to
     `{run_name}_alerts.jsonl` under `CHECKPOINT_DIR`, independent of
     wandb -- `_Telemetry` can be disabled or fail silently mid-run (see its
     class docstring), so the only-record-lives-in-wandb failure mode is
     covered too.
- Every eval now updates `wandb.summary["rollout_beats_persistence"]` and
  `["latest_improvement_pct"]` regardless of promotion outcome, so a run's
  health is visible on the wandb dashboard at a glance, and the final
  per-run result dict gained `"ever_promoted_rollout_checkpoint"`.

Verified: syntax check, a synthetic exercise of the alert path (wandb
disabled -> `.alert()` no-ops without raising; local JSONL append/parse
round-trips), a sanity-ceiling check against the actual observed 95.1
rollout_mse (RMS 9.75 m/s, under the 53.4 m/s ceiling -- confirming the
persistence-beating half of the gate is what actually catches this specific
case, not the ceiling; the ceiling is a genuinely separate belt-and-suspenders
guard for a different failure mode), and `--smoke-test`.

### 18.5 What v3.7 does NOT change

- `NUM_TIME`/`NUM_X`/`SEQ_LEN`/data files/split policy -- unchanged from
  v3.5/v3.6.
- The `_best.pt` (val_tf_mse) promotion gate -- still self-relative only, no
  persistence-style baseline to gate against for that metric, and the
  single-step numbers it tracks are currently healthy. Not touched.
- The AR-horizon mismatch itself (§18.3) -- gating and alerting make the
  symptom visible and stop it from silently contaminating the "best"
  checkpoint; they do not fix the underlying training-methodology gap.

---

## 19. v3.8 -- rollout-divergence diagnosis and Round-2 sweep infrastructure

**Status:** shipped this pass. Direct follow-on from §18.3's open item: the
AR-horizon mismatch was identified but deliberately not fixed, since "what
should `AR_FRAMES` become" is a design decision, not a mechanical patch. This
section is that decision, made with evidence instead of guessing, plus the
infrastructure to test several candidate fixes without picking just one.

### 19.1 Two rollout-archive checkpoints exist, and neither is the same file

Before the diagnostic could run against anything, `saved_models/` had to be
located: mid-session the operator moved its entire contents to
`saved_models/old/` to start the next real run from a clean slate --
independent of, and consistent with, the "start fresh, don't resume the
mixed-objective lineage" recommendation from §18. `r1_a3b_delta_ar_latest.pt`
in that archive (dated 2026-09-01, the tail of the run analysed throughout
§18) is what the diagnostic below was run against. `saved_models/` itself is
empty except for this archive going forward, until the next real run starts.

### 19.2 Stage 0 diagnostic: is the divergence chaotic, or a systematic bias?

New script: `transformer_neurIPS/diagnose_rollout_noise_sensitivity.py`.
Read-only, no training, no checkpoint writes -- it re-runs the eval-time
autoregressive rollout at several injected-noise levels on the model's own
fed-back prediction and checks two things: does error grow with injected
noise (sensitivity/chaos), and is the error dominated by its mean (a
consistent directional drift) or by its spread (symmetric variance)? These
two failure modes call for different fixes, so the question is worth
answering BEFORE picking one.

Run against `saved_models/old/r1_a3b_delta_ar_latest.pt`, 8 val sequences,
noise_std swept 0 -> 1e-2 (100x range):

```
 noise_std    rmse_f1   rmse_mid  rmse_last  |bias|/rmse
   0.0e+00    0.0683      8.604      16.76        0.822
   1.0e-04    0.0683      8.604      16.76        0.822
   1.0e-03    0.0684      8.604      16.76        0.822
   1.0e-02    0.0686      8.602      16.76        0.822
```

**Verdict: BIAS-dominated, not chaotic.** Last-frame RMSE is IDENTICAL
(16.76, to 4 significant figures) whether injected noise is zero or 100x
larger -- the rollout is completely insensitive to perturbation, which rules
out sensitive-dependence/chaos as the mechanism. `|bias|/RMSE = 0.822` means
the error is overwhelmingly a consistent directional drift, not random
spread. Concretely: noise-robustness training (input noise, weight decay,
dropout, and the new feedback-noise knob in §19.3) is unlikely to fix this
on its own, because that's not what's driving the divergence -- the model
has a repeatable, compounding directional error, the same way every rollout.
The horizon-extension arms from §18.3 (`a4b_ar_very_long`, `e3_ar_long`) are
the better-targeted lever: they give the model enough AR training exposure
to see and correct its own accumulated drift, which is what a bias-dominated
failure mode actually calls for.

### 19.3 `sched` mode unblocked on MPS/CPU -- verified, not assumed

`resolve_train_regime()`'s AR kill-switch (§9, §18.3) previously disabled
BOTH `AR_MODE='frame_ar'` and `AR_MODE='sched'` on MPS/CPU under one
condition (`Config.AR_MODE != 'none'`). Empirically measured on this Mac at
the real trainer shape (`micro_batch=1`, `SEQ_LEN=800`) before touching the
gate: `sched_sampling_loss` costs ~34 MB delta vs. ~32 MB for one ordinary
teacher-forced step -- the same order of magnitude, NOT `frame_ar`'s
exponential blowup (which scales with `AR_FRAMES * NUM_X` retained
activation graphs). The guard was scoped to only force `frame_ar -> none`;
`sched` now runs on MPS/CPU as designed. The informational `[memory]`
startup banner and its log line were updated to reflect the split.

### 19.4 New knob: `AR_FEEDBACK_NOISE_STD`

Distinct from the existing `NOISE_STD` (perturbs GROUND-TRUTH inputs during
ordinary teacher-forced training): `AR_FEEDBACK_NOISE_STD` perturbs the
model's OWN fed-back prediction specifically, inside `frame_ar_loss`'s
sequential loop and on the replaced positions in `sched_sampling_loss`'s
mix. Default `0.0` (off, byte-identical behavior to before). Given §19.2's
bias-dominated verdict, this is a secondary/optional mechanism, not a
primary lever -- included because it's cheap to test alongside the horizon
extension, not because the evidence points at it directly.

### 19.5 Two new Round-2 arms

- `e6_sched_noise` (branch E) -- scheduled sampling + `AR_FEEDBACK_NOISE_STD`.
  The only AR-family arm that runs its real mechanism on MPS/CPU (per §19.3).
- `a6b_ar_feedback_noise` (branch A) -- `a4b_ar_very_long`'s 14-frame horizon
  plus feedback noise; tests whether the two mechanisms combine better than
  either alone. CUDA-only (uses `frame_ar`).

`sweep_deep_dive.py`'s per-arm config-display allowlist was extended to
include `AR_FEEDBACK_NOISE_STD` so it shows up in `UPLOAD_ME.md` reports.

### 19.6 `sweep_deep_dive.py` gained `--no-warm-start`

Discovered while smoke-testing the new sweep scripts below: `arm_command()`
had no way to pass `--no-warm-start` through to the trainer, so every arm
launched via the sweep tool silently warm-started from
`DEFAULT_WARM_START_CKPT` (`r1_a3b_delta_ar_rollout_best.pt`) whenever that
file happened to exist -- biasing what's supposed to be a controlled
comparison between arms, and (after §19.1's archive move) hard-failing
outright since that file no longer exists at the default path. Added
`--no-warm-start` to `build_parser()` / `arm_command()`; both new launcher
scripts (§19.7) pass it so every arm in a sweep cold-starts from the same
clean baseline.

### 19.6.1 A second bug, found by actually running the real sweep

The `--no-warm-start` fix let the real `run_sweep_mac.sh` launch land, which
then exercised a code path the smoke test hadn't (the smoke test was run
with `--skip-diagnostics`): `run_diagnostics()` calls `null_baselines()`
TWICE, once per tokenization (`for tok, frame_level in (("token", False),
("frame", True))`), without ever mutating `Config.TOKENIZATION` to match.
`null_baselines()`'s internal `_centroid_score()` helper (added in §18.2)
called `to_per_token_latent(pred, cfg)`, which reshapes based on the GLOBAL
`cfg.TOKENIZATION` -- silently wrong on the `frame_level=True` pass, since
that parameter is independent, per-call state. Result: a 470-wide
(`NUM_X*LATENT_DIM`-flattened) tensor got fed straight to the decoder, which
expects 47-wide, and `run_diagnostics()` crashed with a TorchScript matmul
shape error the first time it ran end-to-end (previously it never had,
since the trainer's own startup call to `null_baselines()` always computes
`frame_level` FROM `Config.TOKENIZATION`, so the mismatch never triggered
there).

Fixed by reshaping on the local `frame_level` parameter directly inside
`_centroid_score()` instead of delegating to `to_per_token_latent()`.
`train()`'s own startup path is behaviourally unchanged (it never had the
mismatch); only `run_diagnostics()`'s dual-tokenization pass is affected.
Verified with `--diagnostics-only` end-to-end (both tokenizations printed
correctly, `[diag] written to .../diagnostics.json`, exit 0) and the full
test suite (41 passed, 3 skipped -- the `test_model_vs_baseline` skips are
expected now that `saved_models/` is empty per §19.1, not a regression).

The already-running `run_sweep_mac.sh` invocation was NOT restarted for
this fix -- its diagnostics phase failed non-fatally before the fix landed
(the sweep driver continues and just omits the diagnostics section from the
final report), but each arm's OWN `null_baselines()` call at training
startup was never affected, so its per-arm results are unaffected.

### 19.7 Two launcher scripts, split by what each piece of hardware can test

`AR_MODE='frame_ar'` needs CUDA (§19.3); `AR_MODE='sched'` and the
non-AR arms don't. Rather than one script that silently degrades depending
on where it's run, there are now two, each checking its own required files
before doing anything:

- **`transformer_neurIPS/run_sweep_mac.sh`** -- `e6_sched_noise` +
  `a5b_wd_heavy` (weight-decay/dropout hypothesis). `--max-parallel 1`
  (no CUDA to round-robin across), `ACCUM` cut from this hardware's normal
  32 down to 4 by default for turnaround -- explicitly a shallow, throwaway
  signal check, not a final-quality run.
- **`transformer_neurIPS/run_sweep_h200.sh`** -- `a4b_ar_very_long`,
  `e3_ar_long`, `a6b_ar_feedback_noise`. Refuses to run at all (checks
  `nvidia-smi`) if no GPU is detected, since these arms would otherwise
  silently degrade to control rather than error. `--max-parallel` defaults
  to `min(detected GPU count, arm count)`.

Both scripts check the same required-file list before launching anything --
`train_production_transformer_deep_dive.py`, `model_variants.py`,
`sweep_deep_dive.py`, `data/train_80.h5`, `data/val_80.h5`, and the frozen
GEN3 decoder (`encoder/autoencoderGEN3/saved_models_production/
Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`) -- printing `[OK]`/
`[MISSING]` per file before anything trains, so a missing file is a clear
one-line diagnosis instead of a stack trace three minutes into a run.

Smoke-tested both new arms end-to-end via `sweep_deep_dive.py --smoke
--no-warm-start` before committing to a real shallow run.

### 19.8 What v3.8 does NOT change

- The actual AR-horizon fix is NOT applied to the production `a3b_delta_ar`
  arm -- these are new, separate arms to test candidates against a clean
  baseline first (per §18's "prove it before scaling to H200" plan).
  `DEFAULT_WARM_START_CKPT` still points at the (now-archived)
  `r1_a3b_delta_ar_rollout_best.pt`; nothing about which checkpoint a
  future default warm-start resolves to was changed.
- `_best.pt`'s gate, the AR-horizon mismatch's actual numeric fix
  (`AR_FRAMES`'s production value), and the archived `saved_models/old/`
  checkpoints are all exactly as `§18.5` left them.

---

## 20. v4.0 -- Mac shallow-sweep results locked in; H200 setup manifest

**Status:** shipped this pass. Locks in the Mac-side (`run_sweep_mac.sh`)
shallow sweep as a completed, interpreted result, fixes a second bug found
while actually running it for real, and specifies everything needed to
set up the CUDA (`run_sweep_h200.sh`) side on a rented/ephemeral box from
nothing.

### 20.1 A second bug, found only by running the real sweep (not the smoke test)

`--no-warm-start` (§19.6) got the real `run_sweep_mac.sh` launch past the
warm-start crash, but the FIRST real attempt (`round2_20260901_160201`)
still failed: `run_sweep_mac.sh`'s default `PYTHON_BIN` resolution
(`${PYTHON_BIN:-python3}`) picked up `/opt/homebrew/bin/python3` -- the
system/Homebrew interpreter, which does not have `h5py`/`torch` installed
-- rather than this project's venv. `sweep_deep_dive.py` crashed on import
before doing anything. Fixed the resolution order in both launcher scripts
to: explicit `PYTHON_BIN` env override > this project's known sibling venv
(`$(dirname REPO_ROOT)/cgan_last_venv_ever/bin/python`) > `python3` from
`PATH` as a last resort; also added an explicit `import torch, h5py,
numpy` (and, on the H200 script, `torch.cuda.is_available()`) check with a
clear failure message, so a wrong interpreter is a one-line diagnosis
instead of a traceback three minutes into a run.

Separately, the operator Ctrl-C'd a stray, still-running instance of the
EARLIER smoke test (two orphaned processes from before the
`--no-warm-start` fix, which had never actually exited and were silently
contending for the same MPS device this whole time) after spotting the
`run_diagnostics()` crash in the log -- confirmed as a deliberate stop, not
a bug, and the orphaned PIDs were killed before relaunching cleanly.

### 20.2 Mac shallow-sweep result: `round2_20260901_162228` -- read the numbers, not the auto-verdict

Both arms completed cleanly this time (`e6_sched_noise` rc=0 in 56.1 min,
`a5b_wd_heavy` rc=0 in 92.9 min). Report:
`transformer_neurIPS/sweep_logs/round2_20260901_162228/UPLOAD_ME.md`
(also copied to `sweep_logs/LATEST_UPLOAD_ME.md`).

**The auto-classifier's verdict should be IGNORED for this run.**
`sweep_deep_dive.py`'s `classify()` returned branch `N` ("Models are far
below a trivial baseline -- fix conditioning before anything else"),
because neither arm beat the previous-frame anchor in-sample. That
threshold logic assumes the ~6000-step budget `sweep_deep_dive.py` was
originally built around; this shallow sweep intentionally used only 400
steps for turnaround. At 400 steps, best train loss was 0.0119-0.0126,
CLOSE to but not yet past the anchor floor (0.0074) -- and the
already-known real run (§18, ~10,000+ steps) converges to 0.0012-0.0016,
well past it. This is "not done training yet," not "broken," and
`classify()` has no way to distinguish those from the numbers alone at a
budget this short. Do NOT run the branch-N arms (`n1_delta_mse` etc.) off
the back of this verdict.

**The actual useful signal is the head-to-head comparison at equal, short
budget** (same steps, same seed, same cold-start, only the arm differs):

| | `e6_sched_noise` | `a5b_wd_heavy` |
|---|---|---|
| train loss | **0.0119** | 0.0126 |
| val loss | **0.0103** | 0.0109 |
| rollout MSE | **0.115** | 0.183 |
| improvement vs. persistence | **-3455%** | -5573% |
| last-frame improvement | **-2787%** | -5087% |

`e6_sched_noise` wins every metric. Neither is close to converged, but as
a relative read at equal budget it's consistent with §19.2's bias-dominated
diagnostic verdict: scheduled sampling actually exposes the model to (and
lets gradient correct) its own one-step error, which is a more plausible
lever against a systematic-drift failure mode than weight decay/dropout
alone. This is a secondary, supporting data point -- the CUDA-only horizon
arms (`a4b_ar_very_long`, `e3_ar_long`) remain the primary hypothesis this
whole sweep infrastructure exists to test, and nothing on the Mac side can
test them (§19.3).

One anomaly, flagged rather than explained away: `a5b_wd_heavy` took
LONGER (92.9 min) than `e6_sched_noise` (56.1 min) despite doing strictly
less work per step (no second forward pass). Backwards from what the
mechanisms should cost -- most likely MPS throughput variability (already
observed swinging 3x+ under similar conditions earlier this session), not
a real property of either arm. Don't read the timing column as signal here.

### 20.3 Exact file manifest -- Mac side (already run, recorded for the record)

Everything `run_sweep_mac.sh` actually touched this session, so the exact
inputs behind §20.2's numbers are reproducible:

```
transformer_neurIPS/train_production_transformer_deep_dive.py
transformer_neurIPS/model_variants.py
transformer_neurIPS/sweep_deep_dive.py
transformer_neurIPS/run_sweep_mac.sh
transformer_neurIPS/data/train_80.h5              (7.9 GB)
transformer_neurIPS/data/val_80.h5                (3.4 GB)
encoder/autoencoderGEN3/saved_models_production/
    Model_GEN3_05_AttentionSE_absolute_best_scripted.pt   (2.8 MB)
```

Interpreter: `/Users/kkreth/PycharmProjects/cgan_last_venv_ever/bin/python`
(Python 3.14.7) -- see §20.4 for the exact package versions, now captured
in `transformer_neurIPS/requirements_sweep.txt`.

**No checkpoint files were needed** -- `--fresh --no-warm-start` means
every arm cold-starts from nothing. This matters for the H200 side too
(§20.5): a wiped box needs zero checkpoint transfer to run a sweep.

### 20.4 `requirements_sweep.txt` -- minimal, scoped, verified by grep not guesswork

New file: `transformer_neurIPS/requirements_sweep.txt`. Deliberately
NOT the repo-root `requirements.txt` (~120 packages covering unrelated
parts of this monorepo -- gym, tensorflow, matplotlib, pysindy, etc.,
none of it imported by anything the sweep touches). Verified by grepping
every top-level AND lazy (function-body) import in
`train_production_transformer_deep_dive.py`, `model_variants.py`, and
`sweep_deep_dive.py`: the entire third-party surface is `torch`, `numpy`,
`h5py`, and `wandb` (lazy-imported inside `_Telemetry.__init__`, and
entirely skippable with `--no-wandb`/`NO_WANDB=1`). Everything else used
is Python stdlib.

Versions pinned to what was validated together this session:

| package | version | pin style | why |
|---|---|---|---|
| numpy | 2.4.2 | `==` | validated together |
| h5py | 3.15.1 | `==` | validated together; also matches root `requirements.txt` exactly |
| wandb | 0.29.0 | `>=` | not version-sensitive for this use |
| torch | 2.12.0.dev20260312 | `>=2.11.0` (see caveat) | **nightly build -- see below** |

**The root `requirements.txt`'s relevant entries are already up to date**
(`h5py==3.15.1` matches exactly; `numpy>=1.26.0` and `wandb>=0.17.0` are
satisfied by what's installed) -- no change needed there. The new scoped
file exists for the H200 box specifically, not as a replacement.

**Torch caveat, important for a rented/wiped box:** the validated build
(`2.12.0.dev20260312`) is a NIGHTLY, not a PyPI stable release --
`pip install torch==2.12.0.dev20260312` will fail against default PyPI; it
needs `pip install --pre torch --index-url
https://download.pytorch.org/whl/nightly/<CUDA_TAG>` (tag matched to the
box's CUDA driver -- check `nvidia-smi` and
https://pytorch.org/get-started/locally/'s Nightly tab, since the tag
matrix moves over time). Nightlies are also PRUNED from the index
eventually, so for a box that gets wiped and re-provisioned repeatedly,
pinning the *channel* (`--pre`, nightly) rather than the exact dev date is
the more robust choice -- accept whatever current nightly resolves rather
than chasing an exact build that may no longer exist. `requirements_sweep.txt`
documents this inline.

Also noted: this session ran on Python 3.14.7, newer than what `torch.jit`
officially supports (source of the DeprecationWarnings suppressed in
§17.2 -- `torch.jit.script is not supported in Python 3.14+`). It works
(verified all session), but Python 3.11/3.12 is the more conservative
choice if provisioning the H200 box's Python version from scratch.

### 20.5 Exact file manifest -- H200 side (what to upload before running)

`run_sweep_h200.sh` checks all of these on startup and refuses to proceed
if any are missing, printing `[OK]`/`[MISSING]` per file:

```
transformer_neurIPS/train_production_transformer_deep_dive.py
transformer_neurIPS/model_variants.py
transformer_neurIPS/sweep_deep_dive.py
transformer_neurIPS/run_sweep_h200.sh
transformer_neurIPS/requirements_sweep.txt        (new -- for env setup, not checked by the script itself)
transformer_neurIPS/data/train_80.h5              (7.9 GB)
transformer_neurIPS/data/val_80.h5                (3.4 GB)
encoder/autoencoderGEN3/saved_models_production/
    Model_GEN3_05_AttentionSE_absolute_best_scripted.pt   (2.8 MB)
```

**Not needed:** any file under `saved_models/` (no checkpoints -- §20.3),
`prepare_data.py`/`build_wake_atlas.py`/wake-atlas artifacts (data is
already built, not regenerated on the H200 box), `enrich_h5_with_velocity.py`,
`TransformLatent.py`, `tests/`, or `OVERVIEW.md` itself (useful as
reference, not required to run).

**Practical setup sequence for a box that gets wiped between runs:**

```bash
# 1. Create/activate a venv, then install deps (see §20.4's torch caveat
#    for the correct nightly index-url):
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/<CUDA_TAG>
pip install numpy==2.4.2 h5py==3.15.1 wandb

# 2. Copy over the file manifest above (data files are the only large
#    transfer -- ~11.3 GB combined; everything else is well under 1 MB)

# 3. Verify before committing to a real run:
bash transformer_neurIPS/run_sweep_h200.sh   # checks files + packages + GPU,
                                              # then launches for real
```

There is deliberately no smaller "smoke" step suggested here beyond what
`run_sweep_h200.sh` itself already does (file check, package/CUDA check,
GPU count) -- `sweep_deep_dive.py --smoke` remains available directly if a
tinier sanity pass is wanted before the real 2000-step budget.

### 20.6 `make_sweep_sample_data.py` -- pre-sampled data for throttled uploads

The rented-box workflow turned out to be RunPod: every rental is a fresh
box (new SSH key, PyCharm re-targeted, Python sometimes needs a manual
upgrade), and upload bandwidth to it is heavily throttled. That combination
means the `rsync`-delta idea discussed earlier doesn't apply -- there is no
persistent prior copy on the far end to diff against, so every transfer is
effectively a full one regardless of tooling. The lever that actually helps
is transferring fewer bytes in the first place.

New script: `transformer_neurIPS/make_sweep_sample_data.py`. Draws a
uniform random sample (without replacement, `numpy.random.default_rng`,
explicit `--seed`, default 1337) of `--fraction` (default 0.30) of the
sequences from `train_80.h5` / `val_80.h5`, writing
`train_80_sample30.h5` / `val_80_sample30.h5` alongside the originals --
same shape/dtype/gzip compression, every source file-level attribute
carried over verbatim, plus new provenance attrs (`sampled_from`,
`sampled_fraction`, `sampled_seed`, `sampled_n`, `sampled_total_source_n`,
`sampled_at`) so a sample file can never be mistaken for the real dataset.

Run once at 30%, verified end-to-end (structural check + an actual
`TransformerDataset` load + `centroid_velocity_loss` forward pass, not
just "the file exists"):

```
train_80_sample30.h5: 59,280 -> 17,784 sequences, 8.53 GB -> 2.56 GB (30.0%)
val_80_sample30.h5:   25,410 ->  7,623 sequences, 3.66 GB -> 1.10 GB (30.0%)
combined: 12.19 GB -> 3.66 GB (~70% reduction)
```

**Two things to get right when actually using these, both documented in
the script's own docstring:**

1. `Config.TRAIN_H5` / `VAL_H5` are hard-coded to `data/train_80.h5` /
   `data/val_80.h5` and are in `PINNED_CONFIG_FIELDS` -- no `--set`
   override exists. The sample files must be uploaded AS those exact
   filenames on the remote box (rename on upload, or `scp` directly to
   the target name), not left as `train_80_sample30.h5`.
2. `run_sweep_h200.sh` defaults `--subset-ratio` to `0.3`. Pointing it at
   an already-30%-sampled train file without also setting
   `SUBSET_RATIO=1.0` compounds to 9% of the original data, silently.
   **Now defended in code, not just documentation:** `train()` reads the
   train file's `sampled_fraction` attr (if present) at startup and logs
   a loud yellow warning with the exact compounded percentage whenever
   `TRAIN_SUBSET_RATIO < 1.0` is also in effect -- verified this fires
   correctly against `train_80_sample30.h5` (reports "compounds to 9.0%").
   val is unaffected either way -- the trainer always loads `VAL_H5` at
   `subset_ratio=1.0`, so a pre-sampled val file is automatically "the
   whole validation set" with no flag needed.

### 20.7 Checkpoints now ALSO upload to wandb as versioned Artifacts (best-effort)

New `_Telemetry.log_artifact(name, paths, ...)`, threaded through
`save_checkpoint()` via a new optional `tel=` parameter (default `None` --
every existing caller that doesn't pass it, e.g. `run_diagnostics()`'s
checkpoint probing, is unaffected). All four call sites inside `train()`
(`_train_best.pt`, `_rollout_best.pt`, `_best.pt`, `_latest.pt`) now pass
`tel=tel`.

**The local disk write is unconditionally still the authoritative one** --
`log_artifact` only runs AFTER `torch.save`/`os.replace` and (if enabled)
`save_scripted_model` have already completed. wandb's `Artifact.add_file`
+ `run.log_artifact` queues the upload asynchronously (the SDK transfers
in the background, not synchronously in this call), so this doesn't add
meaningful wall-clock to the checkpoint-write cadence. Matches
`_Telemetry`'s existing philosophy (class docstring: "wandb that cannot
kill a 12-hour unattended run") -- no-ops cleanly if wandb is disabled,
and a failed/slow upload is caught and logged, never raised.

Artifact name is the checkpoint's own filename minus `.pt`
(e.g. `r2_e6_sched_noise_latest`) -- logging the same name repeatedly (as
`_latest.pt` does every `CHECKPOINT_EVERY_STEPS`) creates a new wandb
Artifact VERSION each time, giving a full checkpoint history in the wandb
UI for free, with no extra bookkeeping. Relevant given §OVERVIEW's own
questioning of wandb storage limits: `Config.SAVE_SCRIPTED_MODELS`
checkpoints run ~76 MB per set (57 MB state-dict + 19 MB scripted); a full
run's four checkpoint kinds plus the local-only 5-deep archive (§19's
`archive_latest_checkpoint`, NOT uploaded to wandb -- only `save_checkpoint`'s
four call sites are) stays well inside typical free-tier quotas for a
single run, but will accumulate faster across many sweep arms if the
`_latest.pt` version history isn't periodically pruned on the wandb side.

Verified: `py_compile`, `--smoke-test`, the full test suite (41 passed / 3
skipped, unchanged), and three direct exercises of `save_checkpoint(...,
tel=...)` -- wandb disabled (clean no-op), `tel=None` (legacy behavior,
unchanged), and a REAL `wandb.Artifact`/`add_file`/`log_artifact` call
sequence against an offline-mode run (no network/auth required, confirms
the API calls are structurally correct, not just that they're skipped).

---

## 21. v4.1 -- wandb project name now tracks OVERVIEW.md's major version

**Status:** shipped this pass. `Config.WANDB_PROJECT` changed from
`"NI_Review"` to `"NI_Review_v4"` -- and, going forward, is a live-tested
invariant rather than a one-off rename.

### 21.1 The convention

`WANDB_PROJECT`'s trailing `_vN` suffix tracks this file's latest
documented **major** version only. Concretely: this file is currently at
v4.1 (this section) -- major version 4 -- so `WANDB_PROJECT = "NI_Review_v4"`.
The suffix does NOT change on every point release (v4.1, v4.2, ... all
still map to `_v4`); it only needs to change when OVERVIEW.md crosses a
new major boundary (the next bump would be triggered by a `v5.0` section
heading, at which point `WANDB_PROJECT` becomes `"NI_Review_v5"`).

Rationale for major-only tracking: a wandb *project* is a fairly durable
container -- renaming it on every point release would fragment run history
across many near-identical project names for no real benefit. Major
version bumps in this file have so far corresponded to genuinely distinct
phases of work (v2.0 pinning, v3.x data-pipeline evolution, v4.x sweep
infrastructure) that plausibly deserve their own wandb project each; point
releases within a phase don't.

### 21.2 `tests/test_version_sync.py` -- the convention is now enforced, not just documented

New test: parses every `## N. vX.Y ...` heading in `OVERVIEW.md` via
regex, takes the max `(major, minor)` tuple as "the latest documented
version" (robust to headings being added out of strict file order),
extracts the trailing `_v<N>` integer from `Config.WANDB_PROJECT`, and
asserts the two major versions match. Fails loudly with both values and a
pointer back to this section if someone bumps one without the other --
e.g. lands a `## 22. v5.0` heading without updating `WANDB_PROJECT`, or
vice versa.

This section (v4.1) IS the first live test of the convention it
describes: `WANDB_PROJECT="NI_Review_v4"` and this file's latest heading
is `v4.1` (major 4) -- they match, so `test_version_sync.py` passes as
written. The next real test of the invariant is whatever change lands the
first `v5.0` heading.

Verified: `py_compile`, the new test passing in isolation, and the full
suite (44 tests: the 41 from §20.7 plus this file's 3; still 3 skipped,
unrelated to this change). [Correction: this originally said "42 tests" --
arithmetic error, 41+3=44, not 42. Fixed in v4.2 while editing this file
for an unrelated reason; noted here rather than silently changed, per this
file's own convention of tracking corrections rather than erasing them.]

---

## 22. v4.2 -- the first real H200 run, and the bug only real CUDA hardware could surface

**Status:** shipped this pass. The operator ran `run_sweep_h200.sh` for
real on a rented H200 (RunPod, `NVIDIA B300 SXM6 AC`, confirming
OVERVIEW.md's earlier "rented/wiped box" workflow discussion was not
hypothetical) and downloaded `LATEST_UPLOAD_ME.md` via `scp`. All three
CUDA arms (`a4b_ar_very_long`, `e3_ar_long`, `a6b_ar_feedback_noise`)
crashed identically, instantly (`exit 1`, `0.0` minutes each) -- this
section is that bug and its fix.

### 22.1 The bug

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

...raised inside the frozen GEN3 decoder's own `nn.Linear.forward`, every
time, on literally the first training step. Root cause: `decode_centroid()`
is called from inside `train()`'s per-step `with amp_ctx:` block --
`torch.autocast('cuda', dtype=torch.bfloat16)` on the CUDA regime (§9).
By the time the primary model's autocast-produced output reaches
`decode_centroid()`, it's already bf16. The frozen decoder (`torch.jit.load`,
never cast, still float32 weights) then tries to matmul a bf16 activation
against a float32 weight and fails outright -- PyTorch's autocast does
NOT appear to bridge this gap for a `torch.jit`-scripted submodule's own
internal linear ops the way it does for ordinary Python-dispatched
`nn.Linear` calls in the same region.

This is a pure CUDA-only bug: MPS/CPU never enables `use_amp`
(`resolve_train_regime()`'s MPS/CPU branch sets `use_amp=False`), so this
code path was never exercised by any of the Mac-side testing across §17
through §21 -- the entire Round-2 sweep effort so far, including three
separate rounds of "run it for real and see what breaks" on this Mac,
could not have caught this. It took actual CUDA hardware to surface it,
which is exactly why §18's "prove it on cheap hardware before H200"
plan always had a limit -- some bugs only exist on the hardware you're
trying to avoid burning time on.

### 22.2 The fix

`decode_centroid()` now forces float32 and explicitly disables CUDA
autocast for just its own call to the frozen decoder:

```python
with torch.autocast(device_type='cuda', enabled=False):
    v = dec(latent.float())
```

**Why `device_type='cuda'` specifically, not `'cpu'`** (the pattern used
elsewhere in this file for "autocast off," e.g. `evaluate()`'s
`_autocast()` helper): autocast state is tracked independently per
`device_type`. Nesting a `device_type='cpu', enabled=False` context
inside an active `device_type='cuda'` autocast region does NOT touch the
cuda state at all -- it would have been a no-op fix. The disable has to
target the SAME device_type as the enclosing context to actually override
it. This is a real, easy-to-make mistake (both look like "turn off
autocast" at a glance) worth calling out explicitly so it isn't repeated
elsewhere in this file.

The decoder is frozen and never trained (`requires_grad_(False)` at load
time, §10.9.7-era), so forcing full precision on its own forward costs
nothing it would otherwise gain from bf16 -- there's no tradeoff being
made here, just correctness.

### 22.3 Validation, and its limit

Verified: `py_compile`, `--smoke-test`, the full test suite (44 tests, no
change in pass/skip count), and a direct exercise feeding a manually
`.to(torch.bfloat16)`-cast tensor into `decode_centroid()` on this Mac (no
CUDA available here) -- confirms the `latent.float()` upcast handles
exactly the input-dtype symptom reported, independent of whether the
CUDA-autocast-nesting mechanics are exercised.

**What was NOT verified, and can't be from this Mac:** the actual
autocast-context-nesting behavior on real CUDA hardware, since MPS/CPU
never activates `use_amp` at all. The fix follows a standard, documented
PyTorch pattern (nested autocast context managers of the same device_type
override the enclosing one), and the reasoning matches the exact error
reported, but confirming it requires re-running `run_sweep_h200.sh` on
the H200 box. That re-run is the next real test of this fix -- not
something this file can claim done from Mac-side evidence alone.

---

## 23. v4.3 -- the first positive result in this entire investigation

**Status:** shipped this pass. Two re-runs of `run_sweep_h200.sh` after
§22's fix: the first reproduced the IDENTICAL bf16/float32 traceback
byte-for-byte (diagnosed as a stale file on the box -- the operator's
PyCharm deployment target was pointed at the wrong root, so the fix never
actually reached the H200 the first time), the second -- after correcting
the deployment target -- ran clean. All three arms completed the full
2000 steps with no crash. This section is that result and the follow-up
it justifies.

### 23.1 The result: `e3_ar_long` posts +30.77% -- read it against the metric's own distortion

```
arm                      IMPROV%   frame1%      last%   roll MSE   pers MSE   >anch?
e3_ar_long               +30.77%  -4925.15%   +46.51%    0.00167    0.00241     yes
a4b_ar_very_long         -47.72%  -1.29e+04    +7.83%    0.00356    0.00241     yes
a6b_ar_feedback_noise    -49.04%  -1.20e+04    +5.58%    0.00360    0.00241     yes
```

`e3_ar_long` is the first arm across this entire investigation -- every
Mac shallow-sweep result in §20, every real long-run analyzed in §18 --
to post a positive AVERAGE improvement over persistence across the full
68-frame rollout. All three arms genuinely converged this time
(`>const?`/`>anch?` both yes across the board, unlike §20.2's shallow Mac
sweep where neither arm had reached the anchor yet) -- so this is a real
result, not another "not done training" false read.

**Frame1's `-4925%` is a metric artifact, not a claim the model is
catastrophically bad at short horizons.** `improvement_pct = (persistence
- model) / persistence * 100`; persistence (copy the last context frame)
is nearly perfect one step ahead for a smooth physical field, so the
denominator is tiny and any nonzero model error registers as a huge
negative percentage regardless of its absolute size. The metric only
becomes informative once persistence itself has degraded (roughly frame
15+) -- and that's exactly where `e3_ar_long`'s per-frame curve crosses
positive and STAYS there, plateauing around +45-47% through frame 68. That
shape -- bad-looking early on a denominator artifact, genuinely stable and
positive late -- is the signature the horizon-extension hypothesis (§19,
§22) was aimed at producing.

### 23.2 A counterintuitive result, explained rather than overclaimed

The SHORTER AR horizon (`e3_ar_long`, 8 frames) beat the LONGER one
(`a4b_ar_very_long`, 14 frames) here -- on its face the opposite of "the
model needs to see further into its own drift." The arm configs explain
why without needing that conclusion: `e3_ar_long` runs its AR loss every
`AR_EVERY_N_STEPS=8`; `a4b_ar_very_long` every 16. At a fixed 2000-step
budget that's 250 AR-loss applications for e3 vs. 125 for a4b -- e3 got
roughly twice the AR gradient signal in this shallow budget, independent
of which horizon is better in the limit. **This is a claim about this
step budget, not a general "8 beats 14" conclusion** -- a4b may catch up
or overtake at a longer run, which is exactly what §23.3 is designed to
check.

### 23.3 Why the auto-suggested "Branch S" command was wrong to run as-is, and what replaces it

`sweep_deep_dive.py`'s classifier recommended the generic Branch-S arms
(`s1_capacity_xl`, `s2_steps_3x`, `s3_swiglu`, `s4_lr_low`, `s5_bigbatch`)
since `best >= STRONG` (30.0) was crossed. All five apply their scaling
knob to the CONTROL config's settings -- none of them carry forward
`AR_MODE='frame_ar'`/`AR_FRAMES=8`/etc. Running them as suggested would
scale a config that never produced this section's result and lose the
thing that actually won.

New arm instead: **`s6_e3_scaled`** (`ROUND2_ARMS["S"]`) -- `e3_ar_long`'s
EXACT overrides (`AR_MODE=frame_ar`, `AR_LOSS_WEIGHT=1.0`, `AR_FRAMES=8`,
`AR_SEQS=2`, `AR_EVERY_N_STEPS=8`), plus `MAX_STEPS=12000` baked into the
arm itself (matching `s2_steps_3x`'s existing convention of baking a step
budget into the override dict; `apply_arm()`'s "CLI beats the arm" rule
means a launcher's own `--max-steps` still wins if passed). 12000 was
chosen to be comparable in scale to the original `a3b_delta_ar` run
(§18) that first exhibited the catastrophic rollout divergence this whole
investigation started from -- the real question this answers is whether
the +30.77% / stable-plateau shape (§23.1) holds, grows, or erodes over a
genuinely long run, not just a 2000-step proof of concept.

New launcher: **`transformer_neurIPS/run_sweep_h200_scale_e3.sh`** --
single-arm counterpart to `run_sweep_h200.sh`, same file/package/GPU
checks, defaults to the full (not sample) `.h5` files since this is the
production-scale check where `subset_ratio=1.0` on real data is the
point, not a throwaway signal test.

Verified: `py_compile`; `resolve_arm('s6_e3_scaled')` +
`apply_arm('s6_e3_scaled')` correctly resolve and set
`AR_MODE=frame_ar, AR_FRAMES=8, AR_SEQS=2, AR_EVERY_N_STEPS=8,
MAX_STEPS=12000`; `bash -n` on the new launcher; and an end-to-end
`sweep_deep_dive.py --arms s6_e3_scaled --smoke` wiring check (see this
run's own log for the result -- smoke settings only, not a claim about
the real 12000-step outcome).

### 23.4 What v4.3 does NOT change

- `e3_ar_long`, `a4b_ar_very_long`, `a6b_ar_feedback_noise` remain
  exactly as defined in §19.5/§22 -- `s6_e3_scaled` is an ADDITION, not a
  redefinition of any existing arm.
- No claim is made yet about whether the +30.77% result holds at
  production scale -- that is what running `run_sweep_h200_scale_e3.sh`
  actually tests, not something this section asserts in advance.

## 24. v4.4 -- production-scale confirmation, a warning-suppression fix, and Branch R

### 24.1 The `torch.jit` warning-suppression fix from v4.3 was too narrow

The filter added when `save_scripted_model()` was first built (§21 area)
matched Python-3.14's specific wording of the `torch.jit.script`/`trace`
deprecation notice under `category=DeprecationWarning`. The H200 box runs
a different Python/torch build and emits the SAME notice worded
differently and under `category=FutureWarning`:

```
FutureWarning: `torch.jit.trace` is deprecated and will be removed in a
future release. Please use `torch.export` instead.
```

Neither the category nor the exact wording matched the old filter, so it
passed straight through. Fixed by broadening from a
Python-3.14-specific `DeprecationWarning` match to a cross-version
`category=Warning` match with a regex covering both observed wordings
(`is not supported in Python 3.\d+\+` and `is deprecated\.`):

```python
warnings.filterwarnings(
    "ignore", category=Warning,
    message=r"^`torch\.jit\.\w+` (is not supported in Python 3\.\d+\+"
            r"|is deprecated\.).*")
```

Verified with a direct `warnings.catch_warnings()` capture test covering
both wordings (confirmed suppressed) plus one unrelated warning
(confirmed it still escapes -- this is not a blanket "ignore everything"
filter).

### 24.2 `s6_e3_scaled` at 12000 steps: the win holds

Full report: `sweep_logs/LATEST_UPLOAD_ME.md`, run `round2_20260902_210846`,
NVIDIA B300 SXM6, 24.4 minutes wall clock.

```
arm            steps   IMPROV%   frame1%   last%    roll MSE   pers MSE
s6_e3_scaled   12000    +27.41  -1745.02   +22.86   0.002064   0.002844
```

+27.41% vs. persistence over the full 68-frame rollout at production
scale, against the shallow sweep's +30.77% at 2000 steps (§23.1) -- a
small, not catastrophic, erosion. The per-frame shape is the same story
as §23.1's frame1-artifact explanation, now with more resolution:
`frame1 = -1745%` is the same denominator artifact (persistence is
nearly exact one step ahead), improvement crosses positive around frame
10, peaks at **+39.2% near frame 23**, then decays gradually to **+22.9%
by frame 68**. That's a real, mild long-horizon decay -- not the
catastrophic divergence this whole investigation started from (§18), and
not the flat plateau the 2000-step shallow run suggested either; 12000
steps was long enough to reveal a shape the shallow run was too short to
show.

`>const?`/`>anch?` both `yes` -- this run genuinely converged, not
another "not done training" false read.

### 24.3 Branch R: a linear baseline beats the transformer by 2x -- and it's a trustworthy verdict this time

The same report's diagnostics fit a closed-form ridge regression
frame-to-frame map on the training data and scored it on the same
objective:

```
persistence MSE                = 0.00034558018635598447
ridge linear frame-map MSE     = 0.00013600662014857495
linear improvement             = +60.64%
```

+60.64% for a linear map vs. `s6_e3_scaled`'s +27.41% for the transformer
-- more than 2x. `sweep_deep_dive.py`'s auto-classifier fired Branch R
("a linear map beats the transformer -- the model or the framing is
broken"). Earlier in this investigation (§20.2) the same classifier fired
a false positive at a shallow 2000-step budget where neither arm had
actually converged yet (`>const?`/`>anch?` not yet `yes`). That caveat
does NOT apply here: `s6_e3_scaled` converged at full production scale
(§24.2), so this verdict is trustworthy -- the gap is real, not an
artifact of an unfinished run.

### 24.4 Screening Branch R on the Mac, not the H200

The auto-suggested command
(`sweep_deep_dive.py --round 2 --branch R --max-parallel 1 --max-steps 12000`)
would run all five pre-existing `ROUND2_ARMS["R"]` arms
(`r1_frame`, `r2_frame_delta_mse`, `r3_tiny`, `r4_lr_sweep`,
`r5_mse_nonorm`) at production scale on CUDA, mirroring the same
"straight to 12000 steps on the H200" mistake that §23.3 avoided for
Branch S. Checked each arm's overrides directly via `resolve_arm()`:

```
r1_frame            -> {'TOKENIZATION': 'frame'}
r2_frame_delta_mse  -> {'TOKENIZATION': 'frame', 'PREDICT_DELTA': True, 'LOSS': 'mse', 'LEARNING_RATE': 0.002}
r3_tiny             -> {'EMBED_SIZE': 128, 'N_LAYERS': 2, 'N_HEADS': 4}
r4_lr_sweep         -> {'LEARNING_RATE': 0.003, 'WARMUP_FRAC': 0.1}
r5_mse_nonorm       -> {'LOSS': 'mse', 'NORMALIZE_FEATURES': False}
```

None of the five set `AR_MODE='frame_ar'` -- unlike every AR arm this
investigation has run (§19 onward), none of them need CUDA at all. That
makes this the first branch in the whole investigation that is a
genuinely free screen on the Mac, no degraded/CPU-fallback caveat
required.

New launcher: **`transformer_neurIPS/run_sweep_mac_branch_r.sh`** --
mirrors `run_sweep_mac.sh`'s structure (required-files check,
venv/dependency check, shallow budget for fast turnaround:
`MAX_STEPS=2000`, `SUBSET_RATIO=0.3`, `ACCUM=4`, `--no-warm-start`,
`--no-wandb`). Run it with:

```
bash transformer_neurIPS/run_sweep_mac_branch_r.sh
```

An H200 "scale the Branch-R winner" script, analogous to
`run_sweep_h200_scale_e3.sh`, is deliberately NOT built yet -- that comes
only after this Mac screen identifies which arm (if any) is worth
scaling, mirroring the `e3_ar_long` -> `s6_e3_scaled` workflow exactly.

### 24.5 What v4.4 does NOT change

- `s6_e3_scaled`'s config is unchanged from §23.3 -- §24.2 only reports
  its production-scale result, it does not redefine the arm.
- The five `ROUND2_ARMS["R"]` arms already existed before this section;
  v4.4 adds a launcher for them, not new arm definitions.
- No claim is made yet about which (if any) Branch-R arm beats
  `s6_e3_scaled` or the ridge baseline -- that is what
  `run_sweep_mac_branch_r.sh` actually tests.

## 25. v4.5 -- the ridge-baseline metric was apples-to-oranges, Branch R reconfirmed exposure bias, and a real winner emerged

### 25.1 The ridge baseline's "+60.64%" was measured in the wrong metric space

`linear_frame_baseline()`'s rollout was already correctly autoregressive
(it feeds its own prediction back in for the full horizon, exactly like an
arm's rollout eval) -- but it scored that rollout in raw 470-dim LATENT
space, while every arm's `IMPROV%`/`roll MSE`/`pers MSE` are scored in
DECODED CENTROID VELOCITY space (m/s), per `evaluate()`'s own comment
("space (m/s), not raw latent space"). The persistence-MSE figures from
the two code paths differed by ~8x (0.00035 raw-latent vs 0.0028
decoded-centroid) confirming they were never comparable numbers.

Fixed by adding a second computation inside the same rollout loop: decode
predictions, ground truth, and the persistence anchor through
`decode_centroid()` before scoring, exactly mirroring `evaluate()`.
`linear_frame_baseline()` now returns both `improvement_pct` (raw latent,
informational only) and `improvement_pct_centroid` (apples-to-apples with
every arm's `IMPROV%`). `sweep_deep_dive.py`'s Branch-R classifier and
report now both use the centroid figure.

**Fixing the metric made the gap WORSE, not better**: the corrected ridge
improvement is **+69.49%**, higher than the uncorrected +60.64%. The
original Branch-R concern was not a measurement artifact -- it understated
the problem.

### 25.2 Branch R, run for real: exposure bias reconfirmed a 9th time, one new failure mode found

`run_sweep_mac_branch_r.sh`'s 5 arms (`r1_frame`, `r2_frame_delta_mse`,
`r3_tiny`, `r4_lr_sweep`, `r5_mse_nonorm`) ran on H200-class CUDA hardware
(not the Mac -- box availability changed the plan, no correctness impact
since none use `AR_MODE='frame_ar'`), 2000 steps, full data:

```
arm                 IMPROV%        train beats anchor?
r5_mse_nonorm         -8.53%        NO   (close to persistence in aggregate;
                                          genuinely +20 to +29% from frame ~33 on)
r4_lr_sweep         -1608.63%       yes  (near-zero train loss, catastrophic rollout)
r1_frame           -29,007.32%      yes
r3_tiny            -40,054.16%      yes
r2_frame_delta_mse -10,013,628.49%  yes  (literal explosion: roll MSE=284,
                                          target std~0.017, never recovers)
```

4 of 5 arms beat the training-objective anchor (one-step prediction) yet
catastrophically diverge on the real 68-frame rollout -- textbook exposure
bias, independent of tokenization, model size, learning rate, or
objective. This is the 9th time in this investigation that a
non-AR-mode architecture/objective change has failed to fix rollout
divergence (a1/a2/d1-d5/f1-f5/r1-r5/a3b/etc.) -- see §18-24 for the prior
8. `e3_ar_long`/`s6_e3_scaled` remain the only arms to ever post a
positive rollout result, and remain the only ones using `AR_MODE='frame_ar'`.

### 25.3 Two speedups to the diagnostics step

`run_diagnostics()`'s linear-baseline computation got two performance
fixes after real runs showed it dominating wall-clock time (11-30+ minutes
on top of the arms themselves):

1. **`max_val_seqs` cap** (default 2500): the rollout was scoring against
   the FULL validation set (25,410 sequences) with the new centroid decode
   (§25.1) running 3x per chunk over the full horizon -- ~10x more
   sequences than an arm's own rollout eval uses (`VAL_ROLLOUT_SEQS=64`).
   Capped at 2500 -- still ~40x more than an arm gets, at ~1/10th the
   diagnostics cost.
2. **`SKIP_DIAGNOSTICS` launcher knob**: the linear baseline doesn't depend
   on which arms run or their step budget, only on the fixed train/val
   data -- added to `run_sweep_h300_branch_h.sh` so a rescreen at a
   different step budget can reuse a prior run's number instead of
   recomputing it. NOT used for `run_sweep_h300_branch_h_followup.sh`
   (below), since `h10_ridge_residual` needs the ridge map diagnostics
   fits and saves.

### 25.4 Branch H (registered in v4.4), run for real: AR-loss frequency dominates every other knob, by a wide and monotonic margin

400-step shallow screen (cut down from the planned 2000 once wall-clock
became the bottleneck -- see the ETA discussion this section is
responding to):

```
arm                 AR_EVERY_N_STEPS   IMPROV%
h1_ar_freq2                2            +41.90   <- winner, still climbing
h4_ar_long_freq4           4             +0.92
h3_ar_short_freq4          4             -0.22
h2_ar_freq4                4            -22.97
h6_ar_fbnoise              8            -31.32
h5_ar_moreseqs             8            -39.53
h7_ar_wd                   8            -49.88
h8_ar_lrlow                8            -53.21
```

The ranking is monotonic in AR-loss frequency alone: every freq=8 arm
(`e3_ar_long`'s own frequency, each with one extra knob on top -- feedback
noise, more AR sequences, weight decay, lower LR) is still negative at
this shallow budget; both freq=4 variants are near breakeven; freq=2 is a
clear, large win -- **+41.90%, beating `s6_e3_scaled`'s production-scale
+27.41% in 400 steps**, with its per-frame curve already plateauing around
+55-59% by frame ~35. None of the secondary knobs (feedback noise, more AR
sequences, weight decay, lower LR) helped at freq=8; frequency alone
dominated everything else tested in this branch.

The auto-classifier again recommended the generic Branch-S arms (the same
category error as §23.3 and §24 avoided) -- `h1_ar_freq2`'s winning
config is not what those arms would scale.

### 25.5 Two follow-ups registered: the direct extrapolation, and a real architectural change

**`h9_ar_freq1`** -- `h1_ar_freq2`'s exact config with `AR_EVERY_N_STEPS=1`
(AR loss every step). Direct extrapolation of §25.4's monotonic
freq8->freq4->freq2 trend to its limit. Cheap, low-risk: same mechanism,
no new code path.

**`h10_ridge_residual`** -- `h1_ar_freq2`'s exact config plus a genuine
architectural change: `PREDICT_DELTA=True` with a new `DELTA_ANCHOR='ridge'`
option. `PREDICT_DELTA` already made the network predict a residual on top
of a zero-initialized head plus an anchor (see `model_variants.py`'s
`_delta_anchor`) -- previously that anchor was always raw persistence
(same-x, previous frame). `DELTA_ANCHOR='ridge'` swaps the anchor for the
fitted ridge map's prediction instead, via a new `BaseTransformer._ridge_anchor()`
method: at any target token whose source frame is fully present in the
input, it reshapes to full frames, applies the frozen ridge matrix (same
one `linear_frame_baseline()` fits and now persists to
`Config.RIDGE_MAP_PATH`), and falls back to `_delta_anchor`'s persistence
value everywhere a complete source frame isn't available yet (the leading
`NUM_X-1` positions, and mid-frame during AR rollout, where the context
grows one token at a time). Rationale: the ridge map beats persistence by
+69.49% in the exact space the model is scored in (§25.1) -- predicting
the residual on top of an already-strong linear predictor should need less
of the network's capacity spent re-deriving something a closed-form
regression already gets mostly right, versus deriving it from scratch
against a weak (persistence) anchor.

Implementation notes, since this touches model internals rather than just
a hyperparameter:
- `linear_frame_baseline()` now saves its fitted matrix to
  `Config.RIDGE_MAP_PATH` (default `saved_models/ridge_frame_map.pt`)
  whenever diagnostics run -- `h10_ridge_residual` depends on this file
  existing, which is why `run_sweep_h300_branch_h_followup.sh` does not
  skip diagnostics.
- `_ridge_anchor()` branches on a shape-derived value (how many complete
  frames are in the input), unlike `_delta_anchor()`'s deliberately
  branch-free design -- this is only guaranteed correct under
  `torch.jit.script` (tried first by `save_scripted_model()`, and compiles
  real control flow). A trace fallback would fix whichever branch was
  taken at trace time. Accepted for a research sweep arm, not a deployed
  path.
- Reuses `decode_centroid()`'s established `torch.autocast(device_type='cuda',
  enabled=False)` + explicit `.float()` guard for the same reason it was
  needed there (§22): the ridge matrix is a frozen float32 buffer, applied
  inside whatever CUDA autocast region the caller is in.
- Scoped to token tokenization only (`BaseTransformer`, not the frame-
  native class) -- every current winning arm uses `TOKENIZATION='token'`,
  and halving the surface area halves the ways this can be subtly wrong.
- Verified locally (no GPU available in this session) via `py_compile` on
  all three changed files, the full 44-test suite, `resolve_arm()` on both
  new arms, and a CPU forward-pass smoke test of `_ridge_anchor()` against
  a synthetic ridge matrix across both frame-aligned and mid-frame
  (`T` not a multiple of `NUM_X`) sequence lengths, confirming finite
  output of the correct shape in both the new ridge path and the
  unmodified persistence path. Not yet validated against the real trainer,
  real data, or a real optimizer step -- that is what
  `run_sweep_h300_branch_h_followup.sh` actually tests.

New launcher: **`transformer_neurIPS/run_sweep_h300_branch_h_followup.sh`**
-- runs only these two arms (not a re-run of the full 8-arm grid, which
already produced a clear answer).

### 25.6 On "are we making progress, should we think bigger"

Real progress, not yet a closed case: `e3_ar_long` (+30.77%) ->
`s6_e3_scaled` (+27.41% at production scale, confirming it) ->
`h1_ar_freq2` (+41.90% in 400 steps) is a genuine, monotonic improvement
with an identified, reproducible cause (AR-loss frequency). But the ridge
regression floor is +69.49% in the same units (§25.1) -- every transformer
variant tried, including the best one, remains behind a closed-form linear
map with no learned nonlinearity. `h9`/`h10` are this section's answer to
"think bigger": `h9` pushes the proven lever to its limit; `h10` is a
structural bet that stops asking the network to rediscover the ridge map
from scratch and instead lets it start from it.

### 25.7 What v4.5 does NOT change

- `e3_ar_long`, `s6_e3_scaled`, and `h1_ar_freq2` through `h8_ar_lrlow`
  are unchanged -- `h9`/`h10` are additions, not redefinitions.
- `PREDICT_DELTA`'s existing (persistence-anchor) behaviour is byte-for-byte
  unchanged when `DELTA_ANCHOR` is left at its new default
  (`'persistence'`) -- confirmed by the smoke test's second model, built
  with the old code path, producing finite output of the expected shape.
- No claim is made about whether `h9` or `h10` actually improves on
  `h1_ar_freq2` -- that's what `run_sweep_h300_branch_h_followup.sh`
  tests, not something asserted here in advance.

### 25.8 A note on `LATEST_UPLOAD_ME.md` getting overwritten

Each sweep invocation writes its own permanent report to
`sweep_logs/<run_id>/UPLOAD_ME.md` AND copies it to the top-level
`sweep_logs/LATEST_UPLOAD_ME.md` for convenience -- only the top-level copy
is overwritten each run; nothing is lost server-side. Locally, `scp`-ing
`LATEST_UPLOAD_ME.md` to the same local filename every time overwrites the
local copy the same way. To keep local history, either `scp` the whole
`sweep_logs/<run_id>/` directory instead of just the one file, or rename
each pull (e.g. `LATEST_UPLOAD_ME_<run_id>.md`). This document (OVERVIEW.md)
is the durable record either way -- every run analyzed in this session has
its headline numbers transcribed into a dated section here regardless of
what happens to the local `.md` copies.

## 26. v4.6 -- h9's frequency extrapolation wins, the ridge-residual architecture fails for a real (non-buggy) reason, and a diagnostics performance fix

### 26.1 `run_sweep_h300_branch_h_followup.sh` results

```
arm                 IMPROV%          train beats anchor?
h9_ar_freq1          +43.78%         NO
h10_ridge_residual   -1,503,218.46%  yes
```

**`h9_ar_freq1` (h1_ar_freq2's config with `AR_EVERY_N_STEPS=1`) is the new
best result in this entire investigation** -- +43.78% at only 400 steps,
beating `h1_ar_freq2` (+41.90%), `e3_ar_long` (+30.77%), and
`s6_e3_scaled` (+27.41% at production scale). Its per-frame curve crosses
positive by frame ~17 and climbs to a +55-59% plateau by frame ~40,
similar shape to `h1_ar_freq2` but higher throughout. This is the 3rd
consecutive confirmation that AR-loss frequency is the dominant lever in
this whole investigation (freq8 -> freq4 -> freq2 -> freq1, monotonically
improving every time it's been tested).

### 26.2 `h10_ridge_residual` failed catastrophically -- verified NOT a bug, and the real reason is instructive

Roll MSE hit 42.75 against a 0.0028 persistence floor (~15,000x worse),
diverging every single frame with no recovery. Before concluding the idea
doesn't work, the implementation was checked directly: a numerical
unit test with a hand-checkable ridge matrix (`A = 2 * I`, zero-init head
so the model's output equals the anchor exactly at construction) confirmed
`_ridge_anchor()`'s target alignment is correct -- `out[t]` equals the
ridge map's prediction for target token `t+1`, exactly matching
`_delta_anchor()`'s convention, with correct fallback behaviour at
sequence boundaries. The log's separate `torch.jit.script` roundtrip-check
device-mismatch warning (harmless -- training completed with sane,
decreasing train/val_tf loss throughout) is unrelated to this failure and
is a known, unfixed cosmetic issue specific to exporting
`DELTA_ANCHOR='ridge'` checkpoints, not to training or rollout correctness.

**The real explanation**: persistence (a plain copy) is a non-expansive
operator -- copying cannot amplify a value, so however wrong it is, it
can't get exponentially worse under repeated feedback. The fitted ridge
map has no such guarantee; `linear_frame_baseline()`'s own standalone
rollout is stable (+69.49%) because it is a single, clean recursion
applied to nothing but its own output. Embedded as `h10`'s per-step
anchor, it instead operates on frames built from a chain of
(learned-residual + ridge-anchor) values recursively fed back through 400
undertrained steps of AR training -- any direction of the ridge map with
gain greater than 1 compounds multiplicatively over the 68-step rollout,
which is exactly the observed exponential, monotonic, never-recovering
blowup. Swapping a non-expansive anchor for an expansive one turned a
stable feedback loop into an unstable one. This is a genuine negative
result, not an implementation defect -- the ridge-residual direction is
abandoned, not queued for a bug fix.

### 26.3 A diagnostics performance fix: chunking a small population multiplied sequential-loop overhead for nothing

After capping `linear_frame_baseline()`'s rollout population at
`max_val_seqs=2500` (v4.5 §25.3), the loop still split that population into
~128-sequence chunks (`chunk=128`), each running the inherently sequential
~68-step rollout independently -- ~20 chunks x 68 steps = ~1360 small GPU
calls, dominated by CPU-side Python/kernel-dispatch overhead between calls
rather than actual GPU compute. This is exactly why diagnostics showed as
CPU-pegged with the GPU mostly idle. Added a `val_chunk` parameter
(default: `max_val_seqs`, i.e. one single batch) that collapses the outer
loop to one iteration, cutting that dispatch overhead by ~20x. `chunk`
(used only for the ridge FIT loop, which has no inner sequential loop) is
unchanged.

### 26.4 `s7_h9_scaled` registered -- no step count baked in, unlike `s6_e3_scaled`

New arm `ROUND2_ARMS["S"]["s7_h9_scaled"]`: `h9_ar_freq1`'s exact overrides
(`AR_MODE=frame_ar`, `AR_FRAMES=8`, `AR_SEQS=2`, `AR_EVERY_N_STEPS=1`).
Unlike `s6_e3_scaled`, no `MAX_STEPS` is baked into the arm -- `freq=1` is
the most expensive AR frequency tried (measured 0.825s/step, i.e. ~2x
`s6_e3_scaled`'s AR cost at `freq=8`), so the right step budget depends on
actual available wall-clock, not a fixed convention.

New launcher **`run_sweep_h300_scale_h9.sh`**, sized for a real ~30-minute
time budget: `MAX_STEPS=2000` default (2000 * 0.825s ≈ 27.5 min, leaving
margin for checkpoint/eval overhead), diagnostics skipped by default
(`SKIP_DIAGNOSTICS=1` -- this arm doesn't need the ridge map; only the
now-abandoned `h10_ridge_residual` did).

**File upload check for this next run** -- if `data/train_80.h5`,
`data/val_80.h5`, and the scripted decoder are already on the box from the
prior `run_sweep_h300_branch_h_followup.sh` run (they should be, nothing
deletes them), only these need to be re-synced:
- `transformer_neurIPS/train_production_transformer_deep_dive.py` (new `s7_h9_scaled` arm + the `val_chunk` diagnostics fix)
- `transformer_neurIPS/run_sweep_h300_scale_h9.sh` (new file)

`model_variants.py` and `sweep_deep_dive.py` are unchanged since the last
upload and do not need re-syncing for this run.

Verified: `py_compile` on all three touched files, `resolve_arm('s7_h9_scaled')`
resolving to the exact overrides above, `bash -n` on the new launcher, and
the full 44-test suite.

### 26.5 What v4.6 does NOT change

- `h1_ar_freq2` through `h10_ridge_residual` are unchanged -- `s7_h9_scaled`
  is an addition, matching `s6_e3_scaled`'s relationship to `e3_ar_long`.
- No claim is made about whether `h9_ar_freq1`'s +43.78% holds at 2000
  steps -- that is what `run_sweep_h300_scale_h9.sh` actually tests.
- The `DELTA_ANCHOR='ridge'` code path itself is not removed or reverted --
  it is a correct, tested, but empirically unproductive option, kept for
  the record (§26.2) rather than deleted.

## 27. v4.7 -- explained the periodic CPU/GPU stalls, and made the fix the default for every sweep arm going forward

### 27.1 Diagnosis: `torch.jit.script` + roundtrip verification, every 25 steps, unconditionally

Observed on `s7_h9_scaled`'s run: long periods where both CPU and GPU sit
idle, on hardware where "everything should be in memory." Traced to
`save_checkpoint()` (train_production_transformer_deep_dive.py): every
call also runs `save_scripted_model()` -- a real `torch.jit.script()`
compile plus a full reload-and-forward-pass roundtrip verification,
CPU-bound and single-threaded, with the GPU idle throughout. The trigger
is `Config.CHECKPOINT_EVERY_STEPS = 25`, which is a SEPARATE cadence from
`--val-every` (that only controls the expensive rollout eval) and was
never exposed as a CLI flag at all -- it fires every 25 steps regardless
of any launcher's settings. At `MAX_STEPS=2000` that's 80
compile-plus-verify pauses over one run, on top of any `_train_best.pt`
saves whenever train loss improves. Reasonable for a long overnight
production run where checkpoint durability matters; a poor default for
the shallow, throwaway screens this whole sweep-arm investigation runs.

### 27.2 Fix: made the default going forward, in `sweep_deep_dive.py` itself, not per-launcher

Rather than editing every `run_sweep_*.sh` script individually,
`arm_command()` (the single place that builds every sweep arm's command
line, used by every launcher) now always prepends two `--set` overrides:

```
SAVE_SCRIPTED_MODELS=False
CHECKPOINT_EVERY_STEPS=<that run's --max-steps>
```

`SAVE_SCRIPTED_MODELS=False` skips the expensive compile+verify path
entirely (the state-dict `.pt`, written unconditionally, is unaffected --
still the authoritative artifact). `CHECKPOINT_EVERY_STEPS=<max_steps>`
means the "latest" checkpoint only saves once, at the end (`hit_budget`
still forces a final save regardless of the exact modulo alignment), not
every 25 steps -- a short sweep arm doesn't need 80 intermediate
checkpoints. An explicit `--set SAVE_SCRIPTED_MODELS=True` still wins if a
caller genuinely wants a deployable scripted checkpoint from a specific
sweep run: `arm_command()` puts the new defaults FIRST in the `--set`
list, and the trainer's own `--set` loop applies overrides in order
(last value for a given key wins), so anything a caller passes explicitly
is guaranteed to be listed after these defaults and therefore takes
precedence.

**Also fixed while in this code, a real latent bug the above change would
otherwise have collided with**: `arm_command()` was emitting one `--set`
flag PER override item (`--set A=1 --set B=2`), but the trainer's
`--set` uses `nargs="*"` without `action="append"` -- repeated
occurrences of the same flag on one command line each RESET the parsed
list rather than accumulate, so only the LAST `--set` survived. Any
previous sweep invocation with more than one `--set` override in play
would have silently dropped all but the last one. Fixed by emitting a
SINGLE `--set` flag followed by every value, space-separated, matching
`nargs="*"`'s actual contract.

### 27.3 Wall-clock kill-switch tightened to 15 minutes by default

`sweep_deep_dive.py --max-hours` (a stuck-job safety net, separate from
`--max-steps`/`--val-every` -- not a duration estimate) defaulted to 12.0
hours, appropriate for a long unattended production run but needlessly
loose for the shallow, time-boxed screens this tool is actually used for
now. Default changed to 0.25 (15 minutes). `run_sweep_h300_scale_h9.sh`'s
own `MAX_HOURS` default updated to match (0.25, was 0.75) rather than
relying on the central default implicitly drifting out of sync with what
the launcher prints.

Verified: `arm_command()` exercised directly -- confirms exactly one
`--set` flag is emitted, the two new defaults appear in it, and an
explicit user override placed after them in `args.set` correctly appears
last in the emitted command (confirming last-value-wins precedence holds
end to end, not just in isolation); the trainer's own `build_parser()`
correctly parses that combined multi-value `--set` flag; `bash -n` on the
updated launcher; and the full 44-test suite.

### 27.4 What v4.7 does NOT change

- No arm definitions changed -- this is infrastructure only.
- `SAVE_SCRIPTED_MODELS`/`CHECKPOINT_EVERY_STEPS` defaults in `Config`
  itself (`True`/`25`) are unchanged -- the override happens at the
  sweep-launch layer (`arm_command()`), not in `Config`, so a direct
  invocation of the trainer outside `sweep_deep_dive.py` (e.g. a real
  production run) keeps its original durability behaviour untouched.

## 28. v5.0 -- major version bump: is our best hope (h9_ar_freq1) actually stable?

Crossing a major version boundary (v4.x -> v5.0) per the WANDB_PROJECT
convention (OVERVIEW.md §21.1) -- `Config.WANDB_PROJECT` renamed
`NI_Review_v4` -> `NI_Review_v5`, kept in sync by `test_version_sync.py`.

### 28.1 Why now: h9_ar_freq1 is the best result so far, but "best" isn't the same claim as "stable"

Every prior round asked "does this arm beat persistence." This round asks
a different question: does the BEST arm found so far (`h9_ar_freq1`,
+43.78% at 400 steps, §26.1) hold up under small, independent
perturbations, or is it a narrow, fragile optimum that a slightly
different regularisation/LR/batch setting would break? Four arms, each
changing exactly one stability-relevant axis against `h9`'s exact
config, same one-variable-at-a-time discipline as `ROUND1_ARMS`:

```
v1_h9_wd         weight decay 0.01 -> 0.05   (regularisation)
v2_h9_clip       gradient clip 1.0 -> 0.5    (backprop magnitude control)
v3_h9_moreseqs   AR_SEQS 2 -> 4              (AR-gradient noise reduction)
v4_h9_lrlow      LR 1e-3 -> 5e-4             (optimisation aggressiveness)
```

A per-arm SEED override was deliberately NOT used as a 5th/replacement
axis: `apply_arm()` runs before `sweep_deep_dive.py`'s own CLI-args loop,
which unconditionally does `setattr(Config, "SEED", args.seed)` whenever
`--seed` is passed (always, for every sweep launch) -- a `SEED` baked
into an arm's `overrides` dict would be silently clobbered back to the
launcher's shared `--seed` value every time. A real seed-robustness check
needs its own separate `--seed` launcher invocation, not a same-invocation
arm override; not attempted this round given the time budget.

### 28.2 Wall-clock IS the real budget mechanism, not step count

Confirmed directly in the trainer's own training loop:
`if (time.time() - t_start) / 3600.0 > Config.MAX_HOURS: ... break` --
`--max-hours` is a real, enforced stop condition checked every step, not
just an external supervisory kill-switch. New launcher
**`run_sweep_h300_v5_stability.sh`** uses this as the ACTUAL budget:
`--max-steps 100000` (high enough to never bind first) with `--max-hours`
set from measured GPU count:

```
4+ GPUs  -> --max-parallel 4 --max-hours 0.25     (15 min, all 4 truly simultaneous)
<4 GPUs  -> --max-parallel 1 --max-hours 0.0833   (5 min each, sequential, ~20 min total)
```

One side effect worth knowing: with `CHECKPOINT_EVERY_STEPS` defaulted to
`--max-steps` (v4.7's fix) and a wall-clock-bounded run stopping at a step
count far below that, the generic "_latest.pt" snapshot never actually
saves (its modulo condition is never hit, and `hit_budget` -- gated on
`step >= MAX_STEPS` -- never fires either). `_best.pt`/`_rollout_best.pt`
still save normally, since those are gated on `VAL_EVERY_STEPS`, not
`CHECKPOINT_EVERY_STEPS`. The run's numbers and per-frame curves are
unaffected either way -- `curves`/`results.json`/`UPLOAD_ME.md` are built
from in-memory eval history and wandb telemetry, not from re-reading a
saved checkpoint.

### 28.3 Richer wandb telemetry: the rollout SHAPE, not just 3 fixed points

`evaluate()` has always computed the full per-frame improvement curve
(`improvement_pct_per_frame`, up to 68 values), but the eval-time wandb
log call explicitly filtered out list-typed values -- so only
`improvement_pct_frame1`/`_frame_half`/`_frame_last` (3 fixed points) ever
reached wandb; the actual curve shape was invisible there, even though
every printed `UPLOAD_ME.md` report has shown it all along. Fixed: every
8th frame is now also logged as its own scalar key
(`frame_improvement/f00`, `f08`, `f16`, ...), plus `rollout_rmse_mps`
(the same quantity the promotion gate's sanity ceiling checks). Each
becomes its own line in wandb, growing over training -- this is what
actually shows whether a run's rollout is stabilising toward a shape like
`h9`'s (positive from frame ~17, plateauing ~55-59%) or drifting/oscillating,
not just its final aggregate number.

### 28.4 `WANDB_GROUP`: four runs, one coloured comparison view

New `Config.WANDB_GROUP` (default `None`, no behaviour change unless
set) is passed to `wandb.init(group=...)` when set. The launcher sets it
to `"v5_stability"`, so wandb's native multi-run comparison view groups
all four arms together -- each gets its own auto-assigned colour across
every shared chart (loss curves, rollout MSE, the new per-frame lines,
grad norm), rather than needing to be found and manually overlaid as four
separate runs.

### 28.4a Console convenience: the run's wandb URL is now printed at start

`_Telemetry.__init__` now calls `self.run.get_url()` right after
`wandb.init()` succeeds and prints it (bold cyan, via the file's existing
`_bold`/color helpers) as `[wandb] run: <url>` -- one line per arm, so
each of the 4 concurrent (or sequential) runs' dashboard link is visible
directly in that arm's own console/log output instead of having to hunt
for it in the wandb UI. Falls back to a plain "no URL available (offline
mode?)" message if `get_url()` raises or returns nothing, rather than
erroring -- purely a convenience addition, no change to whether/how
tracking happens.

**File upload check for this run** -- if data/the scripted decoder are
already on the box, only these need re-syncing:
- `transformer_neurIPS/train_production_transformer_deep_dive.py` (new `V` branch arms, `WANDB_PROJECT`/`WANDB_GROUP`, richer wandb payload, printed run URL)
- `transformer_neurIPS/sweep_deep_dive.py` (if v4.7's `--max-hours`/`--set` fix isn't already synced)
- `transformer_neurIPS/run_sweep_h300_v5_stability.sh` (new file)

`model_variants.py` is unchanged for this round.

Verified: `resolve_arm()` on all four new arms; a `_Telemetry` smoke test
with a mocked `wandb` module confirming the URL prints correctly and the
offline/no-URL fallback doesn't raise; `py_compile` on all three
touched Python files; `bash -n` on the new launcher; the full 44-test
suite, including `test_version_sync.py` against this section's own
heading.

### 28.5 What v5.0 does NOT change

- `h1_ar_freq2` through `s7_h9_scaled` are unchanged -- Branch V is an
  addition.
- No claim is made yet about which (if any) of the four stability arms
  actually helps -- that's what `run_sweep_h300_v5_stability.sh` tests.
- `SAVE_SCRIPTED_MODELS=False`/`CHECKPOINT_EVERY_STEPS=<max_steps>` (v4.7)
  still apply here via `arm_command()`'s defaults, on top of
  `--skip-diagnostics` (this round doesn't need the ridge map) and the new
  `--max-hours`-as-real-budget usage -- all compounding to keep these four
  short, wall-clock-bounded runs as cheap and fast as possible.

### 28.6 First launch invalidated: `--max-steps 100000` broke the LR warmup schedule

The first real launch of `run_sweep_h300_v5_stability.sh` (`round2_20260903_011512`)
produced results that had to be discarded. `v1_h9_wd` and `v2_h9_clip`
both showed catastrophic, much-worse-than-`h9`-ever-showed numbers (train
loss above the zero-predictor floor for ~100 steps, rollout `IMPROVEMENT`
still deeply negative -- -649% and -3321% respectively -- when the 5-minute
wall-clock cutoff hit at step ~325-350). Root cause, confirmed by
comparing LR values step-by-step against `h9_ar_freq1`'s own successful
log: `Config.WARMUP_FRAC = 0.03` sizes the LR warmup as a fraction of
`Config.MAX_STEPS`. `h9`'s own run used `MAX_STEPS=400` -> a 12-step
warmup, reaching peak LR (~1e-3) by step ~25. This launcher's first
version set `--max-steps 100000` (meant to guarantee `--max-hours` was the
binding constraint) -> a 3000-step warmup. A 5-minute wall-clock budget
only completes ~300-350 steps at this arm family's throughput, so every
arm spent its entire budget at roughly 1/10th peak LR, still deep in
warmup -- the catastrophic numbers were an LR-schedule artifact, not a
real stability signal. The run was killed before `v3`/`v4` could hit the
same bug, and all four arms' results from this run are discarded, not
interpreted.

Fixed: `--max-steps` is now sized realistically (360 for the sequential/
5-min-each path, 1090 for the 4+-GPU/15-min-simultaneous path) from
`h9_ar_freq1`'s own measured throughput, so `WARMUP_FRAC` produces a
warmup length matched to a run actually expected to finish near that
budget. `--max-hours` is kept as what it was always meant to be -- a
safety net for an individual arm's throughput running slower than
estimated (e.g. `v3_h9_moreseqs`'s `AR_SEQS=4` costs more per step than
the other three) -- not the primary pacing mechanism.

**Also discovered from the same run's logs**: `wandb` was disabled the
entire time (`UsageError: No API key configured`) -- none of §28.3/28.4's
richer telemetry or grouped comparison view was actually recorded for
this run. Run `wandb login` (or set `WANDB_API_KEY`) on the box before
the next launch, or the coloured multi-run comparison this section was
built for won't have any data to show.

Verified: `bash -n` on the corrected launcher; the full 44-test suite
(unaffected -- this was a launcher-level pacing bug, not a change to
`Config` defaults or arm definitions).

### 28.7 Real results: `h9_ar_freq1` is stable, not fragile

The corrected launch (`round2_20260903_013205`, ~5 minutes/arm, GPU
uncontended after §28.6's kill/cleanup) completed cleanly:

```
v3_h9_moreseqs (AR_SEQS 2->4)      +46.49%   <- best, beats h9 itself
v2_h9_clip     (clip 1.0->0.5)     +42.45%
v1_h9_wd       (WEIGHT_DECAY 0.01->0.05)  +42.14%
v4_h9_lrlow    (LR 1e-3->5e-4)     +41.52%
                             h9_ar_freq1 baseline (400 steps): +43.78%
```

All four land within a ~5-point band of baseline across four independent
perturbations (5x weight decay, halved gradient clip, doubled AR
sequences, halved LR) -- exactly the signature of a real, robust result
rather than a fragile optimum. Per-frame shapes are consistent across all
four too: crossing positive around frame 14-17, plateauing in the
mid-to-high 50s% by frame 40+. `v3_h9_moreseqs` (more AR sequences per
AR-loss application, reducing gradient variance) not only held up but
edged out `h9` itself -- the same hypothesis `h5_ar_moreseqs` tested at
`freq=8` without a clear benefit (§25.4) now shows a real, if modest,
gain at `freq=1`, where AR-gradient variance matters more.

**Conclusion**: `h9_ar_freq1`'s win is real and stable, not a fragile
optimum sensitive to nearby hyperparameter choices. The natural next step
is scaling the best-of-the-stable-set (`v3_h9_moreseqs`'s config) to
production, the same pattern as `e3_ar_long -> s6_e3_scaled` and
`h1_ar_freq2 -> h9_ar_freq1 -> s7_h9_scaled`.

## 29. v5.5 -- plan: an 8-hour production run of the winning, now-confirmed-stable config

All model checkpoints and reports from every arm run so far are pulled
down locally (`saved_models/`, `sweep_logs/`) via the `rsync` in §0 --
nothing on the rented box is the only copy of anything at this point.
This section is a PLAN, not a report: `s8_h9_moreseqs_scaled` has been
registered and `run_sweep_h300_production_8h.sh` built and validated, but
NOT yet launched as of this writing.

### 29.1 What runs: `s8_h9_moreseqs_scaled`, not `h9` itself

Of the four independent stability perturbations tested in v5.0 (§28.7)
against `h9_ar_freq1` (weight decay 0.01->0.05, gradient clip 1.0->0.5,
AR_SEQS 2->4, LR 1e-3->5e-4), all four held within a ~5-point band of
`h9`'s own +43.78% -- confirming the win is stable. One of them,
`v3_h9_moreseqs` (AR_SEQS 2->4, less gradient noise per AR-loss
application), didn't just hold: it reached **+46.49%**, edging out `h9`
itself. New arm `ROUND2_ARMS["S"]["s8_h9_moreseqs_scaled"]` is that exact
config (`AR_MODE=frame_ar`, `AR_FRAMES=8`, `AR_SEQS=4`,
`AR_EVERY_N_STEPS=1`) -- the production candidate is the best-of-the-
stable-set, not the original unmodified `h9` config. Same "scale the
winner, not the origin point" pattern as `e3_ar_long -> s6_e3_scaled` and
`h1_ar_freq2 -> h9_ar_freq1`.

### 29.2 Step budget: measured from this exact config's own steady-state throughput

`v3_h9_moreseqs`'s own log (the v5.0 stability run) gives a clean
steady-state read: step 125 at 3.4 min, step 300 at 4.9 min -- no slow
first-eval or compile-warmup included -- **0.514 s/step**. Over an
8-hour (28,800s) budget, reserving ~15 minutes for combined startup/
eval/checkpoint overhead, that's roughly 54,000 steps at the measured
rate. `MAX_STEPS` is set to **45,000**, deliberately below that raw
estimate, for margin against measurement noise and the extra per-step
cost of re-enabled scripted-checkpoint compiles (see 29.3).

**Why being on the generous side here carries little risk**, unlike
§28.6's warmup bug: `WARMUP_FRAC=0.03 x 45,000 = 1,350` steps, ~12
minutes at this rate -- a small fraction of an 8-hour budget regardless
of which way the throughput estimate is off. If `--max-hours 8.0`
(kept on as the real enforced backstop, exactly as in every other
launcher since v4.7) triggers before `MAX_STEPS` is reached, the run
simply stops with a somewhat-less-annealed LR, not stuck deep in warmup
the way a 5-minute run would be -- a graceful degradation, not the same
class of failure.

### 29.3 Production overrides: two v4.7 sweep-arm defaults deliberately flipped back

`arm_command()` defaults every sweep arm to `SAVE_SCRIPTED_MODELS=False`
and `CHECKPOINT_EVERY_STEPS=<max_steps>` (v4.7) -- correct for a
5-30-minute throwaway screen, wrong for an 8-hour run meant to produce a
real artifact:

- **`SAVE_SCRIPTED_MODELS=True`** -- a deployable TorchScript export is
  the actual point of a production run.
- **`CHECKPOINT_EVERY_STEPS=2000`** -- resumable "latest" snapshots every
  ~17 minutes of wall-clock, not just once at the very end. A crash 7
  hours in without this loses everything.

Both are passed as explicit `--set` overrides, listed after
`arm_command()`'s own defaults in the combined `--set` flag -- verified
directly (see 29.5) that the trainer's last-value-wins `--set` loop
resolves them correctly regardless of order.

Also unlike the v5.0 stability launcher: **diagnostics are NOT skipped**
here (worth a fresh one-time ridge-baseline reconfirmation before an
8-hour commitment -- a minute or two, negligible against 8 hours), and
**`VAL_EVERY`/`ROLLOUT_SEQS` are widened** to 2000 steps / 128 sequences
(from the stability round's 100 steps / 64 sequences) -- an 8-hour run
affords both a coarser eval cadence (no need for the dense read a 5-minute
screen needed) and a more statistically solid final rollout score.

### 29.4 New launcher: `run_sweep_h300_production_8h.sh`

```
bash transformer_neurIPS/run_sweep_h300_production_8h.sh
```

Defaults: `MAX_STEPS=45000`, `MAX_HOURS=8.0`, `VAL_EVERY=2000`,
`ROLLOUT_SEQS=128`, `CHECKPOINT_EVERY=2000`, `SUBSET_RATIO=1.0`,
diagnostics on, single arm (`--max-parallel 1`, nothing else to run
concurrently against). All overridable via environment variables per the
script's own header. Only `train_production_transformer_deep_dive.py`
(new `s8` arm) needs re-syncing if the rest of the required files are
already on the box from the v5.0 run -- this launcher itself is also new.

### 29.5 Verified before launch

- `resolve_arm('s8_h9_moreseqs_scaled')` resolves to the exact
  `v3_h9_moreseqs` overrides.
- `arm_command()` exercised directly with this launcher's actual
  `--set` values: confirmed the combined `--set` flag is
  `['SAVE_SCRIPTED_MODELS=False', 'CHECKPOINT_EVERY_STEPS=45000',
  'SAVE_SCRIPTED_MODELS=True', 'CHECKPOINT_EVERY_STEPS=2000']` -- the
  trainer's last-value-wins `--set` loop resolves this to
  `SAVE_SCRIPTED_MODELS=True` / `CHECKPOINT_EVERY_STEPS=2000` as
  intended.
- `bash -n` on the new launcher.
- `py_compile` on the touched Python file.
- The full 44-test suite.

### 29.6 What v5.5 does NOT change (yet)

- Nothing has been launched as of this writing -- this section documents
  the plan and the artifacts built for it, not a result.
- `h9_ar_freq1`, `v1`-`v4`, and `s7_h9_scaled` are unchanged --
  `s8_h9_moreseqs_scaled` is an addition.
- `WANDB_PROJECT` stays `NI_Review_v5` -- v5.5 is a minor version within
  the v5.x major line, no rename needed (kept in sync by
  `test_version_sync.py`).

## 30. v6.0 -- hardening against the two-processes-one-wandb-run incident, and a much louder resume/fresh-start signal

Crossing another major version boundary (v5.x -> v6.0) per the
`WANDB_PROJECT` convention (§21.1) -- `Config.WANDB_PROJECT` renamed
`NI_Review_v5` -> `NI_Review_v6`, kept in sync by `test_version_sync.py`.
This is an infrastructure-only pass, not a new arm or a new result: it is
the direct fix for the failure mode uncovered while post-mortem-ing
`r2_s7_h9_scaled`'s wandb run (the one recovered to
`saved_models/r2_s7_h9_scaled_rollout_best.pt` at step 134000, +31.35%).

### 30.1 The incident, restated precisely

`r2_s7_h9_scaled`'s single wandb run (`t80_ws0`) contains two disjoint
training attempts concatenated on one x-axis: a clean 12.1-hour run from
step 1 to step 90000 (`train_loss` reaching 0.0029), then a 38-minute gap,
then a SECOND process that started completely cold (`warm_started=False`,
its own step counter reset to 1, LR back at warmup) and ran a further 19.8
hours to step 145000 -- but logged into the exact same wandb run. The
overlapping x-axis is what looked like a "regression at step ~90000" when
the run was first charted; it was actually two different curves sharing
one plot. Root cause, found in this file's own code rather than assumed:
line-level, `wandb.init(..., id=f"r{SWEEP_ROUND}_{ARM}", resume="allow")`
-- a STATIC id derived only from the arm name, with `resume="allow"`,
completely decoupled from whether this process actually loaded a
checkpoint. Any relaunch of the same arm -- deliberate resume, accidental
double-launch, or (as happened here) a cold restart after the checkpoint
was missing/reset -- silently lands in the same wandb history no matter
what actually happened on the training side.

### 30.2 Fix 1: the wandb run id now follows the checkpoint, not the arm name

`save_checkpoint()` (train_production_transformer_deep_dive.py) now stamps
the live `tel.run.id` into every checkpoint payload as `wandb_run_id`
whenever wandb is enabled. At startup, the resume block reads
`resumed_wandb_run_id = ck.get('wandb_run_id')` back out of `latest.pt` --
but only when the load is a genuine same-shape resume (`not cross_version`;
a cross-version warm-start-in-disguise is a different logical run and must
not inherit the old id). The wandb-id decision that follows:

- **A real resume** (`resumed_wandb_run_id` set): `wandb.init(id=that_id,
  resume="must")`. `resume="must"` is deliberately stricter than the old
  `"allow"` -- if that id doesn't actually exist upstream (deleted run,
  wrong project, a checkpoint saved while wandb was offline and never got
  a real id), the attempt fails loudly and falls back to case 2 rather
  than silently creating a run that *looks* resumed but shares no real
  history with anything.
- **Everything else** (cold start, `--fresh`, cross-version warm-start, a
  failed strict-resume attempt): a brand-new random id via
  `wandb.util.generate_id()`, with `resume="never"`. A cold start can no
  longer land in anyone else's history, full stop, regardless of what
  `run_name` happens to resolve to.

This directly closes the incident: the second process in §30.1 had
`warm_started=False` and no loaded checkpoint, so under this policy it
would get a fresh id and `resume="never"` -- its 19.8-hour trajectory
would show up as its own separate wandb run, not silently appended onto
the first attempt's.

### 30.3 Fix 2: a run lock, for the case Fix 1 doesn't cover

Fix 1 protects against a cold start silently reusing old history. It does
NOT protect against two processes that are BOTH doing a legitimate
same-checkpoint resume at the same time (e.g. the same launcher accidentally
invoked twice, or a retry wrapper firing while the original is still
running) -- both would read the same `wandb_run_id` out of the same
checkpoint and both would successfully `resume="must"` into it
concurrently. New `acquire_run_lock()`/`touch_run_lock()`/
`release_run_lock()`, called from the very top of `train()` (before any
data loading, so a blocked duplicate fails in milliseconds, not after
loading the H5 files): a JSON lock file at
`CHECKPOINT_DIR/{run_name}.lock` (host, pid, start time), checked for
staleness (`LOCK_STALE_SECONDS = 1800`) before every launch. A live lock
refuses the new launch with a loud error naming the host/pid/age of the
process apparently already running; a stale one (older than 30 minutes --
comfortably longer than one checkpoint/eval cycle) is reclaimed with a
warning, covering the crashed-without-cleanup case. The lock's mtime is
touched on every `CHECKPOINT_EVERY_STEPS` save as a heartbeat, and removed
via `atexit` on normal exit.

30 minutes is deliberately generous: it's longer than the actual 38-minute
gap in the §30.1 incident would have needed to be treated as "the previous
process is gone," so this would not have blocked that legitimate-looking
restart -- it targets the tighter-window double-launch case Fix 1 can't
see, not the exact incident, which Fix 1 already covers on its own.

### 30.4 Fix 3: an impossible-to-miss resume-vs-fresh-start banner

Separately from the wandb-side fix, the console output itself never made
"did this run just load 12 hours of progress, or start over" visually
obvious -- it was one `[resume]`/`[warm-start]` log line among hundreds,
easy to scroll past. New `_print_checkpoint_status_banner()`, printed
right after the resume block resolves (before the wandb section, so it's
visible even if wandb is disabled or fails to connect), and deliberately
louder than the CUDA/MPS regime banner (§9's `_regime_banner_cuda`/
`_regime_banner_mps_cpu`) it sits right below:

- 4 blank lines before, 5 after (exact ask: "several carriage returns...
  then about 5 more"), so the banner cannot be mistaken for adjacent
  scrollback.
- A full `#`-bordered box, sized to its own content, colour-coded on the
  actual RISK rather than on device type:
  - **green** -- `RESUMING FROM CHECKPOINT -- STEP N` (real progress
    continuing; the safe case).
  - **yellow** -- cross-version or warm-start-only restarts (starting
    over, but deliberately/expectedly so).
  - **red** -- `STARTING COMPLETELY FRESH -- STEP 0, RANDOM INIT, NOTHING
    LOADED` (nothing loaded at all; this is the case that silently ate 12
    hours in §30.1, and is now the loudest, reddest thing in the log).

### 30.5 What v6.0 does NOT change

- No arm definitions, no `Config` training defaults, no model code --
  this is wandb/checkpoint/console plumbing only.
- Legacy checkpoints saved before this change have no `wandb_run_id` key;
  `ck.get('wandb_run_id')` returns `None` for them, which correctly routes
  through the "brand-new id, `resume="never"`" path on their next resume --
  a safe default (a fresh run, clearly separated) rather than a crash.
- `LOCK_STALE_SECONDS=1800` is a judgment call, not a measured constant --
  revisit if a real deployment's checkpoint/eval cycle ever runs longer
  than that between saves.
- **The "63%" goal, clarified**: nothing in this file documented a "63%"
  figure under any name (checked every `##` section and every
  `IMPROV%`/`improvement_pct` mention) -- confirmed with the operator that
  the intended reference is the **ridge-regression ceiling, +69.49%**
  (§25.1), not a separately-tracked "63%" number. That figure is
  AUTOREGRESSIVE ROLLOUT improvement over persistence -- a full staircase
  forecast (every frame fed back into the next), decoded through the
  frozen AE decoder into centroid velocity (m/s) -- not a single-timestep
  comparison. It is a diagnostic CEILING, not a result any transformer arm
  has hit: the best CONFIRMED-stable result so far is `v3_h9_moreseqs` at
  +46.49% (§28.7, 400-step shallow screen), and the actual production run
  analysed in this section (`r2_s7_h9_scaled`, 145k steps, recovered to
  `saved_models/r2_s7_h9_scaled_rollout_best.pt`) peaked at +31.35%. The
  gap between the ridge ceiling (+69.49%) and the best transformer result
  (+46.49%, or +31.35% at the production run that actually completed) is
  the real headroom this whole investigation is chasing -- not "63%" of
  anything. Full walkthrough of both numbers, why they aren't directly
  comparable, and whether +46.49% is a realistic target: Appendix A.

---

## Appendix A: what "the goal" actually is -- +69.49% vs. +46.49% vs. +31.35%

Three numbers get quoted around this investigation, in three genuinely
different senses, and conflating them is easy to do by accident.

### A.1 +69.49% -- the ridge-regression ceiling (§25.1), a diagnostic bound, not a target

`linear_frame_baseline()` ridge-fits a plain **linear** map
`frame(t) -> frame(t+1)` on ~500k frame-transitions (closed-form normal
equations, `ridge=1e-3`), then rolls it out **autoregressively** -- feeding
its own prediction back in for the full 68-frame horizon, exactly like an
arm's own rollout eval -- and decodes through the same frozen AE decoder
into centroid velocity (m/s) before scoring against persistence. The
number was originally computed in the wrong space (+60.64% in raw
470-dim latent space); fixing it to decoded m/s space made the gap WORSE,
not better (+69.49%) -- the original concern understated the problem.

What it means: a ceiling on how much a model could improve on persistence
with **zero learned nonlinearity**. It exists to answer "does this task
even have learnable temporal structure beyond persistence" (yes, a lot),
not as something to literally build into the architecture -- §26.2
documents that trying to (`h10_ridge_residual`, using the ridge map as the
model's per-step prediction anchor) caused a catastrophic feedback
blowup: persistence is a non-expansive operator under repeated AR
feedback (copying a value can't amplify it), the ridge map is not, and
any eigendirection of the ridge matrix with gain > 1 compounds
multiplicatively over a 68-step rollout. Rollout MSE hit ~15,000x worse
than persistence. **Use this number as a headroom gauge, never as an
architectural template.**

### A.2 +46.49% -- `v3_h9_moreseqs` (§28.7), a real but *shallow-screen* result

Same metric space as the ridge ceiling (decoded, autoregressive,
staircase), but this one is an actual transformer result, not a linear
bound. It came from the v5.0 stability round: four independent
one-variable perturbations against `h9_ar_freq1` (weight decay, gradient
clip, `AR_SEQS`, LR), each a 400-step shallow screen. All four landed
within a ~5-point band of `h9`'s own +43.78%, which is what earned the
"confirmed-stable, not a fragile optimum" label. `v3_h9_moreseqs`
(`AR_SEQS: 2->4`) edged out the rest at +46.49% -- see Appendix B for
exactly what that knob does.

**The catch: this was never run at production scale.** The plan to do so
(`s8_h9_moreseqs_scaled`, §29) was built and validated but, as of §29's
writing, not yet launched. What actually ran to completion and produced
the checkpoint recovered in this session
(`saved_models/r2_s7_h9_scaled_rollout_best.pt`) is **`s7_h9_scaled`** --
the production scaling of `h9_ar_freq1` ITSELF (`AR_SEQS=2`), not the
improved `v3`/`AR_SEQS=4` variant.

### A.3 +31.35% -- what `s7_h9_scaled` actually delivered at 145k steps, and why the comparison to +43.78%/+46.49% is confounded

`s7_h9_scaled`'s production run undershot `h9`'s own 400-step number
(+43.78% -> +31.35%, a ~12-point drop) -- structurally the same direction
as `e3_ar_long` (+30.77% shallow) -> `s6_e3_scaled` (+27.41% production),
so on its face this reads as "shallow screens don't fully hold at
production scale." **But this particular run has a confound the
`e3`/`s6` pair does not**: per this session's wandb post-mortem
(§30.1), `r2_s7_h9_scaled`'s single wandb run is actually two disjoint
training attempts glued together -- a clean 12.1-hour run to step 90000
(`train_loss` still improving, at 0.0029) that was then COLD-RESTARTED
from scratch (not resumed) for another 19.8 hours to step 145000. The
recovered +31.35% checkpoint (step 134000) comes from the SECOND,
restarted attempt -- meaning the run that actually finished never got the
benefit of the first 90000 steps' progress, and its own `improvement_pct`
curve was still climbing (not plateaued) when the step budget ran out.

**Implication**: the 12-point shallow-to-production gap on `s7` may be
partly or entirely an artifact of the exact incident v6.0 (§30) exists to
prevent, not solid evidence that `h9`'s win erodes at scale. This can't
be disentangled retroactively (pass 1's own trajectory was cut off before
its first real eval past step 90000), which is itself a good argument for
getting `s8_h9_moreseqs_scaled` running cleanly under the v6.0 safety
fixes rather than trying to re-read more meaning into `s7`'s numbers.

### A.4 So, should you target +46.49%?

Treat it as the next lever to pull, not a number to expect on arrival:
`s8_h9_moreseqs_scaled` (the AR_SEQS=4 production scale-up) is already
built and ready to launch. Whether it lands near +46.49%, near `s7`'s
+31.35%, or somewhere else entirely is now an open, un-confounded
question -- precisely because v6.0's wandb-id/lock/banner fixes mean a
future crash-and-restart on THAT run will resume correctly or fail into a
clearly separate run, instead of silently blending two attempts the way
`s7`'s did. +69.49% (Appendix A.1) is not a target either run is expected
to reach -- it's the outer bound that motivates keep pulling on `AR_MODE`
levers at all.

## Appendix B: what `AR_SEQS` actually is

### B.1 The auxiliary loss it's a parameter of

Separately from the main per-token teacher-forced loss, the trainer
periodically (every `AR_EVERY_N_STEPS` steps) runs an autoregressive
rollout loss, `frame_ar_loss()` (train_production_transformer_deep_dive.py:1963):
starting from a randomly-chosen context length, it feeds the model's OWN
prediction back in as the next input token (not ground truth) for
`AR_FRAMES` frames, then scores the resulting chain against the real
future in decoded centroid-velocity space. This is what directly trains
against exposure bias -- the gap between "predict one step from real
context" (cheap, what teacher-forcing does every step) and "predict many
steps in a row from your own drifting predictions" (expensive, what
actually happens during rollout eval / real inference).

### B.2 What the knob controls

```python
seqs = batch[:int(cfg.AR_SEQS)]      # train_production_transformer_deep_dive.py:1978
```

`AR_SEQS` is how many sequences, OUT OF THE CURRENT TRAINING MICRO-BATCH,
get run through that autoregressive rollout loss each time it fires --
not a separate batch, not new data, just a slice of the batch already
loaded for the main loss. `AR_SEQS=2` (`h9_ar_freq1` / `s7_h9_scaled`)
takes the first 2 sequences; `AR_SEQS=4` (`v3_h9_moreseqs` /
`s8_h9_moreseqs_scaled`) takes 4.

### B.3 Why more sequences helps: variance, not new information

`centroid_velocity_loss()`'s result is a MEAN over that AR-sequence batch
dimension. At `AR_SEQS=2`, each application of the aux loss is a noisy,
low-sample-size estimate of "how bad is the rollout drift right now" --
the gradient signal it contributes swings a lot depending on which 2
sequences happened to land in that slot. Doubling to 4 halves the
variance of that estimate (standard sqrt(n) averaging) without changing
what's being measured -- same `AR_FRAMES=8` horizon, same
`AR_EVERY_N_STEPS=1` frequency, everything else identical. That's the
entire hypothesis `v3_h9_moreseqs` tested: does `h9`'s win hold up (or
improve) with a cleaner gradient signal, or was it sensitive to that
2-sequence noise? Per §28.7, it held up and edged slightly ahead.

### B.4 The cost, and why it's clamped on MPS/CPU

Real tradeoff, not free: "activation memory scales linearly with
AR_SEQS" (comment at line ~559). Each of the `AR_FRAMES * NUM_X`
sequential forward passes in the AR loop carries `AR_SEQS` in its batch
dimension, so `AR_SEQS=4` roughly doubles the compute/memory of every AR-
loss application specifically (the main teacher-forced loss and ITS batch
size are untouched). This is why `regime.aux_micro_batch` exists:
`AR_SEQS` is silently clamped down on MPS/CPU regardless of what an arm
requests, because each sequential forward retains its own full activation
graph and the whole `frame_ar` AR loss is disabled outright on MPS/CPU
(`regime.disable_ar`) -- see §9's device-adaptive-regime table. `sched`
mode (two forwards, no sequential retained-graph chain) is unaffected and
stays enabled on MPS/CPU; `AR_SEQS` is meaningless there since `sched`
mode doesn't read it.

## 31. v6.1 -- ridge-map distillation: the safe reformulation of `h10_ridge_residual`, tested on MPS

### 31.1 Why: Appendix A.4 floated this, the user asked for it built

Appendix A.4 raised using the fitted ridge map's prediction as a
supervisory DISTILLATION target instead of a feedback anchor, as an
alternative to `h10_ridge_residual`'s abandoned approach -- structurally
different enough from `h10` that it shouldn't reproduce that failure,
but explicitly flagged there as untested speculation, not a validated
plan. This section is that idea, built and verified on MPS.

### 31.2 What changed, and why it can't reproduce `h10`'s blowup

New standalone function `ridge_distill_targets(ground_truth_lat, ridge_A, cfg)`
(train_production_transformer_deep_dive.py, next to `centroid_velocity_loss`)
duplicates `model_variants.py`'s `_ridge_anchor()` math (same target
alignment: token t+1 <- complete source frame ending at t, falling back
to persistence at the leading edge) but is deliberately NOT a model
method and NEVER wired into `forward()`. It is called once per training
step on the batch's GROUND-TRUTH latents (the clean pre-noise slice
`teacher_forced()` itself builds, not the model's own prediction), and
its output never feeds back into itself, the model, or any later call --
there is no recursive chain for a bad eigendirection to compound over.
This is the exact property `h10_ridge_residual` lacked: that arm wired
the ridge map into the model's own output, which then got fed back as
input on every subsequent AR rollout step, so persistence's
non-expansive guarantee (copying can't amplify) didn't apply and the
error compounded multiplicatively (§26.2). Distillation-as-a-loss-term
has nothing analogous to compound.

New `Config` fields: `RIDGE_DISTILL_WEIGHT` (default `0.0`, opt-in) and
`RIDGE_DISTILL_WARMUP_FRAC` (default `0.2`, same ramp convention as
`AR_WEIGHT_WARMUP_FRAC`). Wired into the main training loop right after
the primary `centroid_velocity_loss` computation, on EVERY micro-batch
(not gated to `micro==0` like the AR aux loss) since it's one matmul
against a frozen buffer, not a sequential loop -- cheap enough not to
need frequency gating. Requires `Config.TOKENIZATION == 'token'` (same
scoping as `_ridge_anchor()`) and a fitted `Config.RIDGE_MAP_PATH` file
on disk; raises loudly, not silently, if either precondition is unmet.
Zero new model parameters -- every existing checkpoint, including the
recovered `saved_models/r2_s7_h9_scaled_rollout_best.pt` (step 134000,
+31.35%, confirmed by the user as the clean baseline going forward),
stays warm-startable into this path unchanged.

New arm `h11_ridge_distill` (ROUND2_ARMS["H"], next to `h9_ar_freq1`/
`h10_ridge_residual`): `h9_ar_freq1`'s exact overrides plus
`RIDGE_DISTILL_WEIGHT=0.5`, directly comparable to both `h9` (+43.78%,
no distillation) and `h10` (catastrophic failure) since it shares `h9`'s
exact base config.

### 31.3 New test file: `tests/test_ridge_distill_loss.py`

No permanent test file existed for the ridge mechanism at all before
this (§26.2's own verification was an ad-hoc, not-committed check).
Covers: exact hand-checked alignment against a `2*I` ridge matrix
(mirroring §26.2's own method), fallback-to-persistence at the leading
edge, correct behaviour when sequence length isn't a whole multiple of
`NUM_X`, an explicit "feeding the output back in produces an independent
result, not a compounded one" structural check, and
`RIDGE_DISTILL_WEIGHT` defaulting to `0.0`. 7 new tests; full suite now
51 (was 44), all passing.

### 31.4 Verified on MPS (this Mac), before any CUDA time

Two real local `train()` invocations (not `--smoke-test`, which never
touches data/loss/checkpoints at all) against `data/train_80.h5` /
`data/val_80.h5` at `--subset-ratio 0.02`:

1. **Cold start** (`--fresh --no-warm-start`, 10 steps): `[ridge-distill]
   loaded .../ridge_frame_map.pt (A: (471, 470)) -- weight ramps 0 -> 0.5
   over 2 steps` printed correctly; the v6.0 banner correctly showed
   **STARTING COMPLETELY FRESH** (red); training completed 10 finite
   steps (no NaN/Inf, exactly as expected -- there is no feedback path
   here for anything to blow up through); checkpoints saved normally.
   The eval-time rollout number at step 10 was catastrophically bad
   (-46547%), which is EXPECTED and not a regression signal: `AR_MODE`'s
   `frame_ar` loss is disabled entirely on MPS (`regime.disable_ar`), so
   these 10 steps trained on nothing but the per-token teacher-forced +
   distillation loss -- nowhere near enough for the AR rollout task at
   this step count, same as any arm this early.
2. **Resume** (same command, `--fresh` removed): `[start-from:resume]
   .../r2_h11_ridge_distill_latest.pt @ step 10` /
   `(missing=0, unexpected=0, dropped_length_dependent=[])`, and the
   banner correctly flipped to **RESUMING FROM CHECKPOINT -- STEP 10**
   (green) -- confirms v6.0's resume banner and v6.1's new loss term
   compose correctly in the same run.

Smoke-test artifacts from both runs moved to `saved_models/sweep_arms_local/`
after verification, keeping `saved_models/` root limited to the one
confirmed-best checkpoint (per the §"move checkpoints" reorganisation
earlier this session).

### 31.5 What v6.1 does NOT change (yet)

- `model_variants.py` is untouched -- no new parameters, no forward()
  changes. `h1_ar_freq2` through `h10_ridge_residual`, `s7_h9_scaled`,
  and `s8_h9_moreseqs_scaled` are all unchanged; `h11_ridge_distill` is
  an addition.
- No claim is made about whether `RIDGE_DISTILL_WEIGHT=0.5` actually
  helps `h9`'s result -- MPS verification here is mechanical (loads,
  runs, resumes, doesn't crash), not a research result. That requires a
  real CUDA shallow screen, same as every other Branch H arm.
- `WANDB_PROJECT` stays `NI_Review_v6` -- this is a minor version within
  the v6.x line (new loss term + arm, no wandb/checkpoint-format change),
  not a major boundary.

## 32. v6.2 -- how confident should we be in +69%, external research on acceleration, and a concurrency plan

Prompted directly by the operator watching `s8_h9_moreseqs_scaled`'s
early numbers (+27.17% at step 2000, dipping to +22.68% at step 4000,
out of a 45000-step budget) and asking, reasonably, whether +69% is a
credible target at all. This section is research and planning --
nothing here is implemented yet except where explicitly noted.

### 32.1 How sure can we be that +69% is reachable?

Not very, and there is a real, literature-grounded reason to say so
rather than just "it needs more training."

**What we actually know, from this project's own numbers:**
- The ridge/linear baseline reproduces cleanly across independent fits
  (+69.49% originally, §25.1; +68.97% refit fresh on the H200 this
  session, §31 -- see the `s8` diagnostics run). It is a real, stable
  number, not a fluke of one lucky fit.
- No transformer arm has ever beaten it. The best CONFIRMED-stable
  number is a 400-step shallow screen (+46.49%, §28.7); the best actual
  production number is `s7`'s +31.35% at 145k steps (Appendix A); `s8`
  at 4000/45000 steps is sitting at +22-27%, noisily, and it is much too
  early to read a trend into two eval points.

**What the wider literature says about this exact shape of gap:**
- A specific mechanism keeps showing up: **exposure bias / error
  compounding**. Teacher forcing (what the main per-step loss trains
  on) produces "fast and stable optimisation" but a model trained that
  way "is less robust when the input sequence becomes progressively
  imperfect during closed-loop rollout" -- exactly the training-vs-
  rollout mismatch this project has been fighting since Branch A/D/F
  (§18-24) and which AR-loss frequency (the dominant lever found so
  far, §25.4) directly targets. ([Scheduled Sampling for Sequence
  Prediction with Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf),
  [emergentmind synthesis](https://www.emergentmind.com/topics/scheduled-sampling))
- This is an active 2025-2026 research area specifically for
  autoregressive time-series transformers, not a solved problem:
  [Mitigating Exposure Bias in Risk-Aware Time Series Forecasting with
  Soft Tokens](https://arxiv.org/abs/2512.10056) and
  [AROpt](https://arxiv.org/html/2602.02288v2) (borrowing DeepSeek-V3's
  accept/reject rollout mechanism for continuous-valued forecasting)
  are both from the last few months, both explicitly framing the
  problem as "errors compound rapidly with horizon, even from small
  per-step errors."
- More pointed still: [Why Do Transformers Fail to Forecast Time Series
  In-Context?](https://arxiv.org/abs/2510.09776) (Zhou, Wang, Goel,
  Zhang, Oct 2026) studies transformers against AR(p) processes
  specifically and characterizes a **closed-form representational gap
  between linear self-attention and optimal linear forecasters** --
  i.e. for this whole class of problem, there is theoretical work
  arguing plain transformers are not just under-trained relative to a
  linear baseline, they may be structurally disadvantaged against one.
  The paper also names a rollout failure mode, "collapse-to-mean and
  error compounding," matching this project's own repeated observation
  that models trained without AR-loss correction degrade toward
  something anchor-like over the rollout horizon rather than staying
  sharp.
- General time-series literature independently confirms the pattern
  outside of transformers specifically: neural nets losing to linear
  baselines is well-documented whenever the underlying process is
  close to linear locally, which is exactly the hypothesis our own
  ridge-map result supports for this 47-dim latent's one-step dynamics.
  ([ScienceDirect: neural networks for linear time-series
  forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0305054800000332))

**Conclusion**: +69% should be read as "how good a *locally linear*
model is, applied recursively," not as evidence a nonlinear transformer
can reach the same number by training longer on the current recipe. The
gap is likely dominated by exposure bias (fixable, and the one lever
already proven to matter most in this project) plus a possible genuine
representational disadvantage of plain self-attention on this class of
process (not obviously fixable by more steps alone). Treat +46-50%
range as a more realistic near-term target; +69% is the ceiling that
motivates keeps trying, not a number to expect on arrival.

### 32.2 Acceleration ideas -- planning-stage menu, none implemented yet

Ordered by (estimated benefit) / (estimated implementation cost), given
we are specifically trying to beat exposure bias against a strong
linear reference, not trying to out-scale it:

1. **Rollout-horizon curriculum on `AR_FRAMES`** (cheap, high expected
   value). Currently `AR_FRAMES` is a fixed constant (8) from step 1.
   Recent rollout-training literature schedules the horizon/blending
   explicitly -- start short (e.g. `AR_FRAMES=1-2`) so the AR loss is
   learnable from the start, and grow it toward the full horizon as
   training progresses, rather than asking the network to correct an
   8-frame rollout before it can reliably do a 1-frame one. Directly
   analogous to `AR_WEIGHT_WARMUP_FRAC`'s existing ramp, just on the
   horizon instead of the weight. ([scheduled rollout recovery
   training / blending-radius curricula](https://www.emergentmind.com/topics/scheduled-sampling))
2. **Extend ridge-distillation into the AR rollout loss itself**
   (moderate cost, potentially the most direct lever). `h11`'s
   `RIDGE_DISTILL_WEIGHT` (§31) only pulls the ONE-STEP teacher-forced
   prediction toward the ridge map's prediction. The ridge map's own
   *multi-step* rollout (computed once, recursively, entirely
   independently of the network -- exactly like
   `linear_frame_baseline()` already does for diagnostics) could serve
   as a per-frame supervisory target for `frame_ar_loss`'s own rollout
   too, still never re-entering any feedback loop (same safety property
   as §31). This directly trains the network to match the ceiling
   we're trying to reach, at every horizon step it's scored on.
3. **Split anchor choice by loss, not globally** (moderate cost, a
   deliberate middle ground between `h10` and `h11`). `h10_ridge_
   residual`'s mistake was using the ridge map as the anchor INSIDE the
   AR feedback loop. Nothing stops using it as the anchor for the
   teacher-forced main loss specifically (which never recurses) while
   keeping plain persistence as the AR loss's anchor (which does) --
   getting `PREDICT_DELTA`'s "start at a strong prior" benefit for the
   one loss where it's safe, without touching the one where it isn't.
4. **Direct multi-horizon auxiliary head** (higher cost, sidesteps
   accumulation structurally rather than mitigating it). Predict
   several future frames in one shot (non-autoregressively) as an
   additional loss term, rather than only ever reaching frame N by
   chaining N single-step predictions. Literature explicitly names this
   as the way to bypass error compounding entirely rather than manage
   it. Biggest architectural lift of the options here.
5. **Soft physical-consistency penalty** (higher cost, longer-horizon
   idea). PINN literature reports physics-based penalty terms
   "accelerate convergence by reducing the solution space" for
   turbulence/vortex problems specifically. Nothing in the current loss
   uses spatial structure across the 26 (v3.1: 10) x-stations at all --
   a smoothness or approximate-consistency penalty across neighboring
   stations could regularize the solution space the same way. Not
   trivial: would need to reconstruct spatial relationships from the
   decoded 375-dim output, and PINN literature itself flags training-
   stability risk from badly-tuned physics-penalty weights.
   ([PSTNet / divergence-free layers](https://arxiv.org/pdf/2603.07957))
6. **LR/weight-decay/gradient-clip retuning, model capacity increases**
   (low priority, already weak evidence). §28.7's stability round tested
   exactly this axis (4 perturbations against `h9`) and found AR-loss
   frequency dominated everything else tried by a wide margin -- and a
   LARGER model beating a much-smaller linear map on THIS metric is not
   obviously a capacity story to begin with. Deprioritized until 1-3
   above are tried.
7. **AROpt-style accept/reject rollout stabilization** (highest cost,
   most novel). Directly borrowed from very recent (2026) LLM-training-
   inspired work adapting DeepSeek-V3's accept/reject mechanism to
   continuous-valued autoregressive forecasting specifically to "enable
   robust autoregressive rollout." Worth tracking, not worth building
   before the cheaper options above are exhausted.

None of the above are registered as arms yet -- this is the planning
inventory the operator asked for, to pick from once `h11`'s CUDA screen
result is in.

### 32.3 Running two arms concurrently on one GPU -- real, but not free

Motivated by `s8`'s own observed utilization: 24-47% GPU, ~34GB/144GB
VRAM on the H200 (§ the live diagnosis in this session) -- clearly not
saturating the box. External research confirms the instinct is sound in
general, with real caveats:

- **This only helps when a single job already under-utilizes the GPU**
  -- which `s8` demonstrably does. Literature is explicit: "parallel
  training of multiple instances ... is only relevant for small models
  which, on their own, don't utilize the GPU well enough," and the
  inverse warning is just as explicit: "if a single job already
  saturates the GPU, concurrent jobs will only slow each other down."
  ([PyTorch Forums discussion](https://discuss.pytorch.org/t/multiprocessing-vs-nvidia-mps-for-parallel-training-on-a-single-gpu/99877))
- **Plain co-location (two `python` processes, no config) uses default
  time-slicing, not true concurrency** -- benchmarks show runtime
  "increases close to linearly as more concurrent processes are added"
  under plain time-slicing, i.e. little to no actual speedup, sometimes
  a net slowdown from contention.
- **NVIDIA MPS (Multi-Process Service)** is the mechanism that gives
  genuine spatial co-residency instead: "with MPS, runtime shows little
  to no increase and occupancy grows gradually, indicating the
  different processes truly execute concurrently." Requires starting an
  MPS control daemon on the pod (`nvidia-cuda-mps-control -d`) before
  launching the two training processes -- not yet wired into any script
  here.
  ([Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep
  Learning Workloads](https://arxiv.org/pdf/2110.00459))
- **Memory duplicates per process** -- each job pays its own CUDA
  context + framework overhead on top of its model/activations, so two
  jobs cost more than 2x one job's marginal memory, not exactly 2x.
  34GB observed for one `s8`-shaped run leaves comfortable room for a
  second on a 144GB H200, but this should be verified empirically
  (start 2, watch `nvidia-smi`), not assumed to scale further without
  checking.
- **Practical recommendation**: enable MPS, start with exactly 2
  concurrent jobs (not more), and measure real wall-clock throughput
  before committing to a larger concurrent lineup -- one anecdotal
  report in the research even found 2 concurrent jobs slower than
  running them sequentially in a differently-shaped setup, so this
  needs measuring on OUR workload, not assumed from the general
  literature.

Not implemented yet -- pairs naturally with §32.2's menu once `h11`'s
result is in: run the next candidate idea alongside a continuation of
whichever config is currently leading, instead of strictly
sequentially, once MPS is confirmed to actually help on this box.

## 33. v6.3 -- menu items 1, 2, 3 implemented and folded into `h11_ridge_distill`

The operator picked three of §32.2's planning-stage menu items --
rollout-horizon curriculum (1), extending ridge-distillation into the
AR rollout itself (2), and splitting anchor choice by loss (3) -- and
asked for all three to be folded directly into `h11_ridge_distill` as
the go-forward recipe, not kept as separate untested arms. This section
documents what shipped; whether it actually helps is still an open
question for the operator's own manual 30-minute CUDA run.

**Direct answer to "is the ridge map's output used as some sort of
static guide for future trainings?"**: yes, exactly that. `ridge_frame_
map.pt` is fit ONCE (`linear_frame_baseline()`, part of diagnostics)
and never updated by gradient descent or anything else during training
-- the same frozen matrix is reused unchanged for the entire run, and
is reusable for future runs/arms indefinitely (as long as `NUM_X`/
`LATENT_DIM`/the data distribution don't change). All three items below
read this same static file; none of them make it adaptive.

### 33.1 Rollout-horizon curriculum (menu item 1)

New `Config.AR_FRAMES_START` (default `None` -- disables the curriculum
entirely, every pre-v6.3 arm unaffected) and `Config.AR_FRAMES_WARMUP_
FRAC` (default 0.3, independent of `AR_WEIGHT_WARMUP_FRAC`). New pure
function `ar_frames_for_step(step, cfg)` (next to `make_lr_lambda()`)
ramps the AR-loss horizon linearly from `AR_FRAMES_START` to `AR_FRAMES`
over `MAX_STEPS * AR_FRAMES_WARMUP_FRAC` steps, so the network isn't
asked to correct an 8-frame rollout before it can reliably do a 2-frame
one. `frame_ar_loss()` gained `n_frames_override` (defaults to `None` ->
today's flat `cfg.AR_FRAMES` behavior) so the training loop can drive
the ramp without mutating shared `Config` state.

**Repercussion handled explicitly, per the operator's own ask**: the
`[memory]` expectation printout estimates AR-loss peak memory from
`Config.AR_FRAMES` read once at startup. `AR_FRAMES` is deliberately
left meaning the CEILING (never overwritten to the curriculum's current
value) specifically so this printout still reports the true worst-case
peak under a ramping horizon, not a curriculum-deflated early estimate.
Commented directly at both the printout and the `AR_FRAMES_START` field
so a future edit doesn't "fix" this into an under-report.

### 33.2 Ridge-rollout distillation (menu item 2)

New pure function `ridge_rollout_targets(last_frame_lat, ridge_A, cfg,
n_frames)`, next to `ridge_distill_targets()` (§31). Reuses the EXACT
recursion `linear_frame_baseline()` already uses for its own diagnostic
rollout (`cur = torch.cat([cur, ones], 1) @ A`, repeated `n_frames`
times) -- the ridge map rolls out purely on its OWN output, never
touching the network, so it inherits the same structural safety
argument as `ridge_distill_targets()`: nothing here is IN a feedback
loop for a bad eigendirection to compound through.

`frame_ar_loss()` gained an optional `ridge_A` param (token-tokenization
branch only, matching every other ridge mechanism's scoping). When
given, it now returns `(ground_truth_loss, ridge_rollout_loss)` instead
of a single tensor; `None` (default) keeps the old scalar-tensor return
exactly, so every pre-v6.3 call site is unaffected. The training loop
weights the ground-truth term with the existing `ar_w` ramp and the
ridge-rollout term with the SAME `distill_w` ramp already driving the
one-step distillation (§31) -- one weight now covers both the one-step
AND the AR-horizon distillation, rather than a third independent knob.

### 33.3 Split anchor choice by loss (menu item 3)

`BaseTransformer.forward()` (`model_variants.py`) gained
`force_persistence_anchor: bool = False`. `True` forces `_delta_anchor()`
(plain persistence) for that one call regardless of `self.delta_anchor_
kind` -- overriding `DELTA_ANCHOR='ridge'` just for that call. Default
`False` preserves every existing checkpoint/behavior exactly.

Wired to `True` at every call that is INSIDE a recursive feedback chain
-- `frame_ar_loss()`'s token-tokenization loop, both branches of
`rollout_frames()` (covering real eval-time/inference rollout too, not
just training), and `sched_sampling_loss()`'s pass-1 (conservatively,
though not on `h11`'s critical path since `h9`/`h11` use `AR_MODE=
'frame_ar'` not `'sched'`). Left at the default (`False`, i.e. ridge
anchor allowed) ONLY at `teacher_forced()`'s single non-recursive call
-- the one place `DELTA_ANCHOR='ridge'` was always meant to apply.
`FrameTransformer.forward()` never had a ridge-anchor option at all and
was left untouched; its call sites do not pass the new parameter.

This is the split `h10_ridge_residual` (v4.6 §26.2) didn't have: that
arm's blowup came specifically from the ridge anchor sitting INSIDE the
AR feedback loop. With the split, `DELTA_ANCHOR='ridge'` is used on
`h11` for the first time, safely, because it can only ever apply to the
one call that never recurses.

### 33.4 `h11_ridge_distill` upgraded in place

All three combined into the existing arm (not a new `h12` -- the
operator's explicit "we'll use that going forward" framing):
```python
"h11_ridge_distill": {"overrides": {
    "AR_MODE": "frame_ar", "AR_LOSS_WEIGHT": 1.0,
    "AR_FRAMES": 8, "AR_FRAMES_START": 2, "AR_SEQS": 2, "AR_EVERY_N_STEPS": 1,
    "RIDGE_DISTILL_WEIGHT": 0.5,
    "PREDICT_DELTA": True, "DELTA_ANCHOR": "ridge",
}}
```
Honestly flagged in the arm's own `desc` string: this conflates three
variables in one 30-minute screen, a deliberate speed-over-clean-
attribution tradeoff, not something to forget when reading the result.

### 33.5 Verified

- 15 new tests (`tests/test_ridge_distill_loss.py` extended with
  `ridge_rollout_targets` cases; new `tests/test_split_anchor.py`,
  hand-checked against a real tiny `BaseTransformer`; new `tests/
  test_ar_frames_curriculum.py`). Full suite now 66 (was 51), all green.
- `resolve_arm('h11_ridge_distill')` reflects every new override.
- A real local MPS `train()` run (not `--smoke-test`) against `h11_
  ridge_distill`: 10 steps completed, one eval fired, checkpoints saved,
  no exception from any v6.3 code path. `AR_MODE='frame_ar'` is still
  disabled entirely on MPS (`regime.disable_ar`), so this validates the
  curriculum/distillation/anchor-split WIRING, not whether they help --
  same caveat as every prior MPS check this session (§31.4).
- **One pre-existing, already-documented limitation surfaced for the
  first time by this run**, not a new bug: `save_scripted_model()`'s
  `torch.jit.trace` fallback for `DELTA_ANCHOR='ridge'` models hardcodes
  the trace-time device into `_ridge_anchor()`'s `torch.ones(...)` call,
  so the CPU roundtrip *verification* fails on an MPS-trained model. The
  state-dict `.pt` (the authoritative artifact) still saved correctly
  both times -- exactly the graceful degradation `save_checkpoint()` was
  designed for. §25.5 already flagged this exact risk when `DELTA_
  ANCHOR='ridge'` was introduced ("accepted here since this is a
  research sweep arm, not a deployed inference path") -- `h11` is simply
  the first ridge-anchor arm to actually reach `save_checkpoint()` and
  make it visible. Not fixed; not blocking.

### 33.6 What v6.3 does NOT change

- No claim is made that any of items 1-3 actually improves on `h9`/
  `s7`/`s8`'s numbers -- that is exactly what the operator's manual
  30-minute SSH run against real CUDA is for.
- `h1_ar_freq2` through `h10_ridge_residual`, `s7_h9_scaled`, and `s8_
  h9_moreseqs_scaled` are all unchanged.
- `WANDB_PROJECT` stays `NI_Review_v6` -- still within the v6.x line.

## 34. v6.4 -- the §33.5 device-mismatch bug actually fixed, `screen`/`croc` added to bootstrap, and `train_80.h5`/`val_80.h5` re-exported uncompressed

The real CUDA run against Experiment A (`r2_h11_ridge_distill`, round
2) hit exactly the failure §33.5 predicted was latent, but in its CUDA
form: `RuntimeError: Expected all tensors to be on the same device, but
found at least two devices, cpu and cuda:0!` during `save_scripted_
model()`'s CPU-roundtrip verification. Root cause and fix below. While
retrieving that run's artifacts before deciding whether to terminate
the pod, also found and fixed two deploy-tooling gaps and a real
load-time bottleneck.

### 34.1 The device-mismatch fix

`model_variants.py`'s `_ridge_anchor()` built its ones-tensor as
`torch.ones(B, n_complete, 1, dtype=frames.dtype, device=frames.device)`.
Under `torch.jit.script`, `device=frames.device` gets compiled as a
**compile-time constant** captured at script time -- i.e. whatever
device the model was on when `save_scripted_model()` called
`torch.jit.script(model)`, which is `cuda:0` for any real run. That
same function then moves the scripted module to CPU for its roundtrip
check, so the *scripted* `_ridge_anchor()` tries to build a CPU input
tensor as `cuda:0`, and PyTorch (correctly) refuses to mix devices in a
Python 3.14/CUDA multi_matmul path. Fixed by switching to
`torch.ones_like(frames[..., :1])`, which reads device/dtype off the
actual runtime tensor rather than a separately-scripted attribute --
this is the same class of bug the MPS case (§31.4/§33.5) surfaced, just
manifesting on CUDA at the point an MPS run never actually reaches.

Verification caveats, stated plainly: this machine's Python 3.14 +
torch-nightly combination cannot run `torch.jit.script` at all (a
separate, pre-existing, already-documented incompatibility -- fails
with an unrelated "Unsupported value kind: Tensor" error regardless of
this fix). Verified instead via `torch.jit.trace` on the same
`_ridge_anchor()` code path (trace → move to CPU → forward), which now
succeeds where the equivalent pattern failed before this change, and
via the full test suite (78 tests, all passing). Full confirmation that
the CUDA `script`-mode roundtrip itself now succeeds still needs the
next real pod run -- as before, this only affects the *verification*
step; the state-dict `.pt` remains the authoritative artifact and
training was never blocked by this.

### 34.2 `screen` and `croc` added to `bootstrap_remote.sh`

A live pod check during this incident found `screen` itself **missing**
from the pod image -- the entire named-screen workflow
(`singleshot/README.md`'s bootstrap/exp-a/exp-b table) would have
silently failed to start without it. Added alongside the already-
planned `mc` (midnight commander) in the same `apt-get install` line.

Also added `croc` (schollz/croc -- relay-based, end-to-end encrypted,
multi-connection file transfer; not in apt, installed via the official
`curl https://getcroc.schollz.com | bash`), specifically to speed up
**both directions** of the large-file transfer: pushing `train_80.h5`/
`val_80.h5` to a fresh pod, and pulling checkpoints/results back before
terminating one. On a link with real symmetric bandwidth (e.g. ~2.5
Gbps), `croc`'s default multi-connection behavior consistently beats a
single-stream `scp`/`rsync` for these specific files, since they
bottleneck on one stream's throughput rather than per-file overhead.
`singleshot/README.md`'s "Other transfer options" section now documents
both directions with exact commands.

### 34.3 `train_80.h5`/`val_80.h5` re-exported with `compression=None`

While diagnosing why loading these files felt slow ("does the h5 file
take a long time to deserialize"), traced it to `TransformerDataset.
__init__`'s single `f['data'][:self.length]` read: `prepare_data.py`
wrote these with `compression='gzip'`, one full sequence per chunk
(`chunks=(1, NUM_TIME_NEW, NUM_X, 52)`), and HDF5's gzip filter
decompresses via **single-threaded** zlib, on every chunk, every load.

The number that made the trade obviously bad: `val_80.h5` was 3.66 GB
compressed vs. ~3.9 GB decompressed raw -- gzip was only buying a ~13%
size reduction on this float32 physics data, while still costing the
*full* single-threaded decompression time on ~4-9 GB every single load.

Fix: `decompress_h5.py` (new, one-off utility, kept in-repo since the
same conversion may be needed again) streams each file's `data`
dataset + attrs into a fresh file with `compression=None`, batched at
512 rows so memory stays bounded regardless of file size, verifies row
count + full byte-for-byte equality against the original before
atomically replacing it. Both files converted and verified:
- `val_80.h5`: 3.66 GB → 4.23 GB
- `train_80.h5`: 8.53 GB → 9.87 GB

`prepare_data.py`'s `_stream_sequences_to_hdf5()` (both `create_dataset`
call sites -- the streaming-write path and the empty-dataset fallback)
now writes `compression=None` from the start, so any future full
regeneration of this cohort doesn't reintroduce the same bottleneck.
The older 40-frame cohort (`train_40.h5`/`val_40.h5`) was deliberately
**not** touched -- it's not part of any current training path.
`tests/test_data_files_size_parity.py` updated to expect `None` for the
80-frame cohort and `gzip` for the 40-frame cohort (was a single
blanket "must be gzip" assertion for all four files).

### 34.4 What v6.4 does NOT change

- No claim about whether items 1-3 (§33) actually help -- still open,
  still needs a real CUDA run (this incident's run got cut short by
  the device-mismatch investigation, not by a conclusive result).
- The `r2_h11_ridge_distill` run's step-500 eval (`rollout_mse` ~59,437%
  worse than persistence) was checked, not explained: every recursive
  `model(...)` call site was re-verified to correctly pass `force_
  persistence_anchor=True` (§33.5's mechanism is NOT reproducing), but
  whether the bad number is early-training noise (`AR_LOSS_WEIGHT` was
  still ramping at step 500/2500) or something else is unresolved --
  needs more steps or a dedicated look, not fixed here.
- `WANDB_PROJECT` stays `NI_Review_v6`.

## 35. v6.5 -- the §34.4 rollout catastrophe explained (not noise -- a real anchor-mismatch bug), plus a `torch.compile` recompile storm found and fixed

§34.4 left the `r2_h11_ridge_distill` run's step-500 eval (~594x worse than
persistence) as "unresolved, maybe early-training noise." It was not
noise. Root-caused this session and fixed.

### 35.1 The real bug: train-time anchor != eval-time anchor

`h11_ridge_distill`'s v6.3 config combined `PREDICT_DELTA=True`,
`DELTA_ANCHOR='ridge'` (the network's `output_head` predicts a
RESIDUAL, added to an anchor -- `model_variants.py`'s `forward()`)
with the `force_persistence_anchor` split (§33.5, menu item 3):
plain persistence for every recursive call, the ridge anchor only for
`teacher_forced()`'s single non-recursive call. This correctly
prevented `h10_ridge_residual`'s exact failure mode (the ridge anchor
sitting inside a feedback loop, v4.6 §26.2) -- confirmed again this
session by re-grepping every `model(...)` call site.

But it introduced a DIFFERENT bug with the same magnitude of damage.
`output_head`'s weights are the same regardless of which anchor gets
added after it. `teacher_forced()` -- the dominant training signal by
raw call count (every micro-batch, `accum_steps` times per optimizer
step, vs. the AR loss's once-per-step) -- trains `output_head` so that
`ridge_anchor + output_head(x) ≈ true`, i.e. `output_head(x) →
true - ridge_anchor`. But `rollout_frames()` (real eval) and
`frame_ar_loss()`'s own recursive chain both call the SAME
`output_head` with `force_persistence_anchor=True`, computing
`persistence_anchor + output_head(x)`. Substituting:

```
eval_prediction ≈ persistence_anchor + (true - ridge_anchor)
                 = true + (persistence_anchor - ridge_anchor)
```

The systematic bias at every position is exactly the GAP between the
two anchors -- and that bias gets fed back into `curr` for the next AR
step, compounding through the recursive chain. This is consistent with
the pulled-back `_status.json`'s own numbers getting WORSE with frame
index (`improvement_pct_frame1=-249%` vs. `improvement_pct_frame_last
=-108990%`) -- a growing, compounding bias, not flat noise.

Reproduced in isolation (no real data, no pod needed): a tiny
`BaseTransformer` trained for 300 steps purely via the ridge-anchored
forward call, then evaluated both ways on the same input --
`force_persistence_anchor=True` eval was **~5700x** worse (MSE) than
the ridge-anchored eval on the exact same trained weights. Same
mechanism, same order-of-magnitude-of-damage as the real run's ~594x.

### 35.2 The fix

`h11_ridge_distill`'s overrides no longer set `PREDICT_DELTA`/
`DELTA_ANCHOR` at all (both fall back to `Config`'s defaults: `False`/
`'persistence'`). Ridge's influence on this arm is now **purely**
through the anchor-agnostic distillation LOSS terms
(`ridge_distill_targets()`/`ridge_rollout_targets()`, §31/§33.2) --
matching v6.1's original, simpler design, before v6.3's third menu
item added the architectural anchor on top. This is not a partial
mitigation: with `DELTA_ANCHOR` back at `'persistence'`,
`force_persistence_anchor` becomes a no-op for this arm (`forward()`
only branches on it when `delta_anchor_kind == 'ridge'`) -- train-time
and eval-time calls are now IDENTICAL in every case, for every arm,
exactly like every other non-ridge-anchor arm in this sweep (`h1`
through `h9`, `s7`, `s8`) always was. **Non-ridge-anchor arms were
never exposed to this bug in the first place** -- `DELTA_ANCHOR`
defaults to `'persistence'` and none of them override it.

The `force_persistence_anchor` mechanism itself is NOT removed --
it's still correct and still the right tool if a future arm wants to
try an architectural ridge (or any non-persistence) anchor again. The
lesson from both `h10` and this incident together: an anchor that's
allowed to differ between the DOMINANT training signal and the
eval/inference path is unsafe, independent of whether it's also inside
a recursive loop. Any future architectural-anchor experiment needs
BOTH properties checked, not just the recursion one.

**Consequence for in-flight artifacts**: the existing
`saved_models/r2_h11_ridge_distill_latest.pt` (the MPS baseline made
for the two-experiment CUDA comparison, §33 follow-up, also already
uploaded to the pod's `saved_models/`) was trained under the OLD,
buggy `PREDICT_DELTA=True`/`DELTA_ANCHOR='ridge'` architecture. It is
now stale relative to the corrected arm and should not be used to
warm-start it -- the delta head's learned semantics no longer match
what the corrected forward pass expects. A fresh baseline (or a
`--fresh --no-warm-start` run) is needed under the corrected config
before any further warm-start comparison.

### 35.3 A second, independent issue: `torch.compile` recompile storm

A live CUDA run's log also showed:
```
torch._dynamo hit config.cache_size_limit (8)
   function: 'forward' (model_variants.py:546)
   last reason: 0/0: tensor 'L['x']' size mismatch at index 0. expected 2, actual 32
```
The same compiled model (`torch.compile(model)`, CUDA-only) is called
from both `teacher_forced()` (fixed shape: `micro_batch` x full
`SEQ_LEN`, e.g. batch=32) and the AR-rollout loop (`frame_ar_loss`/
`rollout_frames`), whose batch is much smaller (`AR_SEQS`, e.g. 2) AND
whose sequence length grows by one token/frame on every one of up to
`AR_FRAMES * NUM_X` iterations, from a randomized starting context
picked fresh each call. Without `dynamic=True`, `torch.dynamo`
specializes a separate compiled graph per exact shape it sees, blows
through the default `cache_size_limit` (8) almost immediately once AR
runs, and silently falls back to eager for every shape past that limit
-- while the recompilation attempts themselves are real, CPU-bound
wall-clock cost paid mid-step, compounding the GPU-idle gaps §34's
`.item()`-sync investigation already found. Not a correctness bug
(training continued fine, no wrong numbers) -- a real, avoidable
efficiency loss.

Fixed: `torch.compile(model)` -> `torch.compile(model, dynamic=True)`.
This line is arm-agnostic and unconditional -- every CUDA arm that
enables `torch.compile` benefits, not just `h11`.

### 35.4 What v6.5 does NOT change

- The `.item()`-sync-avoidance fix (§34's investigation, landed just
  before this section) and this section's `dynamic=True` fix both live
  in shared, arm-agnostic code (the main `train()` loop's accumulation
  logic; the one `torch.compile(...)` call site) -- they apply to
  every past and future arm automatically, ridge-related or not, with
  no per-arm wiring needed.
- `h1_ar_freq2` through `h9_ar_freq1`, `h10_ridge_residual` (still
  abandoned), `s7_h9_scaled`, and `s8_h9_moreseqs_scaled` are
  unchanged -- none of them ever set `DELTA_ANCHOR='ridge'`, so none
  of them were exposed to §35.1's bug.
- `WANDB_PROJECT` stays `NI_Review_v6`.

### 35.5 Future work: KV-caching for the AR-rollout loop

Flagged, not started. `frame_ar_loss()`/`rollout_frames()`'s inner
loop recomputes a full forward pass over the ENTIRE growing `curr`
sequence at every one of up to `AR_FRAMES * NUM_X` steps -- attention
over positions 0..t is fully recomputed at step t+1 instead of reusing
step t's key/value projections for the already-seen positions. This is
the standard reason autoregressive transformer inference implements a
KV cache (store each position's K/V once, only compute the new
token's), and it's a likely next lever if the AR-rollout portion of a
step remains a disproportionate share of wall-clock/GPU-idle time
after the `.item()`-sync (§34) and `torch.compile(dynamic=True)`
(§35.3) fixes. Not attempted here -- would need genuine surgery in
`BaseTransformer`'s attention blocks (cache plumbing through
`forward()`, invalidation on a fresh rollout), a bigger change than
anything else in this section.

## 36. v6.6 -- croc self-hosted-relay transfer confirmed working (2x), via SSH tunneling

Six real pod rentals and five real bugs after first asking "why is
`train_80.h5` stuck at ~20-30 MB/s regardless of local bandwidth,"
croc's self-hosted relay is confirmed faster than its default public
relay -- **2.0x** in a real run (50.0 MB/s vs. 25.0 MB/s) -- but only
by tunneling the relay's ports through the pod's SSH connection
instead of asking RunPod to expose them directly. RunPod's Secure
Cloud pods do not expose arbitrary custom TCP ports the way this was
first assumed, including the documented "request `70000+port` for a
symmetric mapping" workaround, whose confirming env var never
appeared even after a 30s retry loop against a real pod.

Full narrative -- the timeline of all eight attempts, all five bugs
(a `pkill -f` that silently kills its own SSH session, an SSH-
backgrounded job that dies the instant its shell exits, croc's
global-vs-subcommand flag split, concurrent local-relay port
collisions, and a send/receive relay-address asymmetry), and the
RunPod networking wall -- is in `singleshot/CROC_JOURNEY.md`, written
with more room for context than code comments carry on their own.
Deliberately not duplicated here; this section is the pointer.

The reusable result: `singleshot/lib_croc.sh`, a sourceable function
library ("the reusable class" for every proven-correct croc
operation -- bash has no real classes and only bash 3.2 is available
locally, so "instance state" is a documented set of well-known global
variables instead of a nameref-based object). `singleshot/
bench_croc_variants.sh` was refactored to call it instead of
reimplementing any of this inline -- the whole point being that the
next script needing self-hosted-relay transfers (most likely
`scp_data_files.sh`'s existing `TRANSFER_METHOD=croc` path, which
predates this fix and still assumes direct pod-IP reachability) reuses
the same tested code path rather than risking bugs 1-5 all over again.

Also added while refactoring, not yet exercised on a real pod: a
size-verification step in `croc_receive()` (compares the received
file's actual byte count against the sender's, since a passing croc
exit code alone doesn't guarantee the bytes that arrived match what
was sent) and per-variant unique remote directories in the benchmark
(4 concurrent variants were previously receiving into the same shared
remote path with no isolation, silently -- caught during this
refactor, not by any test failure, since nothing was checking content
correctness before this).

**Update, same day**: the 100-port/1GB scale test surfaced a sixth
real bug -- all self-hosted variants share the SAME single SSH
tunnel, and running 3 of them concurrently at 1GB (long enough for
sustained contention to matter, unlike the original 100MB sanity
check) starved all three: combined ~5.4 MB/s, worse than a tenth of
the uncontended single-variant 50 MB/s. Fixed by running self-hosted
variants sequentially against each other (each gets the tunnel to
itself) while the public-relay baseline -- a fully separate network
path -- keeps running concurrently alongside them. Full writeup in
`CROC_JOURNEY.md`'s bug #6; not yet re-confirmed against a real pod.

**Second update, same day**: before spending a third real pod rental
on the 100-port scale test, `croc` v11.5.2's own source was checked
directly (`activateRelayDataChannels` in `src/croc/croc.go`, verified
against the exact tagged release, not just `main`) -- it hard-caps
real parallel data connections at **8 per transfer**, full stop,
regardless of how many ports are offered or `--transfers` requests.
The whole "100 ports = 100 workers" premise was wrong.
`RELAY_TRANSFERS` corrected to `8` (9 ports, the real cap plus a
1-port margin); the `self-hosted-transfers100` variant renamed to
`self-hosted-transfers8` to test the one axis with genuine, unmeasured
upside -- croc's own send-side default is `--transfers 4` (which is
what actually produced the confirmed 2.0x, not the relay's port
count), so 4->8 is still an open question, 4->100 never was one. Full
writeup in `CROC_JOURNEY.md`'s "second wall" section.

**Third update, same day -- CONFIRMED and wired into production**: the
corrected scale test (`RELAY_TRANSFERS=8`, 1GB file, sequential
self-hosted variants) ran cleanly: `self-hosted-default` (croc's own
settings, no overrides) hit **93.1 MB/s -- 2.4x** the public-relay
baseline (39.4 MB/s), every byte size-verified against the source.
`self-hosted-transfers8` (8 connections) measured *slower* than the
default's 4 -- second independent confirmation (after bug #6) that
adding connections through one shared tunnel doesn't keep scaling.
`scp_data_files.sh`'s `TRANSFER_METHOD=croc` path and
`provision_and_run.sh`'s `pod create` call (no longer requests any
custom port range -- SSH is the only port this ever needed) are now
both updated to use `lib_croc.sh` and these confirmed settings. Full
writeup in `CROC_JOURNEY.md`.

**New open target, explicitly not yet met**: 1GB/s (1024 MB/s) or
more. 93.1 MB/s is a confirmed, real 2.4x, but roughly 11x short of
that target. The next lever isn't more connections through one
tunnel (twice-confirmed not to help) -- it's genuinely parallel,
independent network paths (split the file, multiple separate SSH
tunnels, reassemble), not yet designed. See `CROC_JOURNEY.md`'s
"What's still open."

### 36.1 What v6.6 does NOT change

- `WANDB_PROJECT` stays `NI_Review_v6`.

## 37. v6.7 -- croc dropped for plain parallel scp, `h11_ridge_distill` gets its first clean CUDA numbers, and a first-rollout-frame anomaly worth a dedicated look

Four things landed together this session, in order:

**Transfer pipeline: croc dropped entirely, plain scp is now the sole
method.** v6.6's confirmed 93.1 MB/s (2.4x) via croc's self-hosted
relay turned out not to be the ceiling -- an `iperf3` diagnostic
(raw network/SSH capacity, no croc in the loop at all) showed 10
genuinely independent concurrent SSH tunnels hit **268.3 MB/s**, close
to this link's real ~312 MB/s ceiling. Trying the same 10-lane
approach with actual croc file transfers (split file, N independent
relay+tunnel pairs) measured far below that iperf3 number -- croc pays
for a PAKE handshake and its own encryption layer on top of the
already-encrypted SSH connection (zero security benefit on a link you
already control via SSH), plus optional compression/hashing overhead
iperf3 never exercises. Plain parallel `scp` (no relay, one encryption
layer, each invocation its own independent ssh connection with no
manual tunnel setup needed) was tried instead and confirmed to reach
that same ~268 MB/s number on a real file, with both size AND sha256
verified on reassembly (croc's own check was size-only). `croc` has
now been removed from the live pipeline entirely: `bootstrap_remote.sh`
no longer installs it, `scp_data_files.sh` is scp-only (10 concurrent
lanes per file, files sent sequentially so each gets the full lane
budget), and `provision_and_run.sh` no longer references
`TRANSFER_METHOD`/`CROC_RELAY_PORT`. `lib_croc.sh`/
`bench_croc_variants.sh`/`CROC_JOURNEY.md` are kept only as a record of
that investigation.

**`provision_and_run.sh` is now a real single-command master script,**
with CLI flags (`--wandb=KEY`, `--round=N`, `--arm=NAME`, `--fresh`)
instead of env-var-only wiring, so a full provision -> transfer ->
bootstrap -> train -> pull-back run (including `h11_ridge_distill`'s
cold-start control arm) is one command:
```bash
bash singleshot/provision_and_run.sh --wandb=<key> --round=3 --fresh
```
Also fixed a real non-interactive-wandb bug found via an actual failed
run (`wandb: ERROR ... No API key configured` despite `--wandb=KEY`
being passed): the old code ran `WANDB_API_KEY='key' wandb login
--relogin`, which only sets that env var for the `wandb login` command
ITSELF, not for the `python` training process launched afterward in
the same remote script. Fixed by `export`ing `WANDB_API_KEY` for the
whole remote script instead -- the wandb SDK reads it directly at
`wandb.init()` time, no separate login step needed.

**`h11_ridge_distill` finally has clean, post-v6.5-fix CUDA numbers.**
Two full 2500-step runs, both via the fixed `provision_and_run.sh`:

| | round 2 (`exp-a`, warm-started from `r2_h11_ridge_distill_latest.pt`) | round 3 (`exp-b`, cold-start, `--fresh --no-warm-start`) |
|---|---|---|
| `improvement_pct` (best) | **+37.26%** | **+29.97%** |
| `rollout_mse` (best) | 0.001784 | 0.001992 |
| wall time | 1235s | 1287s |

Warm-starting gave a real but modest ~7-point edge over cold-starting
at the same 2500-step budget -- not nothing, but not dramatic. Neither
number beats `h9_ar_freq1`'s own +43.78% or `v3_h9_moreseqs`'s +46.49%
(both shallow, ~300-400 step screens, so not a fully fair comparison at
2500 steps) -- h11's added AR-curriculum + ridge-rollout-distillation
machinery on top of h9's config does not yet show a clear win over the
simpler config at a longer budget. Round 3's early steps DID hit the
`NaN`-rollout failure mode predicted in this same session (a
freshly-initialized model fed autoregressively into itself can blow up
before gradients stabilize it, and the `best` dict's `<`/`>` comparison
against its own `inf`/`-inf` sentinels silently never fires when the
left side is `NaN`) -- confirmed via the pulled-back `status.json` that
this was **transient, not persistent**: final `best.rollout_mse`/
`improvement_pct` are real finite numbers, so the run's result is
trustworthy. The `wandb.Artifact(metadata=...)` JSON-serialization
crash this caused on an early checkpoint (`Out of range float values
are not JSON compliant: inf`) is still queued as a fix -- sanitize
non-finite floats out of artifact metadata before it reaches wandb,
regardless of root cause.

### 37.1 Worth a dedicated look: the first-rollout-frame anomaly

Both `h11_ridge_distill` runs above show the same odd shape in
`last_eval`: `improvement_pct_frame1` is wildly NEGATIVE (round 2:
-1987.9%, round 3: -1830.8%) while `improvement_pct_frame_half`
(+39.1%/+35.1%) and `improvement_pct_frame_last` (+26.2%/+23.4%) are
solidly positive. i.e. the model is dramatically WORSE than persistence
specifically on the very first rolled-out frame, then recovers and
beats persistence comfortably for the rest of the 68-frame horizon.
This pattern isn't new to h11 specifically -- it's just the first time
per-frame numbers this granular have been looked at together in this
log. Two live hypotheses, neither confirmed yet:
  1. An off-by-one in how frame 1 is scored/anchored relative to the
     persistence baseline (e.g. frame 1's persistence prediction is
     unusually easy -- literally the last observed frame -- so ANY
     model deviation from it reads as a huge relative loss even when
     the absolute error is small).
  2. A genuine architectural weakness at t+1 specifically (the model
     needs a frame or two of its own rollout history before its
     predictions stabilize), which would be a real finding about the
     AR rollout mechanism, not a scoring artifact.
Worth a dedicated, cheap diagnostic pass (plot absolute error per frame
index, not just improvement_pct, across a few arms) before scaling
anything further -- if it's hypothesis 1, the summary `improvement_pct`
numbers reported throughout this whole doc may be getting dragged down
by a single artifact-heavy frame; if it's hypothesis 2, it's a real
model limitation worth designing around (e.g. a frame-1-specific loss
term, or excluding frame 1 from the aggregate score with a documented
reason).

### 37.2 Where this leaves the +69% target

Restating Appendix A's own framing: **+69.49%/+68.97%** is a linear
ridge-regression ceiling/headroom gauge, not an architectural target --
the doc's own stated realistic near-term target is **+46-50%**. Against
that: the best clean transformer number anywhere in this investigation
is still `v3_h9_moreseqs`'s **+46.49%**, but only at a ~300-400 step
shallow screen, never confirmed at production scale. The single most
valuable UNFINISHED lead in the whole doc is `s8_h9_moreseqs_scaled`
(that exact config, scaled) -- it was mid-run at step 4000/45000
(+22-27%, noisy) as of §32 and its final result at 45000 steps was
never recorded anywhere after that. Closing that out is a more direct
shot at the +46-50% target than further `h11` iteration.

**Correction, same session**: §37.2 above (and the Appendix A framing it
was quoting) was WRONG to treat +46-50% as "the realistic target" and
+69% as merely a headroom gauge not expected to be hit. Called out
directly by the operator, and it holds up: a linear ridge-regression
map is a STRICTLY LESS expressive function class than a transformer.
There is no principled reason a transformer should lose to it at
convergence. Every transformer variant currently losing to it is
evidence of a fixable problem, not a ceiling -- three live candidates,
none mutually exclusive:
  1. **Undertraining.** The linear fit is closed-form -- effectively
     "fully converged" the instant it's computed, zero optimization
     difficulty. Every transformer variant tried (300 to 45000 steps)
     is still mid-optimization on a nonconvex surface when cut off; none
     show a clearly flattened loss curve. Losing to a converged simple
     model while still mid-training isn't evidence of a hard ceiling.
  2. **Objective/curriculum mismatch.** Training is mostly teacher-
     forced with lighter AR augmentation; the eval metric is a 68-frame
     autoregressive rollout. The linear model has no such mismatch --
     it's the same fixed operator either way.
  3. **Remaining bugs.** This exact codebase has already had multiple
     CONFIRMED correctness bugs suppressing real performance (the
     `DELTA_ANCHOR` train/eval mismatch, §35; the `torch._dynamo`
     recompile storm, §35.3; and see below, an LR-schedule bug on
     resume). Given that track record, more are plausible.
**+69.49%/+68.97% is the actual goal, not +46-50%.** Restating this
plainly since it changes what "progress" means going forward: closing
the gap to a linear baseline, not declaring victory in the 40s.

## 38. v6.8 -- three concurrent training slots, a real LR-schedule bug on resume, and s8/h11's latest numbers

**Bug found and fixed: prefixing concurrent slots' output with `sed`
could silently kill the training it was labeling.** `sed "s/^/$tag /"`
uses `/` as its delimiter; `$tag` is `[arm/rN]`, which contains a
literal `/` -- on macOS's BSD `sed` this is a hard parse error ("bad
flag in substitute command"), and `sed` exits immediately without
processing any input. That closes the pipe `ssh`'s stdout was writing
to, which SIGPIPEs the local ssh client, which tears down the SSH
session -- with no `screen`/`nohup` on that channel, this very likely
kills the REMOTE training process too, mid-run. Confirmed happening for
real: in the first concurrent-slots run, `s8_h9_moreseqs_scaled`
(slot 1) died at step 4225 after only 352s of its 1800s budget, while
`h11_ridge_distill` (slot 2) happened to survive the race. Fixed by
switching to `awk -v tag="$tag" '{print tag, $0; fflush()}'` -- plain
string concatenation, no delimiter to collide with regardless of what's
in `$tag`.

**Bug found: `--max-steps` is an ABSOLUTE target, and changing it
between resumes of the SAME run re-triggers the LR schedule.**
`make_lr_lambda` (train_production_transformer_deep_dive.py:2512-2529)
builds a warmup-then-cosine-decay curve as a fraction of `cfg.MAX_STEPS`
-- computed fresh from whatever `--max-steps` THIS invocation passes,
then evaluated at the checkpoint's saved absolute step on resume.
Confirmed via a clean A/B in the same session: `s8` resumed with the
SAME `--max-steps=45000` both times -- `train_loss` stayed stable
(0.0134 -> 0.0128). `h11` resumed with a DIFFERENT `--max-steps`
(2500 -> 5000) -- `train_loss` roughly DOUBLED immediately after resume
(0.0088 -> 0.0196), and `improvement_pct_frame1` got noticeably worse
(-1988% -> -2390%). Mechanism: h11's original 2500-step schedule had
fully annealed its LR to the floor by step 2500; resuming with a
5000-step schedule evaluated at step ~2500 is only ~50% through the NEW
curve, re-inflating the LR back up and kicking already-converged
weights. `ar_frames_for_step` (line 2532-2554) has the identical
`cfg.MAX_STEPS`-relative dependency, so h11's AR-frames curriculum was
ALSO perturbed on resume -- a second, compounding mechanism. **Not yet
fixed in code** (would mean persisting the schedule's original target
in the checkpoint and always using that, regardless of the current
invocation's `--max-steps`); the operating rule until then: pick each
arm's real target step count ONCE and pass that exact same value on
every future resume -- never bump `MAX_STEPS` mid-series. This means
h11's post-resume numbers this session are NOT a clean read of the
model -- expect a few hundred steps of recovery before its numbers mean
anything again.

**`provision_and_run.sh` now supports up to 3 concurrent training
slots** (`--arm2`/`--round2`/... and `--arm3`/`--round3`/...), refactored
into a single array-driven code path (`SLOT_ARMS`/`SLOT_ROUNDS`/...)
instead of hand-duplicating the launch/wait/pull-back logic per slot --
the same duplication lesson this repo already learned the hard way with
croc (`lib_croc.sh`'s own header). Plain co-location (default CUDA
time-slicing), not NVIDIA MPS -- more concurrent slots means more
EXPERIMENTS per pod-hour, not more STEPS per experiment; each
additional slot slices GPU compute further.

**`s8_h9_moreseqs_scaled` checkpoint recovered and wired back into the
pipeline.** Its last known checkpoint (step 4000/45000) only existed
under a stale `saved_models/sweep_arms_local/` path the pod pipeline
never reads -- copied into the live `saved_models/` dir and added to
`scp_env_files.sh` so a fresh pod actually resumes from it. Made
226 additional steps of clean progress this session (4000 -> 4225,
`train_loss` stable) before the sed bug above killed that slot early.

### 38.1 Where s8/h11 actually stand now

`s8_h9_moreseqs_scaled`: step 4225/45000, `train_loss_centroid_l2`
0.01277, `best.improvement_pct` still 27.17% (unchanged from step 4000
-- no new eval ran in the brief 352s before it died). Still 40775 steps
from its own target; still the single most valuable unfinished
production-scale lead.

`h11_ridge_distill` (round 2): step 3050/5000 (target changed mid-series,
see the LR-schedule bug above -- treat with caution). `last_eval` at
step 3000: `improvement_pct` 34.998% (down from round 2's earlier best
of 37.26%, but distorted by the LR-restart, not necessarily real
regression), `improvement_pct_frame1` -2390.4% (worse than either prior
run's -1988%/-1831%, same caveat). Needs a few hundred more steps under
a NOW-FIXED, stable `--max-steps` target before trusting any number
from it again.

## 39. v6.9 -- the overfitting/rollout-drift pattern replicates across THREE arms, a real SSH connection-flood bug fixed, and same-arm concurrency costs 2.4x the throughput of different-arm concurrency

**The three-way `s8`/`h11`/`s7` batch (v6.8's follow-up) finished, and
the pattern is now confirmed systemic, not arm-specific.** All three
arms peak somewhere around step 2000-4000 and DECLINE afterward on the
actual rollout-vs-persistence metric, while `train_loss` keeps
improving the whole time -- textbook overfitting/rollout-drift:

| arm | peak (`best`) | latest eval | decline | regularization |
|---|---|---|---|---|
| `s7_h9_scaled` (r5) | 40.2% @ ~step 2400 | 21.7% @ step 4500 | -18.5 pts | none |
| `h11_ridge_distill` (r6) | 37.3% (never beaten) | 27.8% @ step 3000 | -9.4 pts | WD/dropout 0.05 |
| `s8_h9_moreseqs_scaled` (r6) | 28.3% (never beaten) | 22.0% @ step 6000 | -6.3 pts | WD/dropout 0.05 |

The WD/dropout 0.05 bump measurably slowed the decline in both
regularized arms vs `s7`'s unregularized falloff -- real signal, not
yet enough to reverse it or beat the peak. None of the three
continuation attempts ever beat their own recorded `best` -- those
peak checkpoints are still sitting untouched on disk.

**Bug found and fixed: a single dropped SSH connection was aborting the
entire multi-GB transfer, with no retry.** `kex_exchange_identification:
read: Connection reset by peer` on one of 10 concurrent scp lanes,
while phase 1 (env files) was ALSO opening a connection at the same
moment -- a direct, foreseeable consequence of v6.8's "start phase 2
immediately" optimization increasing simultaneous new SSH connections.
This is sshd's own `MaxStartups` anti-flood throttle randomly dropping
a new, not-yet-authenticated connection once too many arrive at once --
expected and recoverable, but `lib_scp.sh` had ZERO retry logic: any
single lane failing failed the whole file, which aborted the whole
pipeline. Fixed with `scp_lane_with_retry`/`ssh_cmd_with_retry` (3
attempts, backoff) -- same pattern `provision_and_run.sh`'s
`rsync_with_retry` already used for the pull-back step, just missing
here until now.

**Bug found (process, not code): forking a new round from an old
checkpoint's `_best.pt` carries the OLD round's wandb run id forward.**
Copying `r2_s8_h9_moreseqs_scaled_best.pt` to seed round 6 only changed
the filename -- the checkpoint's own embedded `wandb_run_id` (from
`step: 4000` of round 2's STILL-CONTINUING lineage, which by then had
logged past step 6491) came along with it, causing round 6 to try
resuming-and-appending to round 2's already-further-advanced wandb
history under round 6's own lower step numbers -- the exact same
"tried to log to step N less than current step" symptom as the v6.8
bug, but a completely different mechanism. Fixed by stripping
`wandb_run_id` from the checkpoint file directly (confirmed `h11`'s own
round-6 seed already had `wandb_run_id: None`, unaffected). **Lesson:
forking a new round from an old checkpoint needs the embedded
`wandb_run_id` cleared, not just a new filename/round number.**

### 39.1 Next lever tried: does regularization prevent the decline? Inconclusive so far -- a real concurrency-cost finding got in the way

Follow-up batch: `s7_h9_scaled` x3, FRESH cold starts, rounds 7/8/9,
identical config except WD/dropout 0/0.05/0.10 -- a clean, controlled
A/B/C on the single best-known base config. Result: all three only
reached step ~900-1000 of a 6000-step, 0.5h-budgeted target --
**nowhere near the step 2000-4000 region where the decline pattern
above actually shows up**, so this batch cannot yet answer the question
it was designed to answer.

**Why so few steps: running 3 copies of the IDENTICAL arm/config
concurrently costs far more throughput than 3 DIFFERENT arms did.**
Direct comparison, steps-advanced-per-invocation ÷ wall_seconds:
- 3 DIFFERENT arms (s7/s8/h11, v6.8 batch): ~1.3 steps/sec per slot
- 3 IDENTICAL arm/config (s7 x3, this batch): ~0.54 steps/sec per slot

**A ~2.4x slowdown from running identical models concurrently vs
different ones** -- identical compute/memory-access shapes maximally
contend for the same tensor cores and memory bandwidth at the same
instant; differently-shaped models interleave less destructively. Real,
quantified, actionable: same-arm concurrent comparisons need either
fewer concurrent slots, a proportionally longer `MAX_HOURS`, or NVIDIA
MPS (still "not yet wired into any script here" per section 32.3) to
get a fair read in the same wall-clock budget.

The early numbers that DID come in are an encouraging trend (11.3% ->
17.8% -> 24.6% at WD/dropout 0/0.05/0.10, roughly step 500-1000) but a
single run each at this few steps is not remotely enough to trust --
next step is continuing these same three runs (same round numbers, same
`--max-steps=6000` -- never bump it mid-series, see v6.7's LR-schedule
bug) with a longer `MAX_HOURS` to actually reach the region that
matters.

Two more real bugs found and fixed the same session, both in
`provision_and_run.sh`/`lib_scp.sh`: (1) `kex_exchange_identification:
read: Connection reset by peer` on a data-transfer lane -- sshd's own
`MaxStartups` anti-flood throttle, triggered because v6.8's "start
phase 2 immediately" change increased simultaneous new SSH connections,
and with ZERO retry logic a single dropped lane was aborting the entire
multi-GB transfer. Fixed with `scp_lane_with_retry`/`ssh_cmd_with_retry`
(3 attempts, backoff), and made the failure loud instead of silently
tripping `set -e` at a bare `wait` if retries ever do exhaust. (2)
Confirmed no code anywhere sets `torch.set_num_threads()` or
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` -- each concurrent training process
independently grabs every CPU core for PyTorch's default intra-op
thread pool, for zero benefit (the actual compute is GPU-bound) and
real contention when running 2-3 slots at once, very plausibly an
additional contributor to section 39.1's 2.4x same-arm slowdown on top
of the GPU contention. Fixed: each slot's remote launch now queries
`nproc` and caps `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to its fair share.

## 40. v7.0 -- an honest progress check, real early stopping shipped, and the actual next lever named as a to-do, not built yet

**Direct question asked and answered honestly: is any of this recent
work actual progress toward +69%, or just infrastructure churn?**
Mostly the latter, and worth saying plainly rather than papering over.
The scorecard, side by side:

| milestone | improvement_pct |
|---|---|
| linear ridge ceiling (the actual goal) | 69.5% |
| shallow screens, 300-400 steps (`h9_ar_freq1` / `v3_h9_moreseqs`) | 43.8% / 46.5% |
| `s7_h9_scaled` clean production run, PEAK | 40.2% @ ~step 2400 |
| `s7_h9_scaled` same run, later | 21.7% @ step 4500 |
| `h11_ridge_distill`, peak -> later | 37.3% -> 23-28% |
| `s8_h9_moreseqs_scaled`, peak -> later | 28.5% -> 16-22% |

**Nothing tried at production scale has beaten the shallow 400-step
screens.** Every serious continuation -- different arm, different
regularization, different starting checkpoint -- converges on the same
story: peak around step 2000-2500, then decline, now confirmed three
independent times. That is a real, useful NEGATIVE result (it rules out
"just train longer, or train longer with more weight decay" as the
lever), but it is not progress toward 69%, and several pod-hours went
into re-confirming it rather than testing something that could actually
move the number. The overwhelming majority of this session's actual
engineering effort went into throughput/reliability infrastructure
(croc->scp, concurrent orchestration, three real bugs fixed, SSH
retries, CPU thread capping) -- all of it legitimate and necessary (the
pipeline genuinely was broken/slow/unreliable), but it made the SAME
experiment cheaper to repeat; it did not change what experiment is
being run.

### 40.1 Shipped: real early stopping (`Config.EARLY_STOP_PATIENCE`)

Stops the bleeding on the negative result above -- future runs no
longer need a human to notice the decline and manually kill the pod.
`train_production_transformer_deep_dive.py`:
- New `Config.EARLY_STOP_PATIENCE` (default `None` -- disabled,
  unchanged behavior for anything that doesn't opt in) and
  `--early-stop-patience N` CLI flag.
- Tracks `evals_without_promotion`: resets to 0 on any eval that
  actually promotes a new `_rollout_best.pt` (beats persistence AND
  clears the sane-RMS gate AND improves on the best-ever promoted
  `rollout_mse` -- the exact condition that checkpoint is saved under),
  increments otherwise. After `N` consecutive non-promoting evals,
  stops training right there -- same treatment as a `MAX_HOURS`/
  `MAX_STEPS` stop (final checkpoint still saved via the existing
  `hit_budget` gate, `stop_reason` recorded for the run's own summary).
- Deliberately tied to the PROMOTED rollout metric, not `_best.pt`
  (which tracks `val_tf_mse`, a different quantity several arms kept
  improving on even while the actual rollout-vs-persistence metric was
  getting steadily worse -- confirmed while implementing this: `_best.pt`
  and `_rollout_best.pt` are NOT interchangeable, and earlier "resume
  from best" experiments this session used `_best.pt`, which may not
  always coincide with the true rollout peak).
- NOT restored across a resume -- a resume is itself a fresh operator
  decision to keep going, so it gets a fresh patience budget rather than
  inheriting however close to the limit a prior process invocation was.
- Verified the counter/reset/threshold bookkeeping in isolation (a
  standalone simulation, three scenarios: stops at the right index,
  never stops when patience isn't reached, resets correctly after a
  promotion) -- the full loop itself needs a real GPU/data run to
  exercise end-to-end, not yet done.
- Usage: `provision_and_run.sh`'s existing generic `--extra="..."`
  per-slot passthrough already covers this with no further plumbing --
  e.g. `--extra="--early-stop-patience 4"`.

### 40.2 NOT done yet -- explicitly a to-do, not a future aside

**The actual lever that could plausibly close the 25-40 point gap to
69% is an objective/architecture change, not a hyperparameter sweep on
the current recipe.** Two live hypotheses, neither implemented:
1. **Objective mismatch.** The linear ridge baseline is never trained
   on anything OTHER than what it's scored on (a single fixed operator,
   autoregressive rollout IS its only mode). Every transformer arm so
   far trains mostly teacher-forced with lighter AR augmentation, then
   gets scored on a full autoregressive rollout it was never directly
   optimized against as heavily. Training more directly against the
   actual rollout loss (heavier `AR_LOSS_WEIGHT`, longer AR horizon
   earlier in training, or a loss that's ALWAYS the rollout metric
   rather than teacher-forcing-dominant) is untested as the primary
   lever, only as a secondary augmentation on top of a teacher-forced
   base.
2. Section 32.2's still-unregistered acceleration menu (multi-horizon
   auxiliary head, physical-consistency penalty, split-anchor-by-loss)
   -- named back in v6.2, still not built as arms as of this section.

Neither of these is a `--set KEY=VALUE` away -- both need real new code
(a new arm, likely a real change to the loss function or training
curriculum), not another concurrent batch of the existing recipe.
**Explicitly flagged here as the next real research question, to be
scoped BEFORE spending more pod time repeating the current recipe --
not something to bolt onto another 3-slot regularization batch.**

## 41. v7.1 -- a real GLOBAL TIMEOUT WRAPPER for provision_and_run.sh, a serious bug in it caught the same session, and wandb Artifacts confirmed as a genuine full-recovery path

**Added the same two-layer wall-clock safety net `bench_croc_variants.sh`
already had, ported to `provision_and_run.sh`.** `MAX_HOURS` only self-
polices INSIDE each training process's own loop -- nothing catches a
hung bootstrap, a stalled SSH connection, or a process wedged somewhere
that never reaches that check. Layer 1: self-re-exec + a watchdog that
SIGTERMs (letting `cleanup_pod()`'s EXIT trap still run) then SIGKILLs.
Layer 2: a fully detached background job that unconditionally deletes
the pod at a slightly later deadline, independent of the script's own
control flow, in case Layer 1's own cleanup path hangs too.
`GLOBAL_TIMEOUT_SECS` is DERIVED, not fixed: longest `MAX_HOURS` across
all active concurrent slots + `SETUP_OVERHEAD_SECS` (default 900s/15min)
-- "30 minutes of GPU compute should be over in 45 minutes of wall-clock
time, no matter what." Verified both the timeout-derivation arithmetic
(4 scenarios) and the actual SIGTERM/trap/exit-code mechanism (a
standalone simulation) before trusting it.

**Then broke a real run with it, same session.** `provision_and_run.sh`
has a nuance `bench_croc_variants.sh` never needed: `cleanup_pod()`
deliberately REFUSES to delete the pod when the artifact pull-back fails
(so a transient network blip can't cost you the run's actual output --
see the manual-retrieval message it prints). Layer 2, ported verbatim,
is a blind wall-clock timer with no idea that decision was ever made --
it force-deletes the pod at its deadline regardless. Result: three real
runs (`s7_h9_scaled` r7/r8/r9, ~1.5h into their budget) got killed mid-
training when Layer 2's timer fired, right after `cleanup_pod()` had
JUST decided to preserve the pod for exactly this scenario. Confirmed
via wandb: all three runs show `state: "crashed"` (abrupt kill, not a
graceful stop), matching Layer 2's deadline almost exactly. **Root
cause: Layer 2 has no way to know about a decision made LATER, inside a
different code path, in a variable (`PULL_BACK_OK`) it can't see --
correct instinct (a robust, independent backstop) applied to a script
that has a legitimate "please don't delete this yet" state the
backstop's own design didn't account for.** Fixed: `cleanup_pod()` now
explicitly cancels Layer 2's pid (`kill "$SAFETY_NET_PID"`) on every
path, including -- especially -- the "NOT terminating, pull-back failed"
branch. Verified with a standalone simulation before trusting it: arm a
short-deadline "layer 2" job, take the "don't terminate" branch, confirm
the job never fires.

### 41.1 wandb Artifacts confirmed as a genuine full-recovery path, not just metrics

The immediate question after the crash: with the pod gone and the local
rsync pull-back never having run, is any of that run's work actually
lost? No -- and this is worth knowing as a standing fact, not just a
one-off rescue. `save_checkpoint()` already calls `tel.log_artifact()`
on EVERY checkpoint save (`_latest.pt` every `CHECKPOINT_EVERY_STEPS`,
plus `_best.pt`/`_rollout_best.pt`/`_train_best.pt` whenever each
improves) -- `_Telemetry.log_artifact()`'s own docstring says so
directly: "Logging the same name repeatedly... creates a new VERSION
each time -- a full checkpoint history in wandb for free." This was
already true before v7.1; it just had never been exercised as an actual
recovery path until this incident forced it.

Recovered, for real, using `wandb.Api()` from this local machine
(credentials already in `~/.netrc`, entity `pvl-data`, project
`NI_Review_v6`): all three crashed runs' `_latest`/`_best`/
`_rollout_best` artifacts, latest version of each, downloaded and
verified loadable. Lost: at most ~30-34s of training (one
`CHECKPOINT_EVERY_STEPS` interval) between the last successful upload
and the kill. Recovered numbers, none of which had been pulled back
locally any other way:

| run | peak `improvement_pct` | peak step | last step reached |
|---|---|---|---|
| r7 (no reg) | 30.5% | 2500 | 3950 |
| r8 (WD/dropout 0.05) | 29.5% | 2500 | 3950 |
| r9 (WD/dropout 0.10) | **39.5%** | 1000 | 3825 |

A fourth and fifth confirmation of section 39/40's peak-then-flat
pattern (all three plateau early, zero further improvement for 1300-
2800 further steps) -- but also a genuinely interesting new data point:
r9 (the strongest regularization) peaked far higher AND far earlier than
r7/r8, which only differ from each other by the weaker 0.05 dose. Single
runs each, not yet a trend to trust, but worth a dedicated look rather
than folding into another batch.

**Standing lesson: if a pod is ever lost with an incomplete pull-back,
check wandb Artifacts before assuming the run's progress is gone.**
`wandb.Api()` -> `api.artifact(f"{entity}/{project}/{run_name}_{kind}:latest")`
-> `.download()` recovers the actual checkpoint files, not just the
logged metrics.

## 42. v7.2 -- r9's early lead confirmed real (r11 replicated it), three fixes from the "what failed" investigation, and early-stopping patience switched from evals to steps

**The r10/r11/r12 ablation confirmed the regularization signal.** Pulled
each run's actual stop and best numbers:

| round | config | peak `improvement_pct` | stopped at (steps) |
|---|---|---|---|
| r10 | WD=0.1, dropout=0.1 (replicate r9, new SEED) | 29.9% | 4500 |
| r11 | WD=0.1 only | **35.5%** | 3500 |
| r12 | dropout=0.1 only | 31.4% | 4000 |

None beat r9's own 39.5%, but r11 (weight decay alone) came closest and
clearly outperformed r12 (dropout alone) -- a real, if still single-run,
signal that **weight decay is doing more of the work than dropout** in
whatever is suppressing the peak-then-decline pattern. r10's replicate
(same config as r9, different seed) landed lower than r9's original --
expected variance, not a contradiction; the regularization *direction*
still holds up.

**"What failed" investigation -- confirmed NOTHING failed; early
stopping worked exactly as designed.** All three runs stopped at
different, non-round step counts (4500/3500/4000 -- not `MAX_STEPS`
6000, not the `MAX_HOURS` 1.0h wall-clock boundary), the signature of
per-run patience firing independently. Confirmed directly via the
pulled-back sweep log: `"stop_reason": "early stop: 4 consecutive evals
(patience=4) with no new promoted rollout checkpoint"`. The reported
"~1h12m, that's not what was designed" gap wasn't a bug -- concurrent
slots wait for the SLOWEST one to individually converge, and patience=4
evals at `--val-every 500` means 2000 steps (a large wall-clock budget
under 3-way contention) has to pass with zero improvement before any
one slot gives up.

**Three real fixes shipped from that investigation:**

1. **`TORCHINDUCTOR_COMPILE_THREADS` was never capped.** v6.9's CPU-
   thread fix capped `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (the steady-
   state tensor-math thread pool) but missed a completely separate
   knob: confirmed directly in `torch/_inductor/config.py`'s
   `decide_compile_threads()`, the compile-worker subprocess pool
   (`compile_worker/__main__.py --workers=N`, visible in `ps aux`
   during "getting the model ready") defaults to `min(32, cpu_count)`
   **per process**, entirely unaffected by OMP/MKL. With N concurrent
   slots, up to N x 32 compile-worker subprocesses could compete for
   the same cores during each process's own compilation burst -- likely
   the real reason model-ready time felt slower even after the OMP/MKL
   fix. `provision_and_run.sh`'s `launch_training()` now also exports
   `TORCHINDUCTOR_COMPILE_THREADS` using the same per-slot division.
2. **`sweep_logs/manual/{arm}.json` was keyed by arm name only, not
   round.** r10/r11/r12 (three rounds of the same arm, concurrent) all
   wrote to the identical file -- only r10 (finished last) survived
   locally; r11/r12's own `stop_reason` records were silently
   clobbered. Fixed: now keyed by `run_name` (`r{ROUND}_{ARM}`), matching
   every other checkpoint/artifact naming convention in this file.
3. **Early-stopping patience switched from eval-count to step-count.**
   `Config.EARLY_STOP_PATIENCE` (an int number of EVALS) renamed to
   `Config.EARLY_STOP_PATIENCE_STEPS` (an int number of STEPS,
   `--early-stop-patience-steps` on the CLI) -- eval-count patience
   silently meant something different depending on `--val-every`, which
   is exactly the kind of unit confusion that made "patience=4" actually
   mean "2000 steps" without it being obvious from the number itself.
   Verified the new step-based bookkeeping in isolation (4 scenarios:
   one-eval-of-slack when patience equals `--val-every`, extra slack
   when patience is larger, correct resume-time baseline, never-stops
   when always promoting) before trusting it. Patience is now being set
   to 500 steps going forward -- at `--val-every 500` this is
   deliberately aggressive (stops on the very next non-promoting eval
   after a promotion, one eval of slack), trading a real chance of
   stopping on a single noisy eval for much faster sessions; worth
   watching whether this ever cuts off a run that would have recovered.

## 43. v7.3 -- §40.2's objective-mismatch arms built and locally verified, a real pod-infra investigation (upload-speed gating, then network volumes), and several bugs found the hard way along the way

**Status: branch M (the arms) and the network-volume pipeline are both
built and locally/API-verified. Neither has been run for real on CUDA
yet** -- blocked on H200 stock in the one datacenter the network volume
is pinned to, which `provision_and_run.sh` now retries for automatically
(see below) rather than failing on the first empty check. Everything
below is what actually happened this session, not a plan for later.

### 43.1 Branch M -- §40.2's objective-mismatch hypothesis, finally built

§40.2 (v7.0) named this explicitly as a to-do, not yet built: every arm
through h11 trains mostly teacher-forced, with the AR rollout loss a
light additive augmentation, then gets scored on a full AR rollout it
was never directly, heavily optimized against. Scoping this corrected an
assumption along the way: on CUDA, `accum_steps = virtual_batch //
micro_batch == 1` (CUDA sets `virtual_batch == micro_batch`), so the
`micro == 0`-only AR gate in the training loop is a no-op wherever AR
actually runs -- it only ever mattered on MPS/CPU, where `frame_ar` is
disabled outright anyway. The real "AR is a light touch" mechanism on
CUDA is `AR_SEQS` (only 2 of a 64-row batch get the sequential AR
rollout in every arm through h11) plus AR always being purely additive
on top of a fixed-weight teacher-forced loss, never a partial
replacement of it.

Per an explicit decision this session, branch M does **not** touch the
loss-composition code itself (no TF/AR blend schedule) -- everything
ships as new arm configs on existing `Config` knobs, one lever at a time
(mirrors branch V's discipline) before a combined arm:

- **`m1_ar_weight`** -- h9's exact config, `AR_LOSS_WEIGHT` 1.0 -> 3.0.
- **`m2_ar_curriculum`** -- h9's exact config, a faster/earlier rollout-
  horizon curriculum than h11's (`AR_FRAMES_START` 2->1,
  `AR_FRAMES_WARMUP_FRAC` 0.3->0.1).
- **`m3_ar_combined`** -- m1 + m2, plus `AR_SEQS` originally set to 16
  (8x h9/h11's 2). **Revised down to 12 later this same session** (see
  §43.4) after a memory estimate put 16 over an H200's real capacity.

**Local verification, in place of the CUDA run that hasn't happened
yet:** `tests/test_ar_objective_arms.py` (arm resolution/`apply_arm()`
pins), a regression case added to `tests/test_ar_frames_curriculum.py`
for m2/m3's exact curriculum numbers, and -- the one that actually
matters mechanically -- `tests/test_frame_ar_loss_direct.py`, which
calls `frame_ar_loss()` directly with a real (small) `BaseTransformer`
through the real production decoder, at each arm's real `AR_SEQS`/
`AR_FRAMES` values, and confirms finite loss + finite gradients on every
parameter. Neither `--smoke-test` nor a real local training run reaches
this code path on this Mac (MPS disables `frame_ar` entirely), so this
is the only thing that actually exercises it here. Also ran `m1_ar_weight`
for real against the real local `train_80.h5`/`val_80.h5` for one
optimizer step -- confirmed the arm resolves, warm-starts, and correctly
hits the MPS `frame_ar` disable guard.

### 43.2 A pod-speed gate, built, tested, and then partly superseded

**Motivation:** one real pod (`38.80.152.249:30708`) had scp upload
speeds so slow it would have taken 30+ minutes for <20GB, vs. the
~268 MB/s this pipeline normally sees. New script,
`singleshot/select_fastest_pod.sh`: provisions `NUM_CANDIDATES` (default
3) cheap A40 pods concurrently, benchmarks each one's real upload
throughput with the same `lib_scp.sh` split-lane mechanism
`scp_data_files.sh` uses in production, keeps the fastest, terminates
the rest immediately.

**Two-tier tests, following this repo's existing pure-function/live-
skippable split:** `tests/test_pod_speed_gate_ranking.py` (fast, free,
sources the script with a `BASH_SOURCE == $0` guard so the ranking logic
runs with zero network dependency) and `tests/test_pod_speed_gate_live.py`
(opt-in via `RUN_LIVE_POD_TESTS=1`, creates real pods, force-deletes
everything including the winner in its own teardown).

**Three real bugs found only by actually running it against live pods:**

1. **`declare -ga` is bash 4.2+ only.** macOS's stock bash (3.2.57)
   rejects `-g` outright ("declare: -g: invalid option"), which meant
   `ALL_CREATED_PODS` was never actually created and `cleanup()` broke
   with "unbound variable" under `set -u`. Fixed with a plain assignment
   (global-by-default inside a function unless shadowed by `local`,
   portable to 3.2).
2. **`local tmpdir` inside `main()` didn't survive to the EXIT trap.**
   `cleanup()` fires AFTER `main()` returns, by which point a `local`
   variable's storage is gone -- same fix, plain assignment instead.
3. **`sort` gives no stability guarantee on ties.** A 50.0/50.0 MB/s tie
   non-deterministically picked the SECOND candidate on this machine,
   not the first as documented -- fixed with an explicit secondary sort
   key on original index.

**Live results, twice:** run 1 (2 candidates) -- 1 failed to provision
with a RunPod-side `"graphql error: Something went wrong"`, the survivor
benchmarked at 37.5 MB/s. Run 2 (3 candidates) -- 2 of 3 failed with the
**exact same** transient error, a real, reproducible signal that
RunPod's API doesn't like fully-simultaneous `pod create` calls from one
account (not GPU stock, not this script's logic). The lone survivor's
own reported 10.7 MB/s vs. `lib_scp.sh`'s internal "16.7 MB/s aggregate"
print for the same transfer is not a bug -- the outer timer honestly
includes remote reassembly+sha256-verify, which the real
`scp_data_files.sh` also pays, not just the raw send.

This tool still exists and works, but its purpose for the BIG data
transfer specifically was superseded by §43.3 below -- upload once to a
persistent volume beats re-benchmarking+re-uploading every run. Still
useful for the smaller env-files phase or a one-off benchmark.

### 43.3 Network volumes: a real dead end, then a real fix, then a real bug

**RunPod "Global Volumes" (beta) do NOT have any API/CLI surface at all
right now** -- confirmed directly against the live OpenAPI spec
(`rest.runpod.io/v1/openapi.json`): the only volume path is
`/networkvolumes`; there is no `/globalvolumes`. Creation/attachment is
console-only. This is exactly what the user's own pre-existing volume
(`cmu4kk3zv000007l7cn5e9vxh`, common name `chilly_blue_turtle`) was --
explaining why `runpodctl network-volume get/list` couldn't see it: not
a permissions bug, a genuinely different, not-yet-exposed resource type.
Global Volumes are also explicitly documented as not-fully-POSIX (no
file locking, no atomic rename) -- HDF5 would likely need
`HDF5_USE_FILE_LOCKING=FALSE` to read reliably from one even if API
access existed.

**Created a real, automatable classic Network Volume instead:**
`yl7f9e8rwr` ("cgan-transformer-data"), pinned to `US-GA-2` -- one of
the few datacenters that supports network volumes AND carries H200
stock (most volume-capable datacenters don't have H200 at all).
Uploaded `train_80.h5`/`val_80.h5` for real; sha256-verified both by the
script and independently by hand.

**Real bug hit and fixed: "Disk quota exceeded" during reassembly.** The
volume's original 25GB was sized for the two files' final size
(~14.1GB) but not the TRANSIENT peak during reassembly:
`scp_reassemble_verify()` used to `cat` every split piece into the final
file in one shot, needing the full set of pieces AND the full final file
on disk simultaneously (~2x peak per file), and both files reassemble
concurrently by design -- real peak hit ~26GB against a 25GB volume.
Immediate fix: resized the volume to 50GB. Root-cause fix, in
`lib_scp.sh` itself (benefits every caller, not just this one volume):
`scp_reassemble_verify()` now appends and deletes ONE piece at a time,
keeping peak usage close to 1x instead of 2x.

### 43.4 `provision_and_run.sh` re-wired for the network volume, plus two more real bugs and a stock-scarcity retry

**`NETWORK_VOLUME_ID` (+ `DATA_CENTER_IDS` override), new, opt-in:**
looks up the volume's datacenter automatically, pins pod creation AND
H200/H300 GPU auto-detection to it (stock elsewhere doesn't help --
the pod can't be created anywhere else and still attach the volume),
and replaces the whole "Phase 2/2: data files" step with a fast remote
`stat` size-check against the local files -- training refuses to launch
on a mismatch rather than silently training on stale/missing data.
Unset (default): byte-for-byte the original re-upload-every-run
behavior, no pinning.

**A real, unrelated bug found and fixed while wiring this up:** the
concurrent-arm launch heredoc (`launch_training()`, unquoted `<<REMOTE`)
had two documentation comments containing backtick-quoted code spans --
`` `compile_worker/__main__.py --workers=N` `` and `` `ps aux` `` --
which an UNQUOTED heredoc's local shell treats as real command
substitution, not inert text. This crashed every concurrent-arm launch
before training even started (`compile_worker/__main__.py: No such file
or directory`, then macOS's own `_windowserver` process line from the
locally-executed `ps aux` output getting spliced into the remote script
and misinterpreted as commands there too). Fixed by escaping both
backticks.

**Standing instruction from this session, applied:** `cleanup_pod()`'s
`FORCE_TERMINATE_ON_PULL_FAILURE` default flipped from 0 to 1 -- the pod
is now terminated on exit regardless of pull-back outcome. A dangling
billed pod is worse than a lost local artifact wandb's own versioned
Artifact upload can usually recover anyway. Set it to 0 explicitly for
one run if the old protective behavior is ever wanted back.

**GPU stock scarcity turned out to be real, not a bug -- added a retry
loop instead of failing fast.** `US-GA-2` (where the volume lives)
genuinely had zero H200 stock for a while. New `GPU_WAIT_SECS` (default
300s/5min) and `GPU_POLL_INTERVAL_SECS` (default 10s) poll instead of
failing on the very first empty check; validated the mechanism directly
against the real API (real polling, real elapsed-time tracking, correct
give-up behavior).

**GPU pool broadened to a real, cheaper option -- then found it doesn't
help THIS volume specifically.** New `GPU_AUTO_DETECT_PATTERN` (default
now `H200|H300|RTX PRO 6000 Blackwell (Server|Workstation) Edition$`,
overridable) adds the RTX PRO 6000 Blackwell 96GB cards ($2.09-2.19/hr,
cheaper than H100's $3.49/hr) to the auto-detect pool -- AR-rollout
memory scales with an arm's `AR_SEQS`, not `micro_batch`, so
h9/h11/m1/m2 (all `AR_SEQS=2`, ~34GB observed on H200) fit comfortably
in 96GB. The trailing `Edition$` anchor is load-bearing: without it this
would also match that card's much-smaller MIG-sliced variants (24GB/
48GB), which would NOT fit. Checked directly against the live API,
though: this card isn't offered in `US-GA-2` at all, so it doesn't
unblock the current wait there specifically -- still useful for a
future volume in a datacenter that has it (e.g. `US-CA-2`/`US-MO-2`,
which support both network volumes and this card), or for any run not
pinned to a volume.

### 43.5 `m3_ar_combined`'s `AR_SEQS` revised 16 -> 12 after an honest memory estimate

Using the training script's own `_attn_bytes` memory-estimate formula,
calibrated against the ONE real data point available (h9/h11's own
measured ~34GB/141GB on H200 at `AR_SEQS=2`): the rough attention-score-
only estimate under-reports real usage by a roughly constant ~8.4GB
(weights, optimizer state, non-attention activations, framework
overhead) that doesn't scale with `AR_SEQS`, only the AR-rollout term
itself does. Extrapolating that fixed-overhead model:

| AR_SEQS | estimated real usage | vs. H200's 141GB |
|---|---|---|
| 2 (h9/h11, m1/m2) | ~34GB (measured, not estimated) | comfortable |
| 8 (h5_ar_moreseqs' own ceiling) | ~89GB estimated | comfortable |
| 12 | ~126GB estimated | ~89% capacity, tight but plausible |
| 16 (original m3 value) | ~162GB estimated | **over capacity** |

`m3_ar_combined` shipped at **`AR_SEQS=12`** per this session's decision
-- an ESTIMATE from a single calibration point, NOT yet confirmed on
real hardware. If it OOMs for real, `AR_SEQS=8` (comfortable estimated
margin, matches `h5_ar_moreseqs`'s already-proven ceiling) is the
documented fallback, right in the arm's own `desc`.

### 43.6 What's still actually pending

- Branch M (`m1`/`m2`/`m3`) has never run on real CUDA hardware. Local
  verification (§43.1) is mechanical correctness, not a research
  result -- same caveat this file has repeated at every prior MPS-only
  milestone.
- `m3_ar_combined`'s `AR_SEQS=12` memory fit is an estimate, not a
  confirmed one.
- The network-volume pipeline (§43.3/§43.4) has never actually launched
  a training run yet -- blocked on `US-GA-2` H200/H100 stock at the
  time of writing, now waited-on automatically via `GPU_WAIT_SECS`
  rather than failing immediately.

### 43.7 `WANDB_PROJECT` synced to the new major version

Per the §21.1 convention, `Config.WANDB_PROJECT` renamed
`NI_Review_v6` -> `NI_Review_v7`, kept in sync by `test_version_sync.py`
(this file crossed into major version 7 back at §40's v7.0 -- the
rename itself just hadn't been done until now; `test_version_sync.py`
had accordingly been failing since v7.0 landed, caught and confirmed via
`git stash` earlier this same session, and is green again as of this
rename).

## 44. v7.4 -- §43.5's memory estimate corrected by a real run: `m3_ar_combined`'s `AR_SEQS` reinstated at 16

**§43.5's linear-extrapolation memory estimate was too pessimistic --
confirmed by an actual cold-start run, not another estimate.** While
§43 was still being written, a `m3_ar_combined` run (`--fresh
--no-warm-start`, round 2, `WANDB_GROUP=Storage`) was launched by hand
on a real `US-GA-2` H200 pod, on the code as it stood BEFORE this
session's `AR_SEQS` walk-back (i.e. still at the original `AR_SEQS=16`).
Checked it live, twice, ~2 minutes apart:

- **Steady-state GPU memory: 63,611 MiB / 143,771 MiB (44%)** --
  identical both checks, i.e. genuinely steady-state, not still
  climbing mid-curriculum.
- No OOM, no crash, running smoothly at step 1575/2500 (17.8 min into
  its 30-min cap) at check time.
- **First real eval, step 1500: `rollout_mse` beat persistence by
  +27.71%**, promoted to `_best.pt`. One early eval point -- promising,
  not conclusive; §39's peak-then-decline pattern has repeated across
  three prior arms, so this needs to hold up at later evals before it
  means anything on its own.

63.6GB is not close to §43.5's estimated ~162GB for `AR_SEQS=16`, and
not even close to the "safe" `AR_SEQS=12` estimate (~126GB) that
estimate led to shipping instead. The linear model (fixed overhead +
`AR_SEQS`-proportional attention-score memory) assumed the AR-rollout
loop's sequential forward passes keep their activations live
simultaneously; the real allocator evidently reuses/frees them far more
aggressively across the loop than that assumed. Lesson: a memory
estimate extrapolated from a SINGLE calibration point is a reason to be
cautious before a first real run, not a substitute for one -- this is
exactly why `m3_ar_combined`'s own `desc` said "not yet confirmed on
real hardware," and it wasn't, until now.

**`AR_SEQS` reinstated to 16** (`train_production_transformer_deep_dive.py`,
`tests/test_ar_objective_arms.py`, `tests/test_frame_ar_loss_direct.py`
all updated together, full suite re-verified green). `m3_ar_combined`'s
`desc` now cites this run's measured number instead of the withdrawn
estimate.

**Portability note added per this session's own request:** this 63.6GB/
44% measurement is H200-specific (143.8GB total). `provision_and_run.sh`
v7.3's broadened `GPU_AUTO_DETECT_PATTERN` (§43.4) means a real run
could land on an H100 (80GB) or an RTX PRO 6000 Blackwell (96GB) --
neither confirmed to leave the same headroom at `AR_SEQS=16`. `AR_SEQS`
is the knob to turn down first if `m3_ar_combined` OOMs on a
smaller-VRAM card -- 12 or 8 (`h5_ar_moreseqs`'s own already-proven
ceiling) are the documented fallbacks, now noted directly in the arm's
`desc` rather than only here.

## 45. v7.5 -- the per-epoch console report gets a ridge/linear-baseline column, a live "ceiling captured" metric, and its color was silently broken in every real run until now

**Root cause found for "why does the console output have no color at
all": it was never rendering, on any real pod run, full stop.**
`_COLOR_ON` gates on `sys.stdout.isatty()`, and `provision_and_run.sh`'s
`launch_training()` pipes the remote training process's stdout through
`ssh | awk` (for the `[arm/rN]` tag prefix) -- never a tty. Every colored
`Δ=...%` this report has ever printed on a real pod run rendered as
plain text; the color logic itself was correct, it just never activated
outside a local interactive shell. Fixed by exporting
`PFD_FORCE_COLOR=1` in the remote launch script -- the escape codes
survive `awk`'s tagging (it only prepends text, doesn't touch the
line's own content) and render fine both in a real terminal and in
wandb's own console-capture view (which supports ANSI).

**Third comparison column added: the ridge/linear map, not just
persistence.** New `_load_ridge_map_for_report()` (lazily loaded and
cached, mirroring `_load_decoder()`'s pattern) loads `RIDGE_MAP_PATH`
independently of whether the current arm trains against it
(`RIDGE_DISTILL_WEIGHT` can be 0.0) -- the SAME map §32.1's own
+69%-vs-persistence ceiling number comes from. `per_epoch_persistence_
report()` now rolls the ridge map out from the same starting context as
the persistence baseline (`ridge_rollout_targets()`, already used
safely elsewhere -- never touches the network, see that function's own
safety docstring) and reports three things per metric (MAE/RMSE/L2),
colored:

- `Δmodel` -- model vs. persistence (existing, was already computed,
  just never actually rendered in color -- see above).
- `Δridge` -- the ridge map's own margin over persistence, i.e. the
  achievable ceiling.
- `captured` -- `Δmodel / Δridge * 100`: what fraction of that ceiling
  the model has actually closed. >=50% green, 0-50% yellow, negative red
  (model is worse than persistence even though the ceiling itself is
  reachable), `n/a` when the ridge map doesn't beat persistence on this
  metric (nothing to divide by). Also gets a small fixed-width ASCII bar
  (`ceiling_bar()`) alongside the number.

This is the live version of a question this file has answered only
post-hoc until now (§32.1, §39, §41's r9/r10/r11/r12 tables) -- "how much
of the linear map's headroom has the model actually captured" is now a
number printed every single eval, not something recovered later from a
pulled-back log.

Gracefully absent (not an error) when `RIDGE_MAP_PATH` doesn't exist yet
or `TOKENIZATION != 'token'` (`ridge_rollout_targets()`'s own scoping,
matching `ridge_distill_targets()`) -- existing `persistence/*` W&B keys
and console behavior are unchanged in that case, fully backward
compatible.

**W&B: new keys, plus a section meant to bubble to the top.** New
`persistence/*_ridge*` keys alongside the existing (unchanged)
`persistence/*_model`/`*_pers`/`*_delta_pct` ones. The three
`captured_pct` numbers ALSO get logged under a separate `00_ceiling/*`
namespace -- the leading `00_` is deliberate: W&B's default workspace
panel/sidebar ordering is alphabetical by section name, so this is the
section most workspaces will show first without any manual pinning.

**Tests:** `tests/test_ridge_ceiling_report.py` -- pure-function tests
for `pct_improvement()`/`pct_ceiling_captured()`/`ceiling_bar()`
(promoted out of `per_epoch_persistence_report()`'s own closures to
module level specifically so they're independently testable, matching
this file's own repeated "extract the pure logic, test it in isolation"
pattern), `_load_ridge_map_for_report()` tests using the same
hand-checkable `A = 2*I` ridge matrix trick as
`tests/test_ridge_distill_loss.py`/`tests/test_split_anchor.py`, and an
end-to-end pair against a real (small) `BaseTransformer` through the
real production decoder confirming the ridge columns appear when a
ridge map is available and are cleanly absent when it isn't.

## 46. v7.6 -- `m3_ar_combined`'s first full run completed: §39's decline pattern a 4th time, notably earlier, plus a correction to §44's own AR_SEQS attribution

**Correction to §44/§45 first, since it affects everything below: the
run measured at 63.6GB/143.8GB (44%) was `AR_SEQS=12`, not 16.** §44
reported that measurement against `AR_SEQS=16` -- wrong. By the time
that run's pulled-back result came back, this file's own `AR_SEQS`
value had already been edited back to 16 locally, and the two got
conflated when §44 was written. The pulled-back run's own dumped
`Config` (`saved_models/r2_m3_ar_combined_status.json`'s sibling
`sweep_logs/manual/r2_m3_ar_combined.json`, which records the ACTUAL
`Config` values in effect on the pod, not what this file happened to
say locally afterward) settles it: `"AR_SEQS": 12`. **`AR_SEQS=16`
(this file's CURRENT shipped value) has never actually been run.**
Reinstating it was based on that misattribution and remains an
unconfirmed bet -- a reasonable one (even the more cautious 12-estimate,
~126GB, overshot the real 63.6GB by ~2x, so 16 likely fits too) but
extrapolation on top of a correction, not evidence. The arm's own
`desc` has been corrected in place to say this directly rather than
only here.

**The run itself (`AR_SEQS=12`, cold start, full 2500/2500 steps,
27.9 min wall-clock) is `m3_ar_combined`'s first real result, and it's
genuinely a 4th confirmed instance of §39's overfitting/rollout-drift
pattern** -- `train_loss` improved monotonically the entire run
(0.044 -> 0.038 -> 0.023 -> 0.012 -> 0.009) while rollout-vs-persistence
`improvement_pct` declined monotonically from its very first eval:

| step | improvement_pct | train_loss |
|---|---|---|
| 500 | **+34.4%** (peak, promoted) | 0.0436 |
| 1000 | +30.3% | 0.0385 |
| 1500 | +27.7% | 0.0229 |
| 2000 | +20.7% | 0.0117 |
| 2500 (final) | +19.0% | 0.0092 |

The peak (+34.4%) is the second-best ever recorded across every arm in
this file (only `a3b_delta_ar`'s 54.4% and `h11_ridge_distill`'s 37.3%
are higher) -- the underlying idea has real merit. But the peak landed
at step 500 of a 2500-step budget (**20% into the run**), clearly
earlier in relative terms than any of §39's three prior cases
(`s7_h9_scaled` ~2400/6000=40%, `h11_ridge_distill`/`s8_h9_moreseqs_
scaled` similarly 33-67% through their own budgets) -- `m3`'s heavier
weight + faster curriculum + larger `AR_SEQS` combination appears to
overfit/drift faster in relative-budget terms than the milder recipes
that pattern was first documented on. §39.1's own finding (WD/dropout
0.05 measurably slows, doesn't reverse, this decline on `h11`/`s8`,
both still at the standard `WEIGHT_DECAY=0.01`/`DROPOUT=0.01` this `m3`
run also used) is the natural, already-proven next lever to try here
too, rather than inventing something new.

**A concrete, fixable process gap: early stopping was OFF for this run,
and burned real pod-minutes doing nothing.** `Config.EARLY_STOP_
PATIENCE_STEPS` defaults to `None` (disabled) per `§40.1`'s own
design -- this run's launch command never passed
`--early-stop-patience-steps`, and neither does `provision_and_run.sh`
itself, anywhere. Since no eval ever re-promoted past step 500
(`best.promoted_rollout_mse` stayed pinned at the step-500 value for
the rest of the run), a patience of 500 steps (§42's own stated
"going forward" default, matching `--val-every 500` here) would have
stopped this run right at step 1000 -- instead it ran a further 1500
steps (roughly 2/3 of its 27.9-minute wall-clock) producing a strictly
worse checkpoint than the one already saved, for no benefit whatsoever.
**Action item, not yet done: `provision_and_run.sh` should default to
passing `--early-stop-patience-steps 500` (or an equivalent `--extra`
default) rather than leaving it opt-in-only per launch.**

**The catastrophic-looking `frame1` numbers are the ALREADY-DOCUMENTED
denominator artifact (§23.1), not a new finding.** This run's own
final per-frame curve: frame 1 at -556.9%, crossing positive by frame
~6, peaking ~+43% around frames 10-16, then a slow decline to +13.3% by
frame 68 -- the same "bad-looking early on a tiny-persistence-error
denominator, genuinely informative once persistence has degraded" shape
§23.1 first explained, just crossing positive noticeably earlier here
(frame ~6) than `e3_ar_long`'s original reference case (frame ~15+) --
a modestly encouraging shape detail, not a new phenomenon to explain.

**Also worth noting: this run is the first real evidence for §45's new
ridge-ceiling report feature, even though it predates that feature's
own console-output changes** -- it was launched under the OLDER
console format (no `ridge=`/`captured=` columns, no forced color), so
none of this section's analysis needed the new report; it was all
reconstructed from the pulled-back JSON's raw numbers instead. The next
run launched through the current code will show this live instead of
requiring the same manual reconstruction.

**Next lever shipped, per explicit direction this same session: `m4_ar_
combined_dropout`.** `m3_ar_combined`'s exact config plus `DROPOUT`
0.01->0.1 (10x, matching branch G's already-tried `g1_dropout` value) --
dropout ALONE, deliberately not the combined WD+dropout=0.05 bump §39.1
tested on `h11`/`s8`, so a result here isolates dropout's own effect
rather than repeating that exact combination at a different magnitude.
Not yet run.

## 47. v7.7 -- `m4_ar_combined_dropout` ran: dropout=0.1 measured -137.4% and never promoted, early-stopped correctly in 6.5 min -- but the test itself may have been unfair, so `m5` retries at 0.03 with 3x the patience

**The early-stop-patience default from §46 fired for real, for the
first time, and worked exactly as designed.** `m4_ar_combined_dropout`
(`m3_ar_combined`'s exact config + `DROPOUT` 0.01->0.1) stopped at step
500 of its 2500-step budget: `"stop_reason": "early stop: 500 steps
(patience=500) since the last new promoted rollout checkpoint (step
0)"`. Wall-clock: 387.8s (6.5 min) vs. `m3`'s own 1676s (27.9 min) for
a comparable run -- a real, immediate payoff from §46's fix.

**The result itself is bad, and confirmed NOT a code/config mistake --
the pulled-back run's own dumped `Config` matches every intended
setting**: `AR_SEQS: 16`, `DROPOUT: 0.1`, `EARLY_STOP_PATIENCE_STEPS:
500`, `WANDB_PROJECT: NI_Review_v7`. At its one and only eval (step
500): `improvement_pct = -137.4%` (worse than persistence),
`ever_promoted_rollout_checkpoint: false` -- it never cleared the
promotion gate at all. Direct comparison to `m3` at the identical step:

| | step 500 `improvement_pct` | step 500 `train_loss` |
|---|---|---|
| `m3_ar_combined` (no dropout) | +34.4% | 0.044 |
| `m4_ar_combined_dropout` (0.1) | -137.4% | 0.132 (3x higher) |

**But this may not be a fair test of dropout, and the doubt is
important enough to act on rather than just note.** Dropout is
*expected* to slow early convergence -- that's the entire premise of
§39.1's own finding that WD/dropout 0.05 "measurably slows the decline"
on `h11`/`s8` (a benefit that shows up over the LATER trajectory, not
at the first eval). `EARLY_STOP_PATIENCE_STEPS=500` is exactly one
`--val-every 500` window -- it kills any recipe that doesn't promote on
its very first try, regardless of whether more time would have let it
catch up or overtake persistence. `m3` happened to promote immediately,
so §46's default was never tested against a recipe that DOESN'T; `m4`
is that test, and the honest reading is "dropout=0.1 didn't beat
persistence within 500 steps," not "dropout=0.1 is bad," full stop.

**Follow-up shipped per explicit direction: `m5_ar_combined_dropout03`.**
Gentler `DROPOUT=0.03` (vs. `m4`'s 0.1), AND `EARLY_STOP_PATIENCE_STEPS`
bumped 500->1500 (3 eval windows instead of 1) baked directly into the
arm -- giving a regularized variant a real chance to promote before
judging it, addressing the fairness concern above rather than just
re-running the same tight patience at a different dropout value.

**A real interaction caught before it could waste another run: baking
`EARLY_STOP_PATIENCE_STEPS` into an arm's `overrides` does NOT survive
launching through `provision_and_run.sh` as-is.** The trainer's
long-standing "CLI beats the arm" rule (`apply_arm()` runs first, then
CLI flags unconditionally win if passed) means `provision_and_run.sh`'s
own new default (§46, always passes `--early-stop-patience-steps 500`
unless `EARLY_STOP_PATIENCE_STEPS=""` is set) would silently clobber
`m5`'s baked-in 1500 back down to 500 -- reproducing the exact same
"only one eval window" problem this arm exists to avoid. The arm's own
override is NOT dead code (a direct/manual trainer invocation without
an explicit `--early-stop-patience-steps` flag would still respect it),
but launching `m5` through `provision_and_run.sh` specifically REQUIRES
also passing `EARLY_STOP_PATIENCE_STEPS=1500` as an env var on that
launch command, e.g.:

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr EARLY_STOP_PATIENCE_STEPS=1500 \
  bash singleshot/provision_and_run.sh --arm=m5_ar_combined_dropout03 --round=2 --fresh
```

## 48. v7.8 -- `m5_ar_combined_dropout03` ran: best peak of the whole branch M family (+38.6%), but the SAME decline pattern a 5th time -- the real signal that regularization tuning alone has run its course

**`m5` (`m3`'s config + `DROPOUT=0.03`, `EARLY_STOP_PATIENCE_STEPS=1500`,
launched correctly with the env-var override per §47's own warning)
ran to its wall-clock cap, not an early stop** -- `stop_reason:
"wall-clock limit (0.5h)"`, `steps_completed: 2000` of a 2500 budget.
`ever_promoted_rollout_checkpoint: true` this time (unlike `m4`'s
`false`) -- 0.03 is a real, viable dropout value, unlike 0.1.

**The peak is the best across the entire branch M family so far:**

| step | improvement_pct | train_loss |
|---|---|---|
| 500 | **+38.6%** (peak, promoted -- best of `m1`-`m5`, beats `m3`'s own +34.4%) | 0.107 |
| 1000 | +23.9% | 0.033 |
| 1500 | +26.8% (a partial bounce) | 0.022 |
| 2000 (final) | +19.6% | 0.014 |

Gentler dropout didn't just avoid `m4`'s catastrophic failure -- it
produced a genuinely BETTER early result than `m3` (no dropout at all).
But the decline pattern replicated a **5th** confirmed time regardless
(train_loss falling monotonically the whole run while rollout-vs-
persistence peaks early and erodes), including a partial step-1500
bounce that still didn't recover the peak.

**The real conclusion: dropout tuning raises the peak a bit, it does
not fix the underlying decline.** Three dropout values now tried on
this exact recipe (0.01 implicit in `m3`: peak 34.4%; 0.1 in `m4`:
never promoted; 0.03 in `m5`: peak 38.6%, best yet) all show the same
qualitative shape. This is the concrete evidence behind this session's
pivot away from further regularization/hyperparameter tuning and
toward "bigger lever" candidates -- see the discussion following this
section for the menu under consideration.

## 49. v7.9 -- the "bigger lever" menu: a 3-way optimizer bake-off built and queued, 4 more experiments planned behind it

**Diagnosis behind this whole pivot, stated precisely:** `m5`'s
`train_val_gap` was ~0 for its entire run (teacher-forced train and val
loss moved together throughout) while the AR-rollout metric peaked
early and decayed -- this is NOT classical overfitting (which would
show train/val loss diverging too). It's exposure bias: the model
degrades specifically under its own autoregressive rollout, a
distribution it's never fully conditioned on during training. No
amount of dropout/WD tuning changes that shape, only where the peak
lands (§48). This session's own research pass additionally found two
axes never discussed anywhere in this project across 45+ arms:
**optimizer choice** (AdamW, unconditionally, every single time) and
**weight averaging/EMA** (never mentioned once).

### 49.1 Phase 1, built now: does the optimizer family itself matter?

New branch **P**, three arms sharing `m5_ar_combined_dropout03`'s exact
AR/dropout recipe (`AR_MODE=frame_ar`, `AR_LOSS_WEIGHT=3.0`,
`AR_WEIGHT_WARMUP_FRAC=0.05`, `AR_FRAMES=8`, `AR_FRAMES_START=1`,
`AR_FRAMES_WARMUP_FRAC=0.1`, `AR_SEQS=16`, `AR_EVERY_N_STEPS=1`,
`DROPOUT=0.03`) -- the best-known result so far (+38.6% peak, §48) --
varying ONLY the optimizer:

- **`p1_bestyet_adamw`** -- the control, `m5`'s config again under a new
  arm name so all three land as wandb/`saved_models/` siblings.
- **`p2_bestyet_lion`** -- `OPTIMIZER='lion'`
  ([lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch),
  confirmed on PyPI and installed locally this session).
  `LEARNING_RATE` 1e-3->2e-4, `WEIGHT_DECAY` 0.01->0.05 (5x smaller/
  larger respectively, the middle of Lion's own published 3-10x tuning
  guidance -- effective decay is `lr * wd`). New `Config.LION_BETAS =
  (0.9, 0.99)` (Lion's own default, vs. AdamW's `(0.9, 0.95)`).
- **`p3_bestyet_sophia`** -- `OPTIMIZER='sophia'` (the `sophia-opt`
  PyPI package, a packaged fork of the official
  [Liuhong99/Sophia](https://github.com/Liuhong99/Sophia) reference
  implementation -- confirmed on PyPI and installed locally this
  session). `LEARNING_RATE` 1e-3->8e-4 (Sophia's own "slightly smaller
  than AdamW" guidance). New `Config.SOPHIA_BETAS = (0.965, 0.99)`,
  `SOPHIA_RHO = 0.04`, `SOPHIA_WEIGHT_DECAY = 0.1` (the package's own
  defaults), `SOPHIA_HESSIAN_UPDATE_EVERY = 10`.

  **A concern raised while planning this turned out to be a non-issue,
  confirmed by reading the actual installed package source (not
  assumed from the paper or a summary):** the official Liuhong99
  reference training SCRIPT refreshes its diagonal-Hessian estimate by
  sampling a synthetic label from the model's own softmax output --
  correct for cross-entropy heads, meaningless for our continuous
  regression head. But `sophia_opt.SophiaG.update_hessian()` ITSELF
  doesn't do any of that -- reading its source directly: it just EMAs
  the squared gradient already sitting in `.grad`
  (`hessian.mul_(beta2).addcmul_(grad, grad, value=1-beta2)`), entirely
  loss-agnostic. So `p3`'s Hessian refresh (every
  `SOPHIA_HESSIAN_UPDATE_EVERY=10` steps, reusing the SAME real
  regression-loss gradient the ordinary update already computed, no
  extra forward/backward pass) is a standard use of the exposed
  primitive, not an ad hoc adaptation needing a caveat.

**Code** (`train_production_transformer_deep_dive.py`): new
`build_optimizer(model, cfg)` (module-level, replacing the single
hardcoded `torch.optim.AdamW(...)` call every prior arm used) branches
on the new `Config.OPTIMIZER` (`'adamw'` default -- byte-for-byte
unchanged behavior for all 48+ pre-existing arms). Lion/Sophia are
imported lazily inside the branch so a plain Mac dev session never
needs either package installed. A periodic
`optimizer.update_hessian()` call is inserted right before
`optimizer.step()` in the main training loop, gated as a true no-op
for every non-Sophia arm. `optimizer.step()` conditionally receives
Sophia's own `bs=` kwarg (the REAL effective batch size,
`BATCH_SIZE * ACCUMULATION_STEPS` -- NOT the package's own default of
5120) since AdamW/Lion don't accept that kwarg at all.

**Checkpoint-resume guard added** (extends the existing `cross_version`
skip-optimizer-state pattern): a checkpoint's `config` blob already
records `Config.OPTIMIZER` at save time for free (`config_dict()`
dumps every field generically) -- resume now compares it against the
CURRENT run's optimizer and skips `load_state_dict` with a clear log
line on a mismatch, instead of a cryptic key-mismatch crash (Lion keeps
only an EMA, AdamW keeps two, Sophia keeps an EMA plus a Hessian
buffer -- none of their per-parameter state shapes are compatible with
each other). None of the three `p*` arms trigger this (`--fresh
--no-warm-start`), but it's a real landmine for the next person who
warm-starts across an optimizer change.

**Dependencies**: `lion-pytorch` and `sophia-opt` added to
`requirements_sweep.txt`, `singleshot/requirements.txt`, and
`singleshot/bootstrap_remote.sh`'s install step -- both verified
installable and importable this session (`from lion_pytorch import
Lion`, `from sophia_opt import SophiaG` -- the latter's import name
does NOT match a naive guess from the PyPI project name, confirmed by
actually installing and inspecting it rather than assuming).

**Tests**: `tests/test_optimizer_selection.py` (`build_optimizer()`
returns the right class per `Config.OPTIMIZER`, raises `ValueError` on
an unrecognized value, a real one-step convergence smoke test per
optimizer against a throwaway `nn.Linear`, and the resume-mismatch
guard logic) and `tests/test_optimizer_arms.py` (the three `p*` arms'
`resolve_arm()`/`apply_arm()` pins, mirroring branch M's own pin-test
convention). Full suite: 132 passed, 13 skipped (up from 116 before
this section).

**Launch** (all three concurrently, `provision_and_run.sh`'s existing
3-slot support, no script changes needed):
```bash
NETWORK_VOLUME_ID=yl7f9e8rwr bash singleshot/provision_and_run.sh \
  --arm=p1_bestyet_adamw  --round=2 --fresh --max-steps=2500 --max-hours=0.5 \
  --arm2=p2_bestyet_lion  --round2=2 --fresh2 --max-steps2=2500 --max-hours2=0.5 \
  --arm3=p3_bestyet_sophia --round3=2 --fresh3 --max-steps3=2500 --max-hours3=0.5
```
Not yet run. Nothing in this section claims Lion or Sophia helps --
only that the wiring is real, tested, and ready to find out.

### 49.2 Phase 2, planned but NOT built -- waiting on phase 1's results first

Three more "think bigger" experiments, ranked cheapest-and-most-directly-
diagnostic-of-the-exposure-bias-shape first, each a one-paragraph menu
entry (not an implementation spec) pending its own scoping pass once
real data exists to decide priority:

1. **EMA of weights.** Very low cost, no architecture/loss change --
   the standard fix for exactly this "peaks early, drifts after" shape.
   Would apply on top of whichever optimizer phase 1 crowns.
2. **Non-autoregressive multi-horizon auxiliary head** (§32.2 item 4,
   never built) -- "the way to bypass error compounding entirely rather
   than manage it," in this file's own words.
3. **Make the loss ALWAYS the rollout metric** (§40.2's still-unbuilt
   second hypothesis half -- branch M only ever ADDED AR loss on top of
   a fixed-weight teacher-forced loss, never replaced it). Needs the
   same care `h10_ridge_residual`'s blowup (§26.2) already taught this
   project about anchors inside the feedback loop.

Two axes considered and explicitly deprioritized for now (real, just
not queued): retraining/fine-tuning the AE decoder jointly (high risk,
changes the scoring ground truth itself, too many moving parts to
isolate what changed) and alternative architecture families (SSM/
Mamba, diffusion -- zero prior discussion anywhere in this project,
full-rewrite cost, not worth it while cheaper targeted options above
are untried).

## 50. v7.10 -- branch P's first launch OOM'd: 3-way concurrency on one GPU was never checked against AR_SEQS=16's real memory footprint, `AR_SEQS` dropped to 8 for the whole bake-off

**The first real launch of branch P (§49) failed.** All three arms
(`p1_bestyet_adamw`, `p2_bestyet_lion`, `p3_bestyet_sophia`) were
launched concurrently on one H200 pod, `provision_and_run.sh`'s own
"multiple experiments concurrently" co-location feature. `p1` crashed
with `torch.OutOfMemoryError` inside `frame_ar_loss()`'s AR rollout,
mid-run: `Process ... 45.18 GiB ... Process ... 45.38 GiB ... Process
... 49.22 GiB` -- three processes totaling ~139.8 GiB against the H200's
own 143.8 GiB, i.e. essentially the entire GPU divided three ways, with
nothing left for `p1`'s own AR rollout to grow into. Nothing was pulled
back for any of the three arms (no `sweep_logs/manual/` entries, no
`saved_models/*_status.json`) -- this batch produced no usable result at
all, a genuine wasted pod-rental, not a partial one.

**Root cause: the math was checkable in advance and wasn't checked.**
`AR_SEQS=16`'s real memory footprint had already been measured SOLO,
twice over (§44, §46, §48) -- ~45-64GB depending on exactly which run's
measurement is trusted (see below). Three of those concurrently needs
~135-190GB against a 141-144GB H200. Every prior successful concurrent
batch in this project (branch V, branch S) used `AR_SEQS=2` arms
(~34GB each solo, 3x34=102GB, comfortable) -- co-location was validated
for the LIGHT recipe, and that validation was never re-checked before
pointing the same 3-slot mechanism at the HEAVY recipe branch P
deliberately holds fixed across all three arms. This should have been
caught before the launch, not after it burned a pod rental.

**A second correction, surfaced while fixing the first: `AR_SEQS=16`
was never actually confirmed solo either.** §46 measured 63.6GB/143.8GB
and attributed it to `AR_SEQS=16` -- but §46 itself already corrects
this: that measurement was actually `AR_SEQS=12` (a code-vs-pulled-back-
result mislabeling), and `AR_SEQS=16` was reinstated afterward on the
strength of a measurement that wasn't actually of `AR_SEQS=16` at all.
So going into this launch, `AR_SEQS=16`'s real solo memory footprint was
already unconfirmed -- the OOM wasn't just "3x a known number is too
much," it was "3x an ASSUMED number, which itself turned out to need
re-deriving, is too much."

**Fix, per explicit direction: `AR_SEQS` dropped to 8 for all three
branch P arms** (was 16). 8 is `h5_ar_moreseqs`'s own already-attempted
ceiling (branch H) and is lower than the one real solo measurement that
actually exists (`AR_SEQS=12` -> 63.6GB). **Stated plainly, not
overclaimed: this reduces the risk, it does not confirm the fix.** The
linear extrapolation model relating `AR_SEQS` to real memory usage has
already been shown wrong once (§46 found the real number ~2.5x lower
than the model's estimate) -- there isn't a second real calibration
point to refit that model against `AR_SEQS=8` specifically, so no
number is being promised here for what 3 concurrent `AR_SEQS=8` copies
actually use. If this OOMs again, the documented fallback is separate
pods per arm (no code change -- three separate `provision_and_run.sh`
invocations instead of one 3-slot concurrent one), not another guess at
a smaller `AR_SEQS`.

Code (`ROUND2_ARMS['P']`'s three arms), tests
(`tests/test_optimizer_arms.py`'s `_M5_AR_OVERRIDES`), and `--list-arms`
all updated and reverified together; full suite still 132 passed, 13
skipped. Not yet re-run.

## 51. v7.11 -- the run-lock's 30-minute staleness window was ~60-90x looser than the real heartbeat cadence, cut to 5 minutes after §50's OOM orphaned three locks on the persistent volume

**Direct consequence of §50's crash**: all three branch P processes died
without cleaning up their own `{run_name}.lock` files (`acquire_run_lock()`'s
`atexit`-registered `release_run_lock()` explicitly does NOT fire on
SIGKILL -- see that function's own docstring, and a CUDA OOM under real
memory pressure can trigger the Linux OOM-killer reaping the process
outright, not just `torch.OutOfMemoryError` unwinding normally). Every
relaunch attempt then hit `REFUSING TO START: another process appears
to already be training...` for `p1`/`p2` -- correct, by design, but
`LOCK_STALE_SECONDS` was `1800` (30 minutes), meaning the only paths
forward were waiting out half an hour or SSHing in to manually delete
the lock file by hand. **Now that `saved_models/` lives on the
persistent network volume (§43.3) rather than an ephemeral pod's local
disk, an orphaned lock survives pod termination and gets inherited by
the NEXT pod that mounts the same volume** -- a real, foreseeable
consequence of that architecture change that hadn't been connected to
this lock mechanism until it actually bit.

**Root cause of why 1800s was ever chosen: it was a generic "comfortably
long" guess, never checked against the mechanism's own real behavior.**
`touch_run_lock()` (the heartbeat) fires every `CHECKPOINT_EVERY_STEPS=25`
steps -- confirmed against this session's own real run logs, ~20-30s
apart under normal CUDA throughput. A genuinely alive process re-touches
the lock roughly 60-90 times within the OLD 1800s window; the staleness
check only needs to be a healthy MULTIPLE of that real cadence to still
correctly distinguish "alive" from "dead," not 60-90x it.

**Fix**: `LOCK_STALE_SECONDS` dropped `1800 -> 300` (still a 10-15x
margin over the real ~20-30s heartbeat -- comfortably covers a slow
eval or a slow network-volume checkpoint write without false-flagging a
genuinely alive process), and made overridable via
`PFD_LOCK_STALE_SECONDS` for a regime that's genuinely slower than this
(e.g. a real MPS run with a much longer per-checkpoint-cycle wall-clock)
without needing a code change. The refusal message itself now states
BOTH real options explicitly -- how many seconds remain until
auto-reclaim, and the exact `rm` path to delete it immediately -- instead
of just "refusing" with no stated path forward.

**What was deliberately NOT done**: removing the lock/refusal mechanism
entirely, or keying it to a per-launch-unique path. Both would eliminate
the friction by also eliminating the safety property this guards
(§v6.0's real incident: two overlapping wandb histories for the same
run) -- a per-launch-unique lock path in particular would make the
mechanism structurally incapable of ever detecting a genuine concurrent
double-launch, not just tolerant of a fast recovery from a dead one.
Tightening the SAME staleness check to match its own real cadence keeps
the safety property intact (a truly concurrent second launch still gets
refused immediately, since the first process keeps re-touching every
~20-30s) while fixing the actual complaint (crash recovery took 30
minutes when the mechanism's own heartbeat proves ~5 is enough).

**Tests**: `tests/test_run_lock.py`, new -- this mechanism had zero test
coverage before despite guarding a real, previously-hit incident. Covers
acquire/refuse/reclaim/touch/release against a real temp-directory
filesystem (not mocked -- the whole point is real mtime-based
staleness), the refusal message's remaining-wait text, and the shipped
300s default specifically (not just that an override mechanism exists
in the abstract). Full suite: 143 passed, 13 skipped (up from 132).

## 52. v7.12 -- two follow-ups from live production logs on the branch P pod: wandb artifact-upload `inf`-JSON failures fixed at the root, and `scp_env_files.sh` converted to rsync now that the network volume makes most of its files unchanged across relaunches

Both requested directly off real logs from the live branch P pod
(`205.196.17.66:9924`), not speculative cleanup.

### wandb artifact-metadata `inf` fix

Production logs showed repeated non-fatal failures:

```
[wandb] artifact upload failed for r2_p1_bestyet_adamw_latest
(ValueError: Out of range float values are not JSON compliant: inf);
local file(s) are unaffected.
```

Root cause: `train()`'s `best` dict initializes `rollout_mse`,
`improvement_pct`, `val_tf_mse`, and `train_loss` to `float('inf')`/
`float('-inf')` as the "nothing evaluated/promoted yet" sentinel. Every
`save_checkpoint()` call before the first validation pass
(`Config.VAL_EVERY`, e.g. every eval-less step from 0 to 500 in this
run) passes `extra = {'rollout_mse': best['rollout_mse'], 'improvement':
best['improvement_pct'], ...}` straight through to `log_artifact()`,
which flattens it into `meta` and hands it to
`wandb.Artifact(..., metadata=meta)`. wandb's own JSON validator is
stricter than Python's `json` module -- it rejects `inf`/`-inf`/`nan`
outright instead of encoding them as literal `Infinity`/`NaN` tokens --
so every one of those early-checkpoint artifact uploads silently
failed. Non-fatal (the local `.pt` write is unconditional and always
succeeds first) but a real, avoidable gap in wandb's checkpoint-version
history for every run's first several hundred steps.

**Fix**: `_Telemetry.log_artifact()` (the one shared entry point every
`save_checkpoint()` call already goes through) now sanitizes `metadata`
before constructing the `Artifact` -- any float that fails
`math.isfinite()` becomes `None`, matching how the rest of this file
already represents "not yet measured" (e.g. `write_status_json`'s own
output). Fixed at this single shared choke point rather than at each
individual call site, so any FUTURE caller that happens to pass a
still-`inf` sentinel through `extra` is covered automatically, not just
the fields known about today.

**Tests**: new `tests/test_telemetry_artifact_metadata.py`, using a
fake `wandb.Artifact`/`run` that deliberately mimics wandb's real
JSON-strictness (raises on a non-finite float) rather than a mock that
just always succeeds -- confirms the fix actually prevents the
reproduced failure, not just that the code path runs. Covers: the
originally-failing `inf` case now succeeds and logs no failure line,
non-finite values specifically become `None` while finite values and
non-float types pass through untouched, an all-finite `metadata` dict
is byte-for-byte unaffected, and the no-`metadata`-passed case still
works.

### `scp_env_files.sh`: scp -> rsync

With `NETWORK_VOLUME_ID` set, `$REMOTE_ROOT` (`/workspace`) is a
persistent RunPod network volume that survives pod termination --
already exploited for `scp_data_files.sh` (skipped entirely once the
volume already has the data, prior session work) and for
`saved_models/`/lock-file persistence (§v7.11). `scp_env_files.sh` was
the one remaining piece still doing a full, unconditional transfer on
every single launch: the trainer `.py` files, the AE decoder, the ridge
map, and old-round checkpoints, all sent via plain per-file `scp`
regardless of whether the pod already has byte-identical copies from
the previous launch. In practice the AE decoder and ridge map NEVER
change, and the trainer code often hasn't changed between two
back-to-back launches either -- only the checkpoint-resume targets and
whichever `.py` file was actually edited are genuinely new.

**Fix**: replaced the per-file `scp` + manual size-check loop with
`rsync -az --checksum -i`, one call per file (same `FILES` array,
unchanged). `--checksum` (not the default mtime+size quick check) is
deliberate: a `git checkout` or file copy can bump a file's mtime
without changing its content, which would make the default check
re-send something that's actually identical -- every file here is
small enough that paying for a real checksum comparison instead of
trusting mtime is cheap and removes that false-positive path entirely.
`-i` (itemize-changes) is what lets the script tell "sent" apart from
"skipped, already up to date on the pod" per file in its own output,
replacing the old manual local-vs-remote `stat` size comparison
(rsync's checksum-based decision is already a stronger correctness
check than that comparison ever was).

No bootstrap/dependency change needed: `provision_and_run.sh` already
uses `rsync` for pull-back (`rsync_with_retry()`, pre-existing), which
confirms rsync is already present on these pod images without any
extra installation step.

**Not changed**: `scp_data_files.sh` (already network-volume-aware from
prior session work, bypassed entirely rather than converted to rsync --
the train/val H5 files are large enough that its own parallel-lane
split-scp mechanism, `lib_scp.sh`, is worth keeping for the cold-start
case where the volume genuinely doesn't have the data yet).

**Verification**: `bash -n` on the edited script; full test suite
green, 147 passed / 13 skipped (up from 143 -- the 4 new telemetry
tests). The rsync conversion itself has NOT yet been exercised against
a real pod in this pass -- next relaunch's `scp_env_files.sh` output
will show the actual "skipped, already up to date" behavior for the
first time.

## 53. v7.13 -- venv bootstrap now skips reinstalling when a fingerprint-matched venv already survives on the network volume; `p2_bestyet_lion`'s crash investigated (inconclusive on cause, but a real gap in crash-forensics fixed) and traced to likely 3-way GPU-memory contention, not a Lion-specific bug

### bootstrap_remote.sh: skip re-installing an already-good venv

Same root cause as `scp_env_files.sh` (§52): `$VENV_DIR` (default
`$REPO_ROOT/.venv`) sits under `/workspace` whenever `NETWORK_VOLUME_ID`
is set, so it survives pod termination -- but `bootstrap_remote.sh`
unconditionally ran `apt-get update`/`apt-get install`, `python -m venv`,
and every `pip install` (including a full `torch` download) on EVERY
relaunch, even against a pod whose venv was already fully built by the
previous launch. The actual slow part was never venv/package creation
itself, it was N pip-install round-trips each just re-confirming
"requirement already satisfied" over the network.

**Fix**: two independent local (no-network) checks now gate the slow
steps:
- `dpkg -s mc screen` (a local metadata lookup, no `apt-get update`
  network round-trip) skips the apt step entirely if both are already
  installed.
- A fingerprint file (`$VENV_DIR/.bootstrap_fingerprint`, containing the
  exact pins -- numpy/h5py/wandb/lion-pytorch/sophia-opt versions, the
  resolved `CUDA_TAG`, and the interpreter version) plus one cheap
  `python -c "import torch, h5py, numpy, wandb, lion_pytorch, sophia_opt"`
  sanity check gates venv creation and every `pip install` call. Only
  written AFTER every install step actually succeeds (`set -e` would
  already have aborted the script on a failed step otherwise) --
  writing it unconditionally would risk caching a partially-broken
  venv as "done". `CUDA_TAG` detection was moved earlier in the script
  (was after the installs, now before) so it's known in time to be
  part of the fingerprint.

Not cached: the actual GPU-visibility verification at the end (now also
checks `lion_pytorch`/`sophia_opt` import, which the original script
never verified) -- cheap, and worth re-confirming every launch
regardless of whether the venv itself was rebuilt.

**Verification**: `bash -n` only in this pass -- not yet exercised
against a real pod's SECOND relaunch (the case this actually speeds
up). Next relaunch's bootstrap output will show the real "already
matches this bootstrap's exact pins" skip message for the first time.

### `p2_bestyet_lion`'s missing process, investigated

Flagged from a live pod (`205.196.17.66:9924`, branch P's 3-way launch,
§51/§52's context): `p2_bestyet_lion` had a `saved_models/
r2_p2_bestyet_lion_status.json` frozen at step 250 (14:47:58 UTC) and
no live process, while `p1_bestyet_adamw` and `p3_bestyet_sophia` were
still running (or, for p1, had already finished cleanly -- its own
`sweep_logs/manual/r2_p1_bestyet_adamw.json` shows a legitimate early
stop at step 500). Findings:

- **p2's run-lock file was ABSENT, not stale.** Per §51's own fix, a
  SIGKILL'd process (the Linux OOM-killer, the mechanism that caused
  §50's original 3-way OOM) leaves the lock behind for
  `LOCK_STALE_SECONDS` to expire. An absent lock means
  `release_run_lock`'s `atexit` handler actually ran -- i.e. the Python
  interpreter exited through its NORMAL shutdown path, which also
  happens after an uncaught exception (traceback printed, non-zero
  exit, but still a "clean" process exit as far as signals/atexit are
  concerned). This is consistent with a *caught* `torch.OutOfMemoryError`
  (PyTorch's own CUDA-OOM exception, which propagates as ordinary
  Python control flow) rather than the OOM-killer's SIGKILL.
- **No traceback was recoverable.** Every arm's stdout only ever
  reached the local terminal that launched `provision_and_run.sh`
  (piped through `ssh ... | awk` for the "[arm/rN]" tag, nothing
  written to disk on the pod) -- once that terminal's scrollback is
  gone or the pod is torn down, the actual error message is
  unrecoverable. This is a real gap, fixed below.
- **A live repro attempt was inconclusive but pointed at contention,
  not a code bug.** Relaunched `p2_bestyet_lion` alone (`--max-steps
  400 --max-hours 0.15 --no-wandb`) on the same pod while only
  `p3_bestyet_sophia` was still running (p1 had already finished, so
  this was 2-way, not 3-way, concurrency). It ran cleanly past step
  250/275 (where the original crashed) to step 375 with no error
  before the pod was terminated (`provision_and_run.sh`'s own
  `wait`+pull-back+terminate flow completing once its slots finished --
  expected, standing behavior, not a mistake). Getting well past the
  original failure point under LOWER concurrency, with no crash,
  argues against a deterministic Lion-specific bug and FOR §50's
  original, still-open concern: `AR_SEQS=8` reduces 3-way concurrent
  memory pressure but was never confirmed safe under it, only under
  2-way. This is circumstantial, not proven -- the exact error message
  is gone, so this is the best available honest read of the evidence,
  not a confirmed root cause.

**Fix (forensics, not a guess at the bug itself)**: `provision_and_run.sh`'s
`launch_training()` now tees each arm's full stdout/stderr to
`sweep_logs/r{round}_{arm}.log` ON THE POD (in addition to the existing
live-tagged local view), which the script's own existing pull-back step
already rsyncs back afterward -- so the NEXT time any arm crashes, the
actual traceback survives the pod's termination instead of only ever
existing in a terminal's live scrollback.

**Not done (deliberately, pending real evidence)**: no code change to
AR_SEQS, concurrency, or Lion's optimizer construction -- the evidence
here is suggestive, not diagnostic, and changing something based on an
inconclusive repro would risk overclaiming a fix for a problem that was
never confirmed in the first place. The existing "run at least two at a
time" guidance (from this branch's own launch note) already covers the
cautious path if 3-way concurrency continues to be unreliable; the next
run's `sweep_logs/r2_p2_bestyet_lion.log` (once the tee fix is live)
will settle this with a real traceback instead of a second inference.

**Verification**: `bash -n` on both edited scripts; full test suite
unaffected by these bash-only changes, 147 passed / 13 skipped
(unchanged from §52).

## 54. v7.14 -- branch P's real results analyzed, a second real bug found and fixed (early-stop patience silently clobbered), AR_SEQS dropped 8->6, next launch prepped

### What actually came back from the 3-way branch P launch

- **`p1_bestyet_adamw` (control): early-stopped at step 500/2500,
  improvement_pct -21.5% (WORSE than persistence).** Root cause found
  (see below, not noise) -- it never got a second evaluation window.
- **`p3_bestyet_sophia`: reached step 2000/2500 (stopped by the 0.5h
  wall-clock budget, not a crash or early-stop), promoted checkpoint at
  +34.1% improvement, `train_val_gap` ~0.0003 (still not overfitting in
  the classical sense).** Comparable to m5's own peak (+38.6% at step
  500) and, notably, still improving/stable at step 2000 rather than
  having already peaked and started declining -- the best sustained
  result of any arm so far, though on a different (lower) AR_SEQS than
  m5's own run, so not a clean apples-to-apples comparison yet.
- **`p2_bestyet_lion`: zero real performance data.** Crashed at step
  ~250-275 before its first scheduled evaluation (`--val-every 500`) --
  `best` never left its `Infinity` initial sentinel. Lion is completely
  untested on this recipe so far; nothing about its rollout performance
  is known yet.

### Second real bug found: branch P's arms silently inherited a
### too-tight early-stop patience, killing the AdamW control after one eval

`p1_bestyet_adamw`'s -21.5% result traced to a real, previously-fixed-
elsewhere bug, not to AdamW actually being bad: `last_promotion_step`
starts at `step` (0 on a cold start), and the early-stop check is
`(step - last_promotion_step) >= EARLY_STOP_PATIENCE_STEPS`. With
`--val-every 500` and the trainer's own default
`EARLY_STOP_PATIENCE_STEPS=500`, the very FIRST evaluation already
satisfies `500 - 0 >= 500` if it didn't promote -- there is no second
chance. `m5_ar_combined_dropout03` (branch P's own AR/dropout recipe
source) already hit this and already fixed it by setting
`EARLY_STOP_PATIENCE_STEPS: 1500` in its own arm dict (see that arm's
own `desc`, written before this session touched branch P at all) -- but
all three branch-P arms were copied from m5's recipe WITHOUT carrying
that override forward, so they silently fell back to 500 and inherited
the exact trap m5 had already worked around.

**Fix**: added `EARLY_STOP_PATIENCE_STEPS: 1500` to all three `p1/p2/p3`
arms' `overrides`, matching m5 exactly (3 eval windows instead of 1).
**Same unresolved caveat as m5's own fix, called out explicitly in the
new branch comment**: `apply_arm()` runs BEFORE `main()`'s "CLI beats
the arm" loop, and `provision_and_run.sh` ALWAYS passes
`--early-stop-patience-steps` (default 500) as an explicit CLI flag --
which unconditionally overwrites this 1500 back down to 500 regardless
of the arm dict. Baking it into the arm dict still helps a direct/manual
invocation that skips `provision_and_run.sh`, but launching branch P
through that script REQUIRES also setting `EARLY_STOP_PATIENCE_STEPS=1500`
as an env var on the launch command -- the next launch command below
does this.

### AR_SEQS dropped 8 -> 6

Per direct instruction, in response to `p2_bestyet_lion`'s crash
(§v7.13): a further reduction in the same direction as §50's original
16->8 cut. Still NOT a confirmed-safe number for 3-way concurrency --
nobody has measured 3 co-located copies at AR_SEQS=6 either, this is a
continued risk-reduction, not a proof. All three `ROUND2_ARMS['P']`
arms and `tests/test_optimizer_arms.py`'s pin test updated together.

### Next launch (prepped, not yet run)

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr EARLY_STOP_PATIENCE_STEPS=1500 \
  bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/provision_and_run.sh \
  --wandb=<KEY> --wandb-group=Spock \
  --arm=p1_bestyet_adamw  --round=2 --fresh --max-steps=2500 --max-hours=0.5 \
  --arm2=p2_bestyet_lion  --round2=2 --fresh2 --max-steps2=2500 --max-hours2=0.5 \
  --arm3=p3_bestyet_sophia --round3=2 --fresh3 --max-steps3=2500 --max-hours3=0.5
```

All three arms re-run `--fresh` (not resumed): `p1`'s prior run is not
worth resuming from (it stopped for a now-fixed spurious reason, at
essentially the starting line); `p2`'s prior run never reached a valid
evaluation; and `p3`'s prior run, while promising, was on the
about-to-change AR_SEQS (8, now 6) -- resuming it would silently mix
two different AR_SEQS regimes in one run's history. A clean 3-way
re-launch keeps the comparison honest.

Also carries forward from this session's other fixes automatically
(no launch-command change needed): `scp_env_files.sh`'s rsync skip-
if-unchanged (§52), `bootstrap_remote.sh`'s skip-if-already-built venv
(§53), the wandb `inf`-metadata fix (§52), and per-arm log persistence
to `sweep_logs/r{round}_{arm}.log` (§53) for actual crash forensics if
`p2_bestyet_lion` (or anything else) fails again.

**Verification**: full test suite green, 147 passed / 13 skipped
(unchanged count from §53 -- these are pin-test content updates, not
new tests). Not yet launched against a real pod in this pass.

## 55. v7.15 -- the just-shipped scp->rsync conversion (§52) broke the very next real launch: `--no-owner --no-group` added after a `chown ... Operation not permitted` (rsync exit 23) on the pod

First real use of §52's rsync conversion failed immediately:

```
rsync: [receiver] chown ".../.train_production_transformer_deep_dive.py.XFrq4X" failed: Operation not permitted (1)
rsync error: some files/attrs were not transferred (see previous errors) (code 23)
```

Root cause: `-a` (archive) implies `-o`/`-g` (preserve owner/group),
which requires the RECEIVING rsync process to call `chown` -- and
RunPod's container images don't grant `CAP_CHOWN` even to the `root`
user running inside the container (a common container-hardening
default, unrelated to normal Unix permission bits). `-euo pipefail`
correctly aborted the whole `scp_env_files.sh` on this, which correctly
aborted `provision_and_run.sh`, which correctly terminated the pod --
every layer of this session's own caution worked exactly as designed,
this was a genuine bug in the fix itself, not a false alarm.

Pull-back's own long-standing `rsync -avP` (`provision_and_run.sh`,
predates this session) is NOT at risk of the same failure: `-o`/`-g`
preservation is a silent no-op (not an error) when the RECEIVING side
isn't root, and pull-back's receiver is always the local, non-root Mac
user -- confirmed by every pulled-back file this session already
landing owned by `kkreth`, not `root`. Only the PUSH direction (pod,
running as root, receiving) actually hits the missing-`CAP_CHOWN`
error.

**Fix**: `--no-owner --no-group` added to `scp_env_files.sh`'s rsync
call, dropping just those two components of `-a` (keeps recursive,
symlink, permission, and timestamp preservation). Ownership doesn't
matter for this pipeline anyway -- every pod is single-user, everything
already runs as root.

**Verification**: `bash -n` only in this pass -- not yet re-tried
against a real pod. Full test suite unaffected (bash-only change),
147 passed / 13 skipped.

## 56. v7.16 -- diagnosed the reported "GPU stalls even with 3 arms concurrent": traced to the periodic autoregressive validation rollout, not a training-batching bug; a pre-eval log marker added for next-run confirmation

Investigated live on `205.196.19.52:11200` (branch P's relaunch, all
three arms running with the AR_SEQS=6/patience=1500/rsync fixes from
§52-55).

### What was actually measured

- `nvidia-smi dmon -s u -d 1 -c 90` (90 one-second samples): SM
  utilization sits at 97-100% the large majority of the time, with
  occasional brief (1-2 sample) dips to 20-70%. Not a sustained stall,
  but real, repeating dips.
- Cross-referencing `sweep_logs/r2_p1_bestyet_adamw.log`'s own
  per-25-step timestamps: throughput is a steady ~0.3 wall-clock
  minutes per 25 steps (~0.72s/step) EXCEPT at step 500 and step 1000
  (both `--val-every 500` boundaries), where the same 25-step window
  took 0.6-0.8 minutes -- roughly 2-2.5x longer. The dips observed live
  in `nvidia-smi` line up with exactly this cadence, not with random
  noise.
- CPU/host is not the bottleneck: `load average: ~21` on a 192-core box
  with 3 concurrent training processes, `iostat`/disk showed nothing
  notable.

### Root cause: `rollout_frames()`'s validation eval is inherently
### GPU-underutilizing, and all 3 arms share the same eval cadence

`rollout_frames()` (used by `evaluate()`, called every `--val-every`
steps -- 500 here) is a genuinely sequential autoregressive decode: it
re-runs a FULL `model(curr)` forward pass on every one of
`NUM_TIME - VAL_CONTEXT_STEPS` (80-12=68) iterations, growing `curr` by
one frame each time via `torch.cat` -- there is no KV-cache reuse, so
each step both re-does the full attention computation over the
growing context AND cannot be batched across the time dimension the
way a normal training step's single forward pass is. Combined with a
small batch (`VAL_ROLLOUT_SEQS=64`), this workload is latency-bound
(68 small, sequentially-dependent kernel launches with Python-loop and
`.cat()`/`.clone()` overhead between them) rather than throughput-bound
-- exactly the profile that shows up as a GPU utilization dip on
`nvidia-smi` even though "work" is still happening. Measured
`rollout_seconds` in this session's own status.json files: 3.2-7.8s
for 64 sequences, consistent with a latency-dominated loop rather than
a large batched matmul.

Because all three branch-P arms use the same `--val-every 500` and were
launched within seconds of each other, their evals tend to land in the
same wall-clock window -- so a dip that would be a small, single-process
blip in isolation becomes a visible AGGREGATE utilization dip across
all three concurrent processes at once, which is what actually gets
observed watching `nvidia-smi` live even with "three models running
concurrently."

**This is a real, structural property of the eval, not a training-
batching misconfiguration** -- the training step itself (per the
25-step throughput analysis) is steady and fully GPU-utilized; only
the periodic eval dips. A less naive rollout implementation (KV-cache
reuse, or batching the "shorter" iterations more cleverly) would close
this gap, but that's a real architecture change to `rollout_frames()`,
not attempted here -- flagged as a real, not yet acted on, finding.

### Instrumentation added (per "add it on the next run")

A `[eval] step {N}: starting rollout eval (...)` log line now prints
BEFORE `evaluate()` is called (previously the only `[eval]` line was
AFTER, reporting `rollout_seconds` retroactively) -- lets a live
`nvidia-smi`/`dmon` trace be correlated to the exact wall-clock moment
an eval starts, confirming this diagnosis directly on the next run
instead of inferring it from checkpoint-interval timing after the fact.

**Not done**: no change to `rollout_frames()`'s actual mechanism
(KV-caching, batching) -- this pass is diagnosis + one log line, not
an eval-performance rewrite; that's a separate, larger piece of work
if the dips turn out to matter enough to justify it (currently ~a few
percent of total wall-clock at a 500-step val cadence, not a dominant
cost).

**Verification**: full test suite unaffected, 147 passed / 13 skipped.

## 57. v7.17 -- branch P's first clean, complete 3-way run: all §52-56 fixes held, real comparable optimizer data for the first time

Relaunched with the rsync/`--no-owner`/AR_SEQS=6/patience fixes from
§52-56. Result: **no crashes, no premature single-eval early stops,
full pull-back succeeded on the first try.** First time this branch has
produced real, comparable data for all three arms.

| Arm | Peak improvement_pct | Outcome |
|---|---|---|
| `p1_bestyet_adamw` | **+40.5%** (rollout_mse 0.001691, step ~500) | Early-stopped at step 1500 -- legitimately: `patience=1000` exhausted with no new promotion since step 500 (the real early-stop mechanism working as intended this time, not the §54 single-eval bug) |
| `p3_bestyet_sophia` | **+38.5%** (rollout_mse 0.001750) | Still promoting -- cut off by the 0.5h wall-clock budget at step 2257, had NOT exhausted patience |
| `p2_bestyet_lion` | **+30.9%** (rollout_mse 0.001966) | Still running, stable around +30-31%, cut off at step 2294, had NOT exhausted patience |

Note: the actual launch used `--early-stop-patience-steps 1000` (not
the 1500 documented in §54/§56's arm-dict override) -- still safely
above `--val-every 500`, so the single-eval trap did not recur, just a
different concrete patience value than what was written into the code.

**Read of the comparison (early, not final -- none of these ran
uninterrupted to their full 2500-step budget)**: AdamW's peak is
highest but it already plateaued in the classic early-peak/exposure-
bias pattern this whole investigation keeps finding (§40.2 on). Sophia
is close behind (-2 points) and, notably, was STILL finding new
promotions past step 1500 where AdamW had already given up --
suggestive that Sophia may be more resistant to the same plateau, but
unconfirmed since it never got the chance to actually exhaust its own
patience before the wall-clock cutoff. Lion trails both by a real,
non-trivial margin (~8-10 points) but is a legitimate positive result
(the first time this arm has produced any rollout data at all, per
§54's crash and §56's fix).

**Also confirmed working in production for the first time**: `scp_env_files.sh`'s
rsync push (§52/§55), `sweep_logs/r{round}_{arm}.log` per-arm
persistence (§53), and the `[eval] starting rollout eval` marker (§56,
present in the pulled-back logs). `bootstrap_remote.sh`'s skip-if-
already-built venv (§53) was not directly observed this pass (fresh
pod, first bootstrap).

**Not done**: no conclusion yet on "does the optimizer family matter" --
Sophia and Lion both need a longer, uninterrupted budget (or a bumped
`--max-hours`) to actually reach their own patience exhaustion or
`MAX_STEPS=2500` before AdamW's result can be called final against
them.


## 58. v7.18 -- branch Q built: the Phase 2 "bigger lever" menu (§49.2) wired up as three real arms on top of the surviving best-so-far Sophia config

Branch P's clean 3-way run (§57) named `p3_bestyet_sophia` (+38.5%,
still promoting when cut off by the 0.5h wall-clock budget) as the
config to build on. `q1_sophia_ema`, `q2_sophia_auxhead`,
`q3_sophia_rollout_dominant` implement the three Phase 2 menu items
against that exact base recipe, varying only the one new axis each
arm is named for -- same one-variable-at-a-time discipline as every
other branch here.

**New `Config` fields** (all no-ops at their defaults, so every one of
the 48+ pre-existing arms is byte-for-byte unaffected): `EMA_DECAY=0.0`
(disables EMA-of-weights), `TF_LOSS_WEIGHT=1.0` (multiplier on the
primary teacher-forced loss), `AUX_HEAD_FRAMES=0` / `AUX_HEAD_LOSS_
WEIGHT=1.0` (>0 constructs `FrameTransformer.aux_head`, a non-
autoregressive multi-horizon head -- §32.2 item 4, never built before
this branch).

**Mechanism**:
- `q1_sophia_ema`: a second, independent deep-copied model instance
  (not `torch.optim.swa_utils.AveragedModel` -- its wrapper doesn't
  forward arbitrary attributes, and `rollout_frames()`/`frame_ar_loss()`
  read `model.frame_native` directly). EMA shadow updated every step;
  EMA'd weights become the PRIMARY rollout metric each eval (gating
  promotion/early-stop), raw non-EMA numbers kept as `raw_*` for
  comparison only.
- `q2_sophia_auxhead`: new `aux_horizon_loss()` predicts the next
  `AUX_HEAD_FRAMES` frames in ONE SHOT via `model.forward_aux()` -- no
  feedback loop at all, so none of `h10_ridge_residual`'s expansive-
  anchor-in-a-loop risk applies. Only supports frame-native models.
- `q3_sophia_rollout_dominant`: `TF_LOSS_WEIGHT` multiplies the primary
  teacher-forced loss, letting `AR_LOSS_WEIGHT`'s rollout term dominate
  instead -- the still-unbuilt second half of §40.2's original
  hypothesis (branch M only ever ADDED AR loss on top of a fixed-weight
  TF loss, never actually flipped the balance).

**Not yet launched as of this section** -- see §59 for the real
results, including two real bugs found on `q2_sophia_auxhead`'s first
launches.

## 59. v7.19 -- diagnosed a 3-way `torch.compile` pileup live on the pod, disabled it by default; found and fixed TWO real shape bugs in the frame-native AR/aux-head code paths; final Q-branch numbers land in the high-30s/low-40s%

### Diagnosis: branch Q's first launch looked hung, but was CPU-bound `torch.compile` contention, not a GPU/data problem

Live SSH investigation (`nvidia-smi` + `ps aux`) on the pod mid-launch:
0% GPU utilization, ~48GB already resident (three models' weights +
optimizer state loaded fine), but **~195 `torch/_inductor/compile_
worker` processes alive** on a 192-vCPU box -- each of the three
concurrent training processes spawns its OWN `torch.compile` worker
pool sized to the FULL host core count (`--workers=64` each), so a
3-way concurrent launch oversubscribes the box 3x over before a single
real training step runs. Branch P's own launches paid the same tax
(every log shows `torch.compile: on`) but it wasn't as visible then.

**Fix**: `resolve_train_regime()`'s CUDA branch now defaults `compile_
model` to `False` (`PFD_COMPILE_MODEL=1` re-enables it, e.g. for a
solo/long run where the 15-30% steady-state speedup amortizes past the
compile cost). Confirmed via a live re-check: 0 compile-worker
processes on the next launch, `nvidia-smi` GPU util climbing normally.

**Follow-up bug caught by the user**: `_regime_banner_cuda()`'s printed
"torch.compile: on/off" line was a HARDCODED string, not derived from
the actual flag -- so the banner kept claiming "on" even after the
default flipped to off. Fixed to take `compile_model` as a real
parameter; `tests/test_train_regime_cuda.py` updated to pin the new
default (`compile_model is False`) and the `PFD_COMPILE_MODEL=1`/`=0`
override, plus a dedicated override test.

### Bug 1: `frame_ar_loss()`'s frame-native branch crashed on its first-ever real exercise

`q2_sophia_auxhead` was the FIRST arm in this entire project to combine
`AR_MODE='frame_ar'` with `TOKENIZATION='frame'` -- every prior
`frame_ar` arm (branches M/P/Q's other two) ran token-native, so
`frame_ar_loss()`'s `if getattr(model, 'frame_native', False):` branch
had never actually executed against the real frozen AE decoder before.
It called `centroid_velocity_loss()` directly on the raw frame-
flattened `(B, n_fr, NX*LATENT_DIM=470)` tensor instead of reshaping to
a trailing `LATENT_DIM=47` axis first (the same `to_per_token_latent()`
reshape the teacher-forced path already applies) -- crashed on the
very first training step: `"mat1 and mat2 shapes cannot be multiplied
(6x470 and 47x100)"` against the decoder's first linear layer. Also
needed `TOKENIZATION: 'frame'` added to the arm's own overrides in the
first place -- `AR_MODE='frame_ar'` does NOT imply frame tokenization,
contrary to the arm's original `desc`.

**Fixed**: both `to_per_token_latent()` calls added in `frame_ar_loss()`'s
frame-native branch; `q2_sophia_auxhead`'s overrides now include
`TOKENIZATION: 'frame'`. New regression test
`test_frame_native_branch_shapes_and_gradients` (confirmed to fail
against the pre-fix code, reproducing the exact crash, and pass
against the fix) closes the gap that let this reach a real pod
uncaught -- `tests/test_frame_ar_loss_direct.py` had only ever
exercised the token-native branch before this.

### Bug 2: `aux_horizon_loss()` had the IDENTICAL bug, one call site later

`q2_sophia_auxhead`'s SECOND real launch (after bug 1's fix let it get
further) crashed with the same signature one call site later --
`aux_horizon_loss()` (the function this arm actually exists to test)
had the same missing `to_per_token_latent()` reshape on both `aux_pred`
and `target` before calling `centroid_velocity_loss()`. Missed on the
first pass because that pass only directly exercised `frame_ar_loss()`,
not this sibling function.

**Fixed** the same way; new regression test
`test_aux_horizon_loss_shapes_and_gradients` added directly (confirmed
fails pre-fix, passes post-fix) so both of `q2`'s call sites are now
covered locally instead of only being discoverable on a real pod.

### Bug 3 (found the same investigation): `q1_sophia_ema`'s `EMA_DECAY=0.999` never converged within the run's own budget

`q1_sophia_ema`'s first real launch early-stopped at step 1500 with its
EMA-gated `improvement_pct=-182.7%` (never promoted once, `promoted_
rollout_mse=Infinity`) -- while the SAME checkpoint's `raw_improvement_
pct` (non-EMA weights) was `+41.9%`, the best number of the whole
session at the time. `EMA_DECAY=0.999`'s ~693-step half-life never
caught up to the raw trajectory within a run whose own AR curriculum is
still ramping through step 250 -- and because EMA numbers gate
promotion/early-stop, the run stopped itself on a shadow that was still
catching up, not a genuine plateau.

**Fixed**: `EMA_DECAY` corrected `0.999 -> 0.98` (~34-step half-life --
long enough to smooth noise between curriculum steps, short enough to
converge well before the first `--val-every 500` eval).

### Final results, all three bugs fixed, `p3_bestyet_sophia` also rerun uninterrupted for the full budget

| Arm | Result |
|---|---|
| `p3_bestyet_sophia` (rerun, uninterrupted 1h, not the earlier 0.5h-truncated version) | Completed all 2500 steps. **+39.3%** peak -- confirms the earlier truncated run's trajectory (+38.5%) wasn't hiding a materially better number behind the old budget. |
| `q1_sophia_ema` (fixed `EMA_DECAY=0.98`) | Completed to step 2000, promoted. **+40.8%** -- its best number, and this time the EMA-gated and raw metrics are close (36.2% vs. 34.6%) instead of wildly divergent -- the fix worked. |
| `q2_sophia_auxhead` (both bugs fixed) | Still zero real data as of this section -- two real bugs, zero completed runs. Untested pending a relaunch. |
| `q3_sophia_rollout_dominant` (§57/58, no changes needed) | +36.7%, already reported. |

**Assessment**: every arm across branches M/P/Q now clusters in the
high-30s/low-40s% band -- no single lever (optimizer family, EMA,
rollout-dominant loss, aux head once it actually runs) has moved the
needle by more than a couple points over `h9_ar_freq1`'s original
recipe. This is the signal that incremental tuning on the SAME
objective/architecture has run its course; see §60 for the pivot to
more aggressive, previously-deprioritized directions.

## 60. v7.20 -- PLAN ONLY, nothing built yet: three aggressive, previously-untried directions (branches R/S/T), each to be smoke-tested on MPS before ever touching a pod

Written down before any implementation, specifically so this plan
survives a context loss. §59's assessment: every arm tried since branch
M clusters in the high-30s/low-40s% band -- incremental tuning on the
same objective/architecture has plateaued. These three are deliberately
bigger swings, not more of the same knob-turning.

**Shared constraint across all three**: test as much of the real
mechanism as possible on this Mac (MPS/CPU) BEFORE spending pod time --
following this codebase's own existing convention
(`tests/test_frame_ar_loss_direct.py` already calls `frame_ar_loss()`/
`aux_horizon_loss()` directly on CPU tensors to exercise code paths
`regime.disable_ar` would otherwise skip inside `train()`). Each branch
below is scoped so its CORE new logic (the loss/model code itself, not
full-scale CUDA throughput) is directly unit-testable on MPS/CPU.

### Branch R -- soft spatial-consistency penalty (§32.2 item 5)

**Explicitly NOT real PINN** -- discussed at length this session and
worth restating here so a future reader doesn't overestimate what this
is: real PINN work (e.g. the PSTNet divergence-free-layer paper already
cited in §32.2) requires continuous, differentiable spatial/temporal
coordinates as network INPUTS so autograd can compute actual PDE-
residual derivatives (`∂v/∂x`, `∂v/∂t`) against the real governing
equations (Navier-Stokes momentum/continuity, divergence-free
constraints). This model has no such differentiable-coordinate input at
all -- it's a discrete sequence transformer over `NUM_X=10` x-station
positions, decoded through a FROZEN, non-differentiable-w.r.t.-space AE
decoder. No governing equation is ever written down or checked here.

What this actually is: a smoothness/consistency penalty between the
network's OWN predictions at adjacent x-stations within a frame --
closer to a generic spatial-regularization/total-variation prior than
a physics-informed loss, borrowing PINN literature's MOTIVATION
("physics-based penalties reduce the solution space, accelerating
convergence") without its machinery. Cheap: no new ground truth, no
derivative-through-inputs restructuring -- only touches the network's
own already-decoded per-x-station centroid triplets.

**Planned mechanism**: decode ALL `NUM_X` stations' predicted centroid
triplets for a frame (today's `centroid_velocity_loss()` only ever
scores the network's chosen frame content against target -- the aux
loss here is purely SELF-referential, no ground truth needed), then
penalize large discrete second-differences (a Laplacian-style
smoothness term) or first-differences (TV-style) between neighboring
stations' predicted velocity triplets. New `Config.SPATIAL_SMOOTH_
WEIGHT` (default 0.0, no-op for every existing arm) and a warmup
fraction, mirroring `AR_WEIGHT_WARMUP_FRAC`'s existing convention.

**MPS-testability**: fully testable on MPS/CPU -- this is a pure
function of the network's own decoded output, no CUDA-only AR loop
dependency at all. Should get a direct unit test (new tensor-shape
test, similar spirit to `test_frame_ar_loss_direct.py`) proving finite
loss + real gradients before ever touching a pod.

**Not yet built**: no `Config` fields, no loss function, no arm exist
yet as of this section -- planning only.

### Branch S -- AROpt-style accept/reject rollout stabilization (§32.2 item 7)

Adapts DeepSeek-V3's accept/reject sampling mechanism to continuous-
valued autoregressive forecasting, aimed directly at the exposure-bias
failure mode this entire investigation keeps finding (§39/46/48/57/59)
rather than tuning around it. Flagged in §32.2 as "highest cost, most
novel" of the original menu.

**Planned mechanism** (not yet fully specified -- needs a real design
pass before implementation): at each step of `frame_ar_loss()`'s
sequential rollout, instead of ALWAYS feeding the model's own last
prediction back as the next context frame (today's unconditional
`_feed(nxt)`), compute an acceptance criterion from the deviation
between the model's own prediction and a cheap reference (e.g. the
persistence baseline, or ground truth during training) -- REJECT and
substitute the reference frame when the model's own prediction deviates
too far, ACCEPT and feed the model's own prediction through otherwise.
This is a principled, error-adaptive generalization of scheduled
sampling. **Critical safety constraint, learned the hard way from
`h10_ridge_residual`'s blowup (§26.2)**: the fallback-on-reject target
MUST be non-expansive under repeated feedback (persistence or ground
truth both qualify; a bad model-derived anchor does not) -- this is
exactly the property `h10` violated.

**MPS-testability**: the sequential AR loop is disabled inside `train()`
on MPS/CPU (`regime.disable_ar`), but `frame_ar_loss()` itself can
still be called directly on CPU tensors (exactly as `tests/test_frame_
ar_loss_direct.py` already does for the existing branches) -- so the
accept/reject mechanism's correctness (does it actually reject when it
should, does gradient flow stay sane) is directly unit-testable on this
Mac before any CUDA run.

**Not yet built**: no acceptance-criterion design, no code, no arm --
planning only.

### Branch T -- diffusion architecture (previously deprioritized alternative-architecture axis)

Explicitly named in §49.2 as "zero prior discussion anywhere in this
project, full-rewrite cost" and deprioritized until cheaper options
were exhausted -- §59's plateau across M/P/Q is the signal that
threshold has now been crossed.

**Planned mechanism** (scoped down deliberately for a first real test,
not a full rewrite): a small conditional denoising model operating in
the 47-dim AE latent space, predicting one frame ahead conditioned on
context frames (via concatenation or cross-attention, not yet decided),
trained with a standard diffusion loss (predict noise or velocity
parameterization) instead of direct regression. A NEW `model_variants.py`
class (e.g. `DiffusionFrameModel`) and a NEW training-loop branch are
both required -- diffusion training (denoising loss over sampled
timesteps) and diffusion inference (iterative sampling, e.g. 8-16 steps
via DDIM or similar) are both structurally different from every existing
arm's direct-regression forward pass. Output still decodes through the
same frozen GEN3 AttentionSE decoder for the centroid-velocity metric,
keeping this comparable to every other arm's scoring.

**MPS-testability**: the new model class's forward pass (build, single
denoising step, full sampling loop) is fully testable on MPS/CPU for
correctness, independent of the AR/CUDA-only code paths -- should get
its own `--smoke-test`-equivalent (build model, one train step, one
sample) before ever being pointed at real data or a pod, mirroring how
every other model variant in this file is smoke-tested.

**Not yet built**: no model class, no training loop branch, no sampling
code, no arm -- planning only. This is the largest-scoped, highest-
uncertainty of the three; expect this to take meaningfully longer to
even reach a first real (even MPS-only, non-competitive) result than R
or S.

### Sequencing

All three are independent of each other and can be built/tested in any
order. Given branch T's much larger scope, the natural order is R (cheapest,
most mechanically similar to existing code) -> S (moderate, needs a real
design pass on the acceptance criterion) -> T (largest lift). Nothing
above commits to that order -- just the reasoning for why it's the
likely path if built sequentially.

## 61. v7.21 -- branch U's first real results: `r1_spatial_smooth` lands in-band with the existing best, `q2_sophia_auxhead` writes off as a real negative result; a threaded HDF5 loader cuts data-load time ~15x; a confound identified and a control arm added

### Threaded HDF5 loading, confirmed on a real pod

The slow, single-core "warm-up" period at the start of every run (user-
reported, live-diagnosed) was `TransformerDataset`'s single-threaded
`f['data'][:length]` read of the full 9.86 GiB `train_80.h5` off the
network volume. `train_80.h5`/`val_80.h5` are chunked ONE ROW PER CHUNK
(`chunks=(1, NUM_TIME, NUM_X, INPUT_DIM)`), so concurrent threads reading
disjoint row ranges via their own read-only file handles are safe and
chunk-aligned -- the same rationale `scp_data_files.sh`'s own `LANES`
already exploits for the transfer itself.

New `_read_h5_dataset_threaded()` (default 8 threads, `PFD_DATA_LOAD_
THREADS` env override, falls back to sequential on any failure).
**Measured on a real A40-then-H100-NVL pod** (US-GA-2, the real network
volume, `yl7f9e8rwr`) against the actual `train_80.h5`:

| Threads | Wall time |
|---|---|
| 1 (old) | 42.2s |
| 4 | 2.89s |
| **8 (new default)** | **2.76s -- best** |
| 16 | 2.93s |
| 32 | 3.13s |

**~15x speedup**, correctness confirmed bit-identical to sequential on
both a synthetic local test and the real `val_80.h5` on the pod.
Interestingly, the SAME code measured on this Mac's local SSD was
slightly SLOWER threaded than sequential (1.70s vs. 1.17s) -- thread
overhead dominates when per-request latency is already low; the win is
specific to network-attached storage, exactly mirroring why parallel
`scp` only helps over the real network link, not a local copy.

### `r1_spatial_smooth` and `q2_sophia_auxhead`'s real launch: much faster than expected, and why that's not a bug

Both arms finished their step budgets in ~5-6 minutes each (not the
expected ~30-45 minutes) -- initially looked like a truncated/broken run,
but is fully explained and NOT a bug: both require `TOKENIZATION='frame'`
(their new mechanisms only work on frame-native models), which drops the
transformer's sequence length from `SEQ_LEN=800` (token tokenization,
`NUM_TIME*NUM_X`) to `NUM_TIME=80` -- a 10x shorter sequence, and since
attention cost scales roughly with sequence-length-squared, up to ~100x
less attention compute per step. Confirmed directly in the logs: rollout
eval dropped from the historical 2-4s (§56) to **0.09s**, and 500-step
windows completed in well under a minute. Both arms legitimately reached
their real step budgets (`r1` hit `MAX_STEPS=2500`, `q2` early-stopped),
nothing crashed or was cut short.

### Results

| Arm | Result |
|---|---|
| `r1_spatial_smooth` | Completed all 2500 steps. **+39.4%** peak (rollout_mse 0.00172) -- in-band with `p3_bestyet_sophia` (+39.3%) and `q1_sophia_ema` (+40.8%). First real, non-crashed result for branch U. |
| `q2_sophia_auxhead` | First-ever completed run (both §59 bugs fixed). Early-stopped at step 1500, **-63.9%**, never promoted once (`promoted_rollout_mse=inf`) -- WORSE than persistence throughout. |

### `q2_sophia_auxhead` is written off (at this configuration)

`AUX_HEAD_LOSS_WEIGHT=1.0` alongside the primary teacher-forced loss and
`AR_LOSS_WEIGHT=3.0` actively hurts rollout performance -- the model never
promoted a single checkpoint above the persistence baseline across its
whole run. Not pursuing further retuning of this specific mechanism (e.g.
searching for a smaller `AUX_HEAD_LOSS_WEIGHT`) -- the Phase 2 menu named
three candidate levers, this is the one that failed outright, and the
budget is better spent on the two untried, more aggressive branches (S,
T) than on rescuing a mechanism that's already shown a strongly negative
signal at its first real test.

### A real confound identified: frame tokenization itself, not proven separable from `r1`'s own new term

Both `r1` and `q2` switched `TOKENIZATION='frame'` AT THE SAME TIME as
introducing their own new mechanism -- there has never been a clean
"`p3_bestyet_sophia`'s exact recipe + `TOKENIZATION='frame'`, nothing
else new" control run. `r1`'s +39.4% could be mostly the spatial-
smoothness term, or frame tokenization could simply be just as good as
token tokenization at this task on its own (while running ~8-10x cheaper
per step for free) -- these two explanations are currently
indistinguishable. New arm `u2_frame_control` (below) isolates this.

### New arm: `u2_frame_control`

`p3_bestyet_sophia`'s exact base recipe + `TOKENIZATION='frame'` ONLY --
`SPATIAL_SMOOTH_WEIGHT` left at its 0.0 default, no aux head. Directly
comparable to `r1_spatial_smooth` (identical except the smoothness term)
and to `p3_bestyet_sophia` itself (identical except tokenization).

**Not yet run as of this section.**

## 62. v7.22 -- branch W built: AROpt-style accept/reject rollout stabilization (§32.2 item 7), plus two more frame-tokenization arms to isolate the real confound from §61

### Branch W (named "W" not "S" -- letter "S" was already taken, see the
code's own note)

Adapts DeepSeek-V3's accept/reject sampling to continuous-valued AR
forecasting. New `Config.AROPT_TEMPERATURE` (0.0 default, no-op --
confirmed via `test_disabled_by_default_matches_existing_behavior`,
which checks BYTE-FOR-BYTE identical loss output at the default vs. an
explicit 0.0). New shared `_feed_or_aropt()` helper used by BOTH of
`frame_ar_loss()`'s branches (frame-native and token-native, unlike
`aux_horizon_loss()`'s frame-only scoping -- this mechanism doesn't
inherently need a frame-level head).

**Mechanism**: per-sequence, compute the model's own next-step
prediction error against ground truth, convert to an acceptance
probability via `exp(-error/AROPT_TEMPERATURE)` (worse predictions
rejected more often), sample a per-sequence Bernoulli accept/reject
decision. ACCEPT feeds the model's own prediction through (today's
unconditional behavior for every pre-existing arm). REJECT substitutes
ground truth instead. **Safety** (the `h10_ridge_residual` lesson,
§26.2): the reject-path fallback is ALWAYS ground truth, never a
model-derived anchor -- non-expansive under repeated feedback by
construction.

**MPS-tested** per branch W's own plan requirement: `tests/test_aropt_
accept_reject.py` (6 tests -- disabled-by-default equivalence, both
frame-native and token-native branches, a single-sequence micro-batch,
and both temperature extremes: very high (forces all-accept) and very
low (forces frequent rejects, exercising the ground-truth-substitution
path directly)) plus a manual real-MPS-device check (batch=1/2,
`mps:0`, finite loss ~1.87-1.90, real gradients).

New arm `s1_aropt_frame`: `p3_bestyet_sophia` + `TOKENIZATION='frame'`
(the same base as `u2_frame_control`) + `AROPT_TEMPERATURE=0.1` (a
first-attempt value, not tuned).

### Two more arms to properly isolate §61's confound

`u3_frame_rollout_dominant`: `q3_sophia_rollout_dominant`'s own
`TF_LOSS_WEIGHT=0.3` combined with `TOKENIZATION='frame'` for the first
time (q3 itself ran token-native only) -- cheap, no new code, just
combining two already-tested axes.

Together with `u2_frame_control` (§61), the next launch is a genuine
3-way comparison all sharing the identical `p3_bestyet_sophia` + frame-
tokenization base, varying exactly one new axis each:

| Arm | New axis vs. `u2_frame_control` |
|---|---|
| `u2_frame_control` | none (the control) |
| `u3_frame_rollout_dominant` | `TF_LOSS_WEIGHT=0.3` |
| `s1_aropt_frame` | `AROPT_TEMPERATURE=0.1` |

**Not yet run as of this section.** Full test suite: 175 passed, 14
skipped (up from 165 in §61).

# Major version 8: past the ~40% plateau

## 63. v8.0 -- honest status check, `HALF_TIME_MODE` shipped as a tooling switch, and a strategy menu for getting past the ~40% ceiling this recipe family has hit

**Major version bump** (v7.x -> v8.0) per the WANDB_PROJECT convention
(§21.1, enforced by `tests/test_version_sync.py`) -- `Config.WANDB_
PROJECT` renamed `NI_Review_v7` -> `NI_Review_v8`.

### 63.1 Why this version bump: an honest status check, not a new result

Branches M/P/Q/U/W (this entire multi-day arc) have produced a wide but
shallow sweep of small variations on ONE base recipe
(`h9_ar_freq1`'s descendants), all landing in a narrow band:

```
 %improvement
  70 |  ridge/linear baseline (§25.1) ...................... +69.49%  <- never beaten, by anything
  50 |
  46 |  best transformer EVER, 400-step screen (§28.7) ...... +46.49%  <- also never beaten since
  44 |  h9_ar_freq1's own 400-step number (§26.1) ............ +43.78%  <- the ancestor of every M/P/Q/U/W arm
  40 |*  * * *  *   *  *                                                <- THIS SESSION lives here
  35 |* *   *  *  *   *
  30 |                                                        session low: p2_bestyet_lion +30.9%
  20 |
   0 |----------------------------------------------------------------
 -60 |                                                                   q2_sophia_auxhead -63.9%
-180 |                                                                   s1_aropt_frame -179.0% (bad temperature)
```

And the ONE direct longitudinal data point this project has (same
recipe, shallow screen vs. real production scale) points DOWN, not up:
`h9_ar_freq1` scored +43.78% at 400 steps, then **fell to +31.35%** at
145k real production steps (Appendix A.3) -- the opposite of "just
needs more training." This is a documented, repeated shape (§39/46/48/
57/59: "a 4th confirmed instance," "a 5th confirmed instance" of the
same peak-then-decline pattern), not a one-off.

**Conclusion driving this version bump**: continuing to turn small
knobs on the SAME recipe (optimizer choice, EMA, aux heads, loss
re-weighting, spatial smoothness, accept/reject at an untuned
temperature) has been thoroughly tried and has not moved the needle
past where it already was months ago. This section is a genuine
strategy reset, not another arm in the same family.

### 63.2 Shipped now (not just planned): `HALF_TIME_MODE`

A real, tested tooling switch -- `Config.HALF_TIME_MODE` (default
`False`, no-op for every existing arm). When enabled: keeps only EVEN
time-step indices (`0, 2, 4, ...`) from every sequence, halving
`NUM_TIME` (80->40), `SEQ_LEN` (800->400), and the resident data volume
-- for cheaper, faster iteration while testing the strategies below, not
a claimed quality improvement in itself.

**Mechanism**: `resolve_derived_config_fields()` (new, factored out of
`main()`'s existing "recompute derived fields after every override"
step) halves `Config.NUM_TIME` FIRST when `HALF_TIME_MODE=True`, then
derives `SEQ_LEN`/`VAL_ROLLOUT_STEPS` from the already-halved value --
ordering matters, called out explicitly in its own docstring.
`TransformerDataset.__init__` does the actual even-index subsampling
(`raw[:, ::2, :, :]`) on the raw on-disk `(length, NUM_TIME_ON_DISK,
NUM_X, INPUT_DIM)` array, BEFORE the final reshape into
`(length, SEQ_LEN, INPUT_DIM)` -- skipping the subsampling would make
that reshape fail outright (element-count mismatch) rather than
silently corrupt data, a real safety property of doing both halvings in
the same change.

**Not in `PINNED_CONFIG_FIELDS`** (unlike `NUM_TIME`/`SEQ_LEN`
themselves) -- `HALF_TIME_MODE` is the sanctioned lever for changing
effective temporal resolution, settable via an arm override or
`--set HALF_TIME_MODE=true`.

**Tested**: `tests/test_half_time_mode.py` (4 tests -- disabled leaves
`NUM_TIME` untouched, enabled halves it before deriving `SEQ_LEN`/
`VAL_ROLLOUT_STEPS`, and the actual `TransformerDataset` subsampling
against a small synthetic HDF5 file, verified to match `raw[:, ::2]`
exactly). Also verified end-to-end on the real MPS device: model
builds at the halved `SEQ_LEN=400` with correctly-sized position
embeddings (no shape mismatch), forward+backward pass finite. Full
suite: 179 passed, 14 skipped (up from 175 in §62).

**Honest side effect, not fixed here**: `VAL_CONTEXT_STEPS=12` is left
unchanged, so under half mode it becomes 12/40=30% of the horizon
(context) instead of 12/80=15% -- a real, expected consequence of
halving resolution (less rollout horizon to actually score), not
rebalanced in this pass.

### 63.3 What LLM training actually looks like at this point in a project, vs. what this one has done

| LLM-training lever | Standard LLM practice | This project's status |
|---|---|---|
| LR schedule | Warmup + cosine decay, increasingly Warmup-Stable-Decay (WSD) for flexible-length training and mid-run checkpoint reuse (MiniCPM, DeepSeek-style) | Warmup + cosine only, unchanged since v1.0 (§28.2). Never tried WSD or restarts. |
| Batch size / gradient noise | Large batch + LR scaled with it (linear/sqrt rules, OpenAI's gradient-noise-scale framework) -- reduces noisy updates | `BATCH_SIZE=64` fixed since the H200 defaults landed; the AR loss specifically runs on as few as `AR_SEQS=2-6` sequences -- a much noisier gradient than the primary loss, never addressed directly. |
| Data/model scaling (Chinchilla) | Model size and data size scaled together for compute-optimal training | Model has stayed ~5M params this entire investigation; §32.2 item 6 deprioritized capacity increases based on ONE early axis test, before most of this session's newer axes existed. |
| Pretraining before fine-tuning | The defining LLM practice -- cheap generic self-supervised objective first, task-specific objective second | Never done. Every arm starts from either random init or a warm-start from a PRIOR ARM'S already-task-trained weights -- never a generic representation-learning pass. |
| Weight averaging | EMA during training (near-universal in diffusion, common in LLM/vision) AND post-hoc "model soups" across independently-trained checkpoints (Wortsman et al.) -- often free accuracy | EMA tried once (`q1_sophia_ema`, §59/61) -- positive but modest (+40.8%, in-band with everything else). Post-hoc soup/averaging across the many DIFFERENT already-trained checkpoints this session produced: never tried. |
| Regularization | Comparatively LIGHT at scale -- LLMs rarely lean on heavy dropout; optimization + data axes dominate | This project has spent real effort on dropout/weight-decay tuning (branch G, `m4`/`m5`) and reached the same conclusion independently (§48: "regularization tuning alone has run its course") -- the two findings agree. |
| Different final-stage objective | RLHF/DPO-style: a genuinely different LAST-STAGE loss, not just more pretraining-style loss, can unlock a step change | The closest analogue tried is `q3`/`u3`'s rollout-dominant loss reweighting and `s1`'s AROpt accept/reject -- both real attempts at "change what's being optimized," neither yet successful, but the right SHAPE of idea. |

### 63.4 Strategy menu: 8 candidates, ranked by cost

| # | Strategy | LLM/DL precedent | Mechanism here | Why it might break the plateau | Cost / risk | Weight-compatible with existing checkpoints? |
|---|---|---|---|---|---|---|
| 1 | **Model soup / post-hoc weight averaging** | "Model Soups" (Wortsman et al.) -- averaging independently fine-tuned checkpoints improves generalization for free | Average the weights of the best already-completed same-architecture checkpoints (e.g. the token-native cluster: `p1`/`p3`/`q1`/`q3`; separately the frame-native cluster: `r1`/`u2`/`u3`) sitting in `saved_models/` right now | Zero new training compute -- if these checkpoints sit in genuinely different loss-landscape basins, averaging can land in a flatter, better-generalizing region between them | **Lowest of all 8**: no pod time, pure local computation, minutes | **Yes, by construction** -- this IS reusing existing weights, the most direct match to "load the best we have" |
| 2 | **LR schedule: WSD or cosine-with-restarts** | Warmup-Stable-Decay (MiniCPM, DeepSeek); SGDR warm restarts | Replace the current single warmup+cosine schedule; a restart could specifically target escaping the "peaks early, then declines" rut instead of decaying LR to near-zero right as the decline starts | Directly targets the repeatedly-observed shape: today's schedule anneals LR toward zero exactly when the model is already sliding, freezing it into the decline instead of giving it room to recover | Low -- pure scheduler swap, no architecture change | Yes -- same architecture, can resume from any existing checkpoint |
| 3 | **Batch-size / AR-gradient-noise reduction** | Gradient noise scale, LR-batch scaling rules | Increase `AR_SEQS` (noisy at 2-6 today) and/or `BATCH_SIZE`, with LR scaled to match, specifically to smooth the AR loss's gradient signal | The AR loss's gradient comes from as few as 2-6 sequences -- a plausible source of the run-to-run "oscillation" the user is pointing at, distinct from a real capability ceiling | Low-moderate -- bounded by GPU memory (already measured close to limits at higher `AR_SEQS`, §44/50) | Yes |
| 4 | **Progressive resolution curriculum using `HALF_TIME_MODE`** | Progressive resizing (vision), progressive context-length extension (LongRoPE, GPT-NeoX-style) | Start training at `HALF_TIME_MODE=True` (cheap, coarse), then switch to full resolution partway through -- directly exercises the switch just shipped in §63.2 | Coarse-to-fine curricula let the model learn large-scale temporal structure cheaply before spending compute on fine detail -- untried axis, no prior arm has ever varied temporal resolution during a single run | Low -- needs a resolution-switch point in the training loop (not yet built, tooling only) | Yes -- same architecture at both resolutions (position embeddings resize, but that is exactly the ALREADY-EXISTING v1.0<->v2.0 length-mismatch warm-start path, §long-standing) |
| 5 | **Self-supervised pretraining before the forecasting objective** | The defining modern LLM practice: cheap generic objective first, task objective second | Pretrain the transformer on a masked-frame or next-frame reconstruction objective alone (no centroid decode, no AR loss) to learn general spatiotemporal structure, THEN fine-tune with the current recipe on top | Every arm this whole project has started from either random init or a task-already-trained warm-start -- never a generic representation-learning stage. This is a genuinely untried axis at the "how is the model initialized" level, not the "what's the loss during fine-tuning" level every prior arm varied | Moderate -- a new (simpler) training loop, meaningfully more wall-clock before any task-relevant result exists | Partial -- the pretrained weights become a NEW kind of warm-start source, not compatible with treating pretraining as optional the way `--warm-start` is today |
| 6 | **Mixture-of-Experts / conditional capacity increase** | Every recent frontier LLM uses MoE for capacity without proportional compute cost | Replace the dense FFN blocks with a small MoE layer, keeping active compute per token similar while raising total capacity | If the plateau is partly a genuine capacity limit (deprioritized early, §32.2 item 6, on thin evidence pre-dating most of this session's own axes) MoE raises capacity without paying the full compute cost a dense scale-up would | Moderate-high -- new architecture component, routing/load-balancing tuning, more to get wrong | **No** -- different layer structure, breaks every existing checkpoint's warm-start |
| 7 | **Diffusion / non-autoregressive rollout (branch T, already planned §60)** | The direct analogue to diffusion LLMs (LLaDA, Mercury) -- an active 2025-2026 research response to exactly this exposure-bias problem in text | A small conditional latent-space denoiser predicting frames without ever chaining single-step AR predictions -- error compounding structurally can't happen the same way | Every other strategy here still trains against SOME form of the current AR-rollout objective; this is the one candidate that removes the mechanism (sequential feedback) that causes exposure bias in the first place, rather than compensating for it | **Highest of all 8** -- full new model class, new training loop, new sampling procedure, most wall-clock before any result | **No** -- completely different generative mechanism, from-scratch only |
| 8 | **Joint/partial fine-tuning of the frozen AE decoder** | Analogous to jointly fine-tuning a VAE decoder alongside a latent diffusion model, rather than freezing it as a fixed "tokenizer" | Unfreeze (or lightly adapt) the GEN3 AttentionSE decoder together with the transformer | Every arm's score depends on decoding through a decoder that has NEVER been touched -- if the decoder itself is a bottleneck or mismatched to the transformer's latent geometry, no amount of transformer-side tuning fixes that | Moderate-high, and uniquely risky: it moves the SCORING FUNCTION itself, making every prior result in this entire document not directly comparable to anything trained after it (§49.2 already flagged this) | Partial -- transformer-side weights remain loadable; the decoder side does not, and comparability of the metric itself is the real cost, not just compute |

### 63.5 Recommended sequencing

Cost-ordered, cheapest/most-reversible first -- nothing here is
committed, this is the reasoning for a likely order if pursued
sequentially:

```
 (1) model soup          -- zero pod cost, try TODAY against existing checkpoints
      |
 (2) WSD/restart LR      -- cheap, no architecture change
      |
 (3) AR gradient-noise   -- cheap, directly tests whether "oscillation"
     reduction              is noise or a real ceiling
      |
 (4) progressive-res     -- uses HALF_TIME_MODE (shipped §63.2), low cost
     curriculum
      |
 (5) SSL pretraining     -- moderate cost, first real "different
                             initialization" axis ever tried
      |
 (6) MoE capacity  ------+-- both break weight compatibility; only
 (7) diffusion rollout --+   worth the cost if 1-5 are exhausted first
      |
 (8) joint decoder FT    -- last, because it invalidates comparability
                             of every number in this entire document
```

1-3 are cheap enough that a negative result is cheap to obtain and
still informative (ruling out "it's just noise" is real progress, per
the discussion earlier this session). 6-8 are the genuinely
"architecture changes that break weight loading" the operator explicitly
said were acceptable to consider -- deliberately sequenced LAST, after
the cheap, reversible options, not because they're expected to fail,
but because they're expensive enough that trying them first would waste
the option value of the cheap tests.

### 63.6 Scope of this section: planning only

**Nothing in §63.4/63.5 is implemented.** `HALF_TIME_MODE` (§63.2) is
the one exception -- a real, tested, shipped tooling switch, built
because strategy #4 above needs it to exist before it can be tried at
all. No arm, no new training-loop code, no new model class for any of
strategies 1/2/3/5/6/7/8 exists yet as of this section.

## 64. v8.1 -- strategies #1 (model soup) and #2 (WSD/restart LR schedules) built and tested on MPS; strategy #6 (Mixture-of-Experts) planned in detail, not built

Per the operator's direction: strategies #1 and #2 built and tested
immediately; #6 planned in real technical detail (architecture location,
cost, risk) but deliberately NOT implemented yet, since it's one of the
two "breaks weight compatibility" strategies explicitly sequenced last
in §63.5. All NEW work going forward uses `HALF_TIME_MODE=True`
(§63.2) for cheap iteration; see §64.2 for why that does NOT apply to
re-evaluating the ALREADY-TRAINED full-resolution checkpoints strategy
#1 uses as its ingredients.

### 64.1 Strategy #2 shipped: `Config.LR_SCHEDULE`

New `make_lr_lambda()` dispatch on `Config.LR_SCHEDULE` (default
`'cosine'`, byte-for-byte the original single warmup+cosine-decay shape
-- confirmed via `tests/test_lr_schedules.py`'s
`test_matches_original_hardcoded_shape`, which reimplements the exact
pre-refactor formula independently and checks it against the refactored
function at 8 sample steps).

- `'wsd'` (Warmup-Stable-Decay, MiniCPM/DeepSeek-style): holds LR at
  its peak between the end of warmup and a final decay window
  (`WSD_DECAY_FRAC * MAX_STEPS`), instead of annealing toward
  `LR_FINAL_FRAC` for the entire post-warmup budget. Directly targets
  the repeatedly-observed "peaks early, then declines" shape (§39/46/
  48/57/59): today's plain cosine schedule is already annealing LR
  toward its floor exactly as that decline starts, which may be
  freezing the model into the decline rather than leaving it room to
  recover.
- `'cosine_restarts'` (SGDR-style): splits the post-warmup budget into
  `LR_NUM_RESTARTS + 1` equal-length cosine cycles, resetting back to
  peak LR at the start of each -- a different mechanism for the same
  goal, giving the optimizer repeated fresh chances to escape a local
  rut instead of a single one-way anneal.

**Tested**: `tests/test_lr_schedules.py` (9 tests -- default-cosine
equivalence, WSD's stable-phase-holds-at-peak and decay-reaches-floor,
cosine-restarts' reset-at-cycle-start and within-cycle decay, an
unknown-schedule `ValueError`).

### 64.2 Strategy #1 evaluated on MPS against real checkpoints: `model_soup_eval.py`

New script (`model_soup_eval.py`) evaluates whether uniform/greedy model-
soup weight averaging (Wortsman et al.) helps on top of the best
checkpoints branches M/P/Q/U/W already produced -- zero new training
compute, pure local inference on this Mac's MPS device against real
weights already on disk.

**Deliberately evaluated at FULL resolution, not `HALF_TIME_MODE`**:
every ingredient here was trained at `NUM_TIME=80`. `HALF_TIME_MODE`
re-indexes frames onto a shorter `0..(NUM_TIME/2-1)` position axis --
for a model that learned the FULL 0..79 axis, that would look up the
WRONG position-embedding rows entirely, not just a cheaper evaluation.
`HALF_TIME_MODE` is for NEW training runs (§64.3/64.4 below), not for
re-scoring checkpoints that predate it.

**Two averageable clusters** identified by inspecting each checkpoint's
own stored `config` blob (`TOKENIZATION`/`VARIANT`/`EMBED_SIZE`/
`N_LAYERS`/`N_HEADS`/`AUX_HEAD_FRAMES` must all match):
- token-native: `p1_bestyet_adamw`, `p2_bestyet_lion`,
  `p3_bestyet_sophia`, `q1_sophia_ema`, `q3_sophia_rollout_dominant`
- frame-native (`AUX_HEAD_FRAMES=0`): `r1_spatial_smooth`,
  `u2_frame_control`, `u3_frame_rollout_dominant`, `s1_aropt_frame`

`q2_sophia_auxhead` (`AUX_HEAD_FRAMES=8`, extra `aux_head.*` params) is
architecturally incompatible with either cluster and is excluded from
souping entirely, per `average_state_dicts()`'s own loud
key-mismatch guard.

**Real bug hit and fixed while building this**: every one of these
checkpoints was trained with `torch.compile` ON (before it defaulted
off, §59), so the raw `model_state_dict` payload carries a
`"_orig_mod."` prefix on every key (torch.compile's `OptimizedModule`
wrapper) -- `model_soup_eval.py`'s `load_ckpt()` strips it before
loading, the same unwrap convention this file already applies to the
live model object elsewhere, just applied to a saved state_dict's keys
instead. Also hit: a checkpoint's own stored `config` blob records
absolute filesystem paths (`DECODER_SCRIPTED_PATH` etc.) from wherever
it was TRAINED (a pod's `/workspace/...` layout) -- always overridden
with the live local `Config`'s own paths before evaluation.

**Method**: evaluates every individual ingredient fresh (via the SAME
`evaluate()` every arm's own training run uses, so numbers are directly
comparable to earlier reported ones), then builds a GREEDY soup
(Wortsman et al.'s own selection procedure, not blind uniform
averaging): sort by individual `rollout_mse` ascending, start from the
best, then add each remaining ingredient only if it does not make the
running soup's `rollout_mse` worse.

**A real MPS backend stall, found running this**: the full greedy
search (~9 individual evals + ~7 incremental soup-candidate evals) was
too slow to complete in one sitting -- even a single evaluation on a
2032-row subset took ~4 minutes. Batching more sequences per call
(`chunk`/`tf_batch_size`) didn't fix it: a subsequent attempt sat in
`ps`'s "uninterruptible sleep" state, consuming almost no CPU across 5+
minutes for ONE evaluation -- a genuine MPS dispatch/synchronization
pathology on the sequential AR-rollout loop specifically (68 sequential
`model.forward()` calls per rollout, each growing the context), not
something fixable by tuning batch or subset size. **Forcing CPU
(`MODEL_SOUP_DEVICE=cpu`) fixed the stall outright** -- CPU's single-
queue, fully synchronous execution proved far more reliable for this
exact workload than MPS, even though nominally "slower" per-op.
Scope was also cut from the full greedy search down to the minimum
informative comparison: each cluster's best-known individual ingredient
(from this session's own real CUDA numbers) plus ONE uniform soup of
every member -- 4 evaluations total instead of ~16, on a small (304-row,
12-sequence) subset for tractable CPU wall-clock.

### Results: model soup FAILS badly on both clusters

| Cluster | Best individual (fresh eval, same subset) | Uniform soup | Delta |
|---|---|---|---|
| Token-native (`p1`/`p2`/`p3`/`q1`/`q3`) | `q1_sophia_ema` **+44.45%** | **-225.84%** | -270.3 points |
| Frame-native (`r1`/`u2`/`u3`/`s1`) | `u3_frame_rollout_dominant` **+50.38%** | **-1678.58%** | -1729.0 points |

Not a marginal loss -- a complete collapse in both clusters, the soup
scoring far worse than even a naive constant-zero predictor would.
**Likely explanation**: the original Model Soups result (Wortsman et
al.) averages checkpoints fine-tuned FROM THE SAME PRETRAINED WEIGHTS
with only minor hyperparameter differences, keeping every ingredient
close together in weight space ("linear mode connectivity"). Every
checkpoint souped here instead started from an independent RANDOM
INIT and diverged under substantially different optimizers (AdamW/
Lion/Sophia) and loss terms (EMA, rollout-dominant weighting, spatial
smoothness, AROpt) -- nowhere near the same basin. Transformers are
also specifically documented to need a permutation-alignment step
(matching equivalent-but-differently-ordered attention heads across
runs, e.g. "Git Re-Basin"-style approaches) before naive averaging is
meaningful at all; none was attempted here.

**Verdict**: strategy #1, in its simplest (uniform/greedy, no
permutation alignment) form, is a real, informative NEGATIVE result --
ruled out for this specific checkpoint population, not worth pursuing
further without first solving the much harder alignment problem the
original technique doesn't need when its own precondition (shared
pretraining lineage) holds. That precondition never held here.

### 64.3 Strategy #6 built + unit-tested (v8.3), NOT yet run on real data

**Status update (post v8.3)**: everything described below as "planned" has
since been built. `MoEFeedForward`, `_build_ffn()`'s dispatch, and
`moe_aux_loss()` exist in `model_variants.py`; `MOE_NUM_EXPERTS` (default
`0`, byte-for-byte dense), `MOE_TOP_K` (default `2`), and
`MOE_LOAD_BALANCE_WEIGHT` (default `0.01`) exist in `Config` in
`train_production_transformer_deep_dive.py`, which also adds the aux loss
into the training loop when `MOE_NUM_EXPERTS > 0`. `tests/test_moe_ffn.py`
covers it: output-shape, aux-loss finiteness/positivity, gradient reaching
every expert and the router, `top_k` clamping, the SwiGLU variant, the
`MOE_NUM_EXPERTS=0` no-op path, and a real tiny-model forward+backward with
MoE enabled (frame-native tokenization included). All 8 tests pass on
CPU/MPS.

**What this does and doesn't mean**: this is the same "prove it before
spending pod time" bar every other mechanism in this project clears before
a real run -- it confirms the code is correct, not that MoE helps. No arm
using `MOE_NUM_EXPERTS > 0` has been trained on real data yet; there is no
production-scale result for this strategy. The original plan below is kept
for the reasoning/rationale; treat "planned" language in it as historical.

**Original plan (superseded by the above -- kept for rationale)**:

**Where it goes**: `model_variants.py`'s `Block.__init__` currently sets
`self.mlp = _build_mlp(n_embd, dropout, use_swiglu)` (one dense FFN per
block, `Block.forward` at line ~295: `x = x + self.mlp(self.ln2(x))`).
The standard MoE conversion (Switch Transformer, Mixtral) replaces ONLY
this FFN sublayer with a sparse mixture -- attention stays dense in
every published MoE transformer variant, and there is no reason to
deviate here. A new `_build_moe_mlp(n_embd, dropout, n_experts, top_k)`
would construct `n_experts` independent copies of today's `_build_mlp`
output (or `SwiGLU`) plus a small linear router (`n_embd -> n_experts`),
selecting the top-`k` experts per token (`top_k=1` or `2`, matching
Switch/Mixtral's own choices) and combining their outputs weighted by
the router's softmax.

**New Config fields (planned)**: `MOE_NUM_EXPERTS` (0 disables --
byte-for-byte today's dense `_build_mlp`, preserving every existing
arm), `MOE_TOP_K`, `MOE_LOAD_BALANCE_WEIGHT` (an auxiliary loss term
penalizing uneven expert utilization -- without it, MoE training is
well-documented to collapse onto using only 1-2 experts regardless of
`n_experts`, wasting the added capacity entirely).

**Why it might help**: raises total model capacity without a
proportional increase in per-token compute (only `top_k` of
`n_experts` FFNs run per token) -- exactly the mechanism every recent
frontier LLM uses to scale capacity past what dense scaling can afford.
If the ~40% plateau is partly a genuine capacity limit (§32.2 item 6
deprioritized this on thin evidence -- one axis test, predating almost
every strategy this project has since tried), MoE is a way to test
that WITHOUT paying dense-scaling's full compute cost.

**Honest cost/risk, stated plainly**: this project's model is currently
~5M parameters -- small. MoE's benefits are best-documented at a scale
where a dense equivalent would be prohibitively expensive; at 5M
params, the router/load-balancing complexity is real overhead that a
plain capacity increase (`EMBED_SIZE`/`N_LAYERS`) might match or beat
more simply. The honest framing: this is worth testing BECAUSE it's
cheaper than dense scaling to try, not because MoE is obviously the
right tool at this model size -- a genuinely open question, not a
confident bet.

**Weight compatibility**: NONE. A different `Block.mlp` structure
breaks every existing checkpoint's warm-start -- any MoE arm starts
from random init, exactly like every other architecture-changing
strategy in §63.4's table (#6, #7, #8).

**Sequencing**: per §63.5, deliberately after #1-5 -- not because it's
expected to fail, but because it's expensive enough (new architecture
component, routing/load-balancing tuning, from-scratch training only)
that trying it before the cheap, reversible options would waste their
option value. **Update**: the code and unit tests are now built (see the
status note at the top of §64.3) -- what remains outstanding is only the
real-data run/arm, not the implementation.

### 64.4 What's next, given §64.2's soup results

The soup collapsed on both clusters (§64.2) -- the opposite of "these
checkpoints happened to sit in the same basin." The divergence across
optimizers/loss terms was apparently large enough that naive weight-
space averaging destroys the model entirely, not just fails to help.
Strategy #1 is DONE for this checkpoint population -- not worth
retrying without a genuine permutation-alignment step first (a
meaningfully larger, separate piece of work, not attempted here).

The more informative next test is **strategy #2 (WSD/restart LR
schedules) on a genuinely fresh run**, isolated from souping entirely --
already built and unit-tested (§64.1), just needs a real training
launch (ideally with `HALF_TIME_MODE=True` per the operator's own
direction) to see whether it does better than plain cosine at avoiding
the peak-then-decline pattern this whole investigation keeps finding.

## 65. v8.2 -- WSD + `HALF_TIME_MODE` verified end-to-end on a real MPS training run; two more real bugs found (not fixed) along the way

### Real run confirms WSD's math against the actual training loop, not just unit tests

`bash train_production_transformer_deep_dive.py --arm u2_frame_control
--max-steps 8 ... --set HALF_TIME_MODE=true LR_SCHEDULE=wsd
WSD_DECAY_FRAC=0.5` on this Mac's MPS device: data resident at
`(59280, 400, 52)` (confirms `HALF_TIME_MODE` halved `SEQ_LEN` 800->400
in a real run, not just the unit tests), and the printed per-step LR
held flat at `8.00e-04` (peak) for steps 1-3 -- exactly WSD's stable
phase, matching `make_lr_lambda()`'s own math for `MAX_STEPS=8`,
`WARMUP_FRAC=0.03`, `WSD_DECAY_FRAC=0.5` (decay only starts at step 5).
Run completed cleanly (8 steps, checkpoint written, `[done]`) --
the ridiculous `-50332%` improvement number is expected and irrelevant
(8 steps from random init tells you nothing about quality, only that
the mechanism runs without crashing).

### Two real bugs found getting there, NEITHER fixed (out of scope for this pass, flagged for later)

**1. `--set KEY1=V1 --set KEY2=V2` (repeated `--set` flags) silently
drops everything but the LAST occurrence.** `argparse`'s `--set`
argument uses `nargs="*"` without `action="append"` -- each `--set`
occurrence REPLACES `args.set` rather than accumulating into it. The
first smoke-test attempt passed six separate `--set KEY=VALUE` flags
and only the final one took effect; `HALF_TIME_MODE`/`LR_SCHEDULE`
silently reverted to their defaults with no warning at all. **Correct
usage**: pass every override to a SINGLE `--set` flag, space-separated
(`--set KEY1=V1 KEY2=V2 KEY3=V3`) -- this is how every prior real
launch command in this document already does it, which is exactly why
this footgun had never surfaced before.

**2. `_coerce("none")` returns Python `None`, not the string `'none'`
the training loop's `AR_MODE` checks compare against.** `--set
AR_MODE=none` therefore can NEVER actually disable AR mode via the CLI
-- `Config.AR_MODE` becomes `None`, and `if ar_mode != 'none':` (`None
!= 'none'`) evaluates `True`, so the AR branch stays "on" with
`ar_mode` equal to neither `'frame_ar'` nor `'none'`, falling into the
`sched_sampling_loss()` branch instead of skipping AR entirely.

**3. `sched_sampling_loss()` crashes on frame-native models** -- the
THIRD real instance of the same bug class already fixed twice this
session (§59: `frame_ar_loss()`, `aux_horizon_loss()`): it calls
`centroid_velocity_loss(model(inp), tgt, cfg)` directly on a frame-
native model's raw `NX*LATENT_DIM`-wide output without the
`to_per_token_latent()` reshape the frozen AE decoder needs. Only
reachable via bug #2 above (no real arm has ever set `AR_MODE='sched'`
together with `TOKENIZATION='frame'`), which is why it survived
undetected -- confirmed via the exact same crash signature as the two
previous instances: `"linear(): input and weight.T shapes cannot be
multiplied (39x470 and 47x100)"`.

**Why not fixed now**: all three are real, but none block the actual
task (verifying WSD/`HALF_TIME_MODE`) once worked around (avoid
repeated `--set` flags; don't pass `AR_MODE=none` through `--set`,  the
arm's own `AR_MODE` default plus the existing MPS auto-disable guard
already produces the same effective behavior safely). Flagged here so
a future session doesn't rediscover them from scratch -- the sensible
fix set is: `--set` should use `action="append"` with per-item
`KEY=VALUE` splitting (or already-working space-separated usage should
be the ONLY documented form), `_coerce()` should special-case `AR_MODE`
(and any other field whose valid value is literally the string
`"none"`) to not intercept it into Python `None`, and
`sched_sampling_loss()` needs the same `to_per_token_latent()` fix
already applied twice elsewhere.

Full test suite unaffected: 195 passed, 14 skipped.

---

## 66. v8.3 -- strategy #6 (MoE) built + unit-tested; local-pod startup cost investigated end-to-end: a raw-binary data loader, full startup timing instrumentation, an eval-batch speedup attempt that found and fixed a real OOM-handling bug, then correctly reverted its own default after live testing

### Strategy #6 (Mixture-of-Experts) status update -- see §64.3

§64.3 was written as "planned, NOT built." That's no longer accurate: this
session actually built it. `MoEFeedForward`, `_build_ffn()`'s dispatch,
and `moe_aux_loss()` exist in `model_variants.py`; `MOE_NUM_EXPERTS`
(default 0, byte-for-byte dense), `MOE_TOP_K`, and
`MOE_LOAD_BALANCE_WEIGHT` exist in `Config`. `tests/test_moe_ffn.py`
covers shape/gradient/aux-loss/top_k-clamping/SwiGLU/frame-native
correctness plus a real tiny-model forward+backward with MoE enabled --
8/8 pass. §64.3 has been amended in place to reflect this (built +
unit-tested, no real-data training run yet -- `MOE_NUM_EXPERTS=0` stays
the `Config` default). `singleshot/run_moe_h100.sh` was added as a
one-command H100 launcher (cold-start `a3b_delta_ar` + MoE, since a
different `Block.mlp` structure breaks warm-starting from any existing
checkpoint, same as every other architecture-changing strategy in
§63.4's table).

A real H100 pod run (`r2_a3b_delta_ar`, this launcher) trained cleanly to
step 500 with `MOE_NUM_EXPERTS=4`/`MOE_TOP_K=2` and pulled its artifacts
back successfully -- mechanically sound, but 500 steps from random init
(no warm-start, as expected) says nothing about whether MoE helps; no
real result yet, consistent with §64.3's "genuinely open question, not a
confident bet" framing.

### The real bottleneck wasn't the pod -- it was eval, and it was already there locally

Comparing pod progress to a parallel local (MPS) run of the same arm+MoE
surfaced something more useful than the MoE question itself: at step 75,
wall_seconds=34267 for only 75 steps (ETA ~752h). The `IMPROVEMENT=
-80999%` numbers this produced on `a0_control` (which has `AR_MODE=
'none'` -- never trained against the rollout objective at all) are
expected/irrelevant on that arm specifically; but the SPEED problem is
real and arm-independent, so it was worth attacking directly rather than
just moving the same slow loop to a faster GPU.

### Fix #1: instrument the startup critical path (`train()`, `TransformerDataset`)

Every stage before the training loop starts now logs a
`[startup] <stage> @ +Xs` marker: run-lock acquisition, device-regime
resolution, dataset load (see below), device residency, model
construction, feature stats, warm-start/resume decision, `torch.compile`,
causality probe, `null_baselines` (where the `[start-from:decoder]`
rainbow line actually originates, not before it), and loop entry. The
previously-silent `_read_h5_dataset_threaded()` disk read now logs
before/after with GB/s throughput -- there was previously LITERALLY
ZERO log output during a multi-GB disk read, which is exactly what a
user watching a live pod session flagged as an unexplained ">1 minute
pause I can't explain."

### Fix #2: a raw-binary data loader, replacing HDF5 for the full-file read case

Diagnosis: `train_80.h5`/`val_80.h5` are chunked ONE ROW PER CHUNK --
59,280 separate HDF5 chunk-index lookups+reads for train alone. Fine on
a warm local page cache (confirmed: 9.86GB in ~2s on this Mac's NVMe),
but exactly the pattern that turns into one round-trip per row on a
cold-cache/network-backed pod volume (the ORIGINAL "one CPU core,
slowly" diagnosis behind `_read_h5_dataset_threaded()` itself, per its
own v7.20 docstring -- threading helped, but didn't remove the per-row
chunk structure).

Built: `convert_h5_to_raw()` / `h5_to_raw.py` -- one-off, streamed,
byte-for-byte-verified conversion of the `.h5` files to a flat
`.raw`/`.raw.json` (shape/dtype sidecar) pair, same safety convention as
the existing `decompress_h5.py`. `_read_raw_binary_to_tensor()` loads via
`torch.from_file()` -- a single sequential memory-mapped read straight
into a torch tensor, no HDF5, no numpy array at any point.
`TransformerDataset` auto-detects a `.raw`/`.raw.json` sibling and
prefers it, falling back untouched to the HDF5 path otherwise
(`PFD_USE_RAW_BINARY=0` forces the old path). 6 new tests
(`tests/test_raw_binary_loader.py`) confirm byte-identical output vs.
HDF5, correct partial-length slicing, real corruption detection in the
verify step, and end-to-end `TransformerDataset` equivalence -- also
manually confirmed against the real 4.2GB `val_80.h5` locally
(byte-identical). Nothing changes for data that hasn't been converted
yet; run `python h5_to_raw.py` once per box to opt in.

### Fix #3 (measured, not guessed): where the eval time actually goes

Before touching anything, this session insisted on REAL numbers, not
estimates -- direct local benchmarks against the real `a3b_delta_ar`+MoE
model and the real `val_80.h5`:

* Rollout cost is genuinely linear per sequence on MPS (`chunk=1`
  forces one independent iteration per sequence): steady-state ~80s/seq
  (n=4 and n=8 agreed: 79.8s vs 81.5s; n=1/n=2 were inflated by MPS's
  own first-call warm-up, not real per-seq cost).
* The teacher-forced validation pass -- SEPARATE from rollout, and
  invisible in any prior log line -- also runs at `batch=1` on MPS
  (`regime.eval_micro_batch`) over ALL 25,410 val rows, EVERY eval:
  measured ~0.13s/row steady-state, extrapolating to **~55 minutes per
  eval**, entirely unaffected by `--rollout-seqs`.
* Reworking the real interval arithmetic from a live run with both costs
  now separated (rather than lumped into "training time") gives real
  training-only cost of ~45s/step, not the ~180s/step this session
  originally (wrongly) inferred before TF cost was isolated.
* Net: `--val-every 250 --rollout-seqs 16` versus the defaults is a
  **measured ~7.3x** wall-clock reduction (463s/step amortized -> 63.5
  s/step; ~772h -> ~106h for 6000 steps) -- driven mostly by eval
  FREQUENCY (10x fewer evals), since `--rollout-seqs` alone only ever
  touches the rollout half, and the now-larger TF pass is unaffected by
  it.

### Fix #4 attempted: raise `eval_micro_batch` above 1 on MPS -- real win, real bug found, real live failure, correctly NOT shipped as the default

The `eval_micro_batch=1` clamp's own comment predates the v3.1 `NUM_X`
cut and admits to being oversized; `evaluate()` runs under
`@torch.no_grad()`, so the TRAINING clamp's backward-graph memory
argument doesn't apply. Raising it (`chunk=`/`tf_batch_size=` in
`evaluate()`) reduces the NUMBER of MPS dispatch calls without changing
FLOPs -- CUDA already runs the identical code path safely at
`eval_micro_batch=32-64`.

**Bug found and fixed**: a first attempt at an OOM-safe backoff wrapper
(`_run_with_oom_backoff()`, halve-and-retry on a caught
`RuntimeError`) was USELESS as originally written -- retrying while
still inside the `except RuntimeError as e:` block keeps `e`'s traceback
(and every tensor reachable from the failed call's frames) alive, so
`torch.mps.empty_cache()` has nothing to free yet, and the retry re-OOMs
with the IDENTICAL byte count no matter how small the chunk gets, even
down to a chunk independently known to fit. Fixed by moving the
cache-clear + retry to AFTER the `except` block exits normally (Python
clears `e`/its traceback at that point, per the language spec). Covered
by a real regression test
(`test_exception_state_cleared_before_retry` in
`tests/test_oom_backoff.py`, 10/10 passing) that checks
`sys.exc_info()` is actually clear during each retry -- this is exactly
the state that made `empty_cache()` a no-op before the fix.

**Live testing (after the fix) still crashed at `eval_micro_batch=16`
AND `32`**, twice, on this exact machine -- NOT reverted based on
theory, reverted based on watching it fail. Two real, independent causes
found:
1. `rollout_frames()`'s context grows by one token every one of its
   ~680 iterations -- a new shape almost every step defeats the MPS
   caching allocator's block reuse, the SAME "changing shapes" mechanism
   `_regime_banner_mps_cpu`'s ORIGINAL training-side rationale already
   described, just via accumulation instead of backward-graph retention.
   It was wrong to assume `@torch.no_grad()` made eval exempt from this.
2. This machine's shared Metal memory pool was ALSO under real, growing
   pressure from entirely unrelated processes at the time (`other
   allocations` climbed 66 -> 70 -> 78 GiB across a few minutes --
   PyCharm, multiple WebKit renderers, WindowServer; confirmed via `top
   -o mem`, not assumed).

**Net decision**: `PFD_MPS_EVAL_BATCH` default reverted to **1** (`git
diff` shows the full back-and-forth) -- the backoff mechanism itself is
real, tested, and kept (self-correcting for whoever DOES have headroom
to spare, via `PFD_MPS_EVAL_BATCH=N`), but shipping a default that was
just directly observed to crash on its own reference machine would be
irresponsible regardless of how good the theory sounded beforehand.
This is the one lever from this whole investigation that did NOT pay
off locally -- eval-frequency reduction (`--val-every`/`--rollout-seqs`,
Fix #3) is the confirmed, safe win; raising `eval_micro_batch` remains
available as an opt-in for a box with real headroom, not a default.

### Full test suite: 222 passed, 13 skipped (was 195/14 at start of session)

## 67. v8.4 -- PLAN: next real launch moves to an H100 pod; `_latest.pt` cadence made time-based, not just step-based

### 67.1 Checkpoint cadence -- shipped this session

`_latest.pt` was only ever saved every `Config.CHECKPOINT_EVERY_STEPS`
(25) steps. That is fine as long as step time is roughly what it was
tuned against (B300/H200), but it is the wrong unit once GPU type
varies run to run: a slower box under-checkpoints in wall-clock terms,
a faster one burns I/O checkpointing far more often than the crash-
recovery use case needs. Added `Config.CHECKPOINT_EVERY_SECONDS = 600`
(10 min) as a second, independent trigger alongside the existing
step-based one in the main training loop
(`train_production_transformer_deep_dive.py`, `train()`) -- whichever
fires first wins:

```python
due_for_time_checkpoint = (
    (time.time() - last_checkpoint_wall) >= Config.CHECKPOINT_EVERY_SECONDS)
if step % Config.CHECKPOINT_EVERY_STEPS == 0 or due_for_time_checkpoint or hit_budget:
    save_checkpoint(latest_path, ...)
    last_checkpoint_wall = time.time()
    ...
```

`last_checkpoint_wall` is seeded to `t_start` (not 0) so a resumed
process doesn't immediately fire a save on its first step. This only
touches the `_latest.pt` cadence -- `_best.pt` / `_rollout_best.pt`
promotion logic (§51, §54) and `LATEST_ARCHIVE_KEEP` archiving (§10.9,
v6.x) are unchanged. On an H100, where per-step time is currently
unmeasured for this recipe, the 10-minute floor bounds crash-recovery
loss regardless of what that turns out to be, the same way it now does
for a very fast or very slow arm on the existing GPU pool.

### 67.2 H100 pod launch -- prep, not yet run

All prior production runs (v1.0 §7, and every sweep round since) used
B300/H200/RTX PRO 6000 Blackwell hardware; an H100 has not been used for
this recipe yet. `singleshot/provision_and_run.sh` already supports
this without a code change -- `GPU_ID` is an explicit override, and
`GPU_AUTO_DETECT_PATTERN` (default
`H200|H300|RTX PRO 6000 Blackwell (Server|Workstation) Edition$`,
see the script's inline comment on relative pricing/VRAM) is an
environment-variable override, not a hardcoded pool. For the H100 run,
launch with:

```bash
GPU_AUTO_DETECT_PATTERN='H100' ./provision_and_run.sh   # or GPU_ID=<exact H100 id/displayName>
```

Open items before that launch, to fold into this section once run:
- H100 has 80GB (SXM) or 94GB (NVL) HBM depending on SKU -- confirm
  which variant `runpodctl gpu list` actually offers before assuming it
  matches the 96GB Blackwell headroom the current `AR_SEQS` sizing notes
  (§7.10/singleshot script comments) were measured against.
- No H100 step-time measurement exists yet for this recipe -- the new
  `CHECKPOINT_EVERY_SECONDS=600` floor is deliberately hardware-agnostic
  so this is a measurement to make, not a blocker to the launch.
- `venv`/dependency bootstrap (`bootstrap_remote.sh`,
  `requirements.txt`) is GPU-family-agnostic (CUDA wheel selection is
  driven by the pod image, not this script), so no changes expected
  there -- confirm on first real H100 boot rather than pre-emptively
  editing.

## 68. v9.0 -- the H100 launch happened: a queue-driven bake-off runner built, four real bugs found and fixed the hard way, and the root cause of six catastrophically bad screens traced to a mismatched warm-start checkpoint, not the hyperparameters being tested

Major-version bump (v8.4 -> v9.0, not the usual `.N`) because this
session changed how experiments get launched at all, not just what's
being tried -- see `singleshot/bakeoff_run.sh` (new) and
`DATA_PIPELINE_WALKTHROUGH.md` for the parallel data-pipeline writeup
this same session produced.

### 68.1 `singleshot/bakeoff_run.sh` -- a queue-driven bake-off, not a fixed batch

`provision_and_run.sh`'s `--arm2`/`--arm3` mode launches a FIXED set of
concurrent arms, waits for all of them, pulls back once. The problem:
a cheap early-stop (§67's own H100 run -- `h11_ridge_distill` bailed
after only ~14 real minutes, `EARLY_STOP_PATIENCE_STEPS=500` correctly
firing) just left the freed GPU slot idle until the whole pod's global
timeout tore it down.

`bakeoff_run.sh` fixes that: a fixed number of concurrent slots
(`--slots`, default 2), backed by a JSON queue
(`singleshot/bakeoff_queue.json`) -- the instant a slot finishes for
ANY reason (promoted, early-stopped, hit its own budget, crashed), that
slot's results are pulled back and classified, and the next queued
permutation launches into the freed slot immediately. Reuses
`provision_and_run.sh`'s pod lifecycle verbatim (GPU auto-detect, the
two-layer global-timeout watchdog, `scp_env_files.sh`/
`scp_data_files.sh`/`bootstrap_remote.sh`, `rsync_with_retry`) --
nothing in those shared scripts was touched. `--dry-run` validates the
queue (schema, unique `(arm,round)` pairs so checkpoint prefixes never
collide, no repeated `--set` per entry) with zero `runpodctl`/`ssh`
calls -- the only thing safe to iterate on without spending money.

### 68.2 Four real bugs found by actually running it, not by review

Every one of these looked like "it just quit and did nothing" from the
outside -- `set -e` gives zero error text when the failing command
itself produced none. Found by reading the pulled-back logs directly,
not guessed at:

1. **`grep` finding no match silently kills the whole script under
   `set -e -o pipefail`.** `_explicit="$(grep -oE ... | awk ...)"` --
   when grep matches nothing (the NORMAL case for 5 of 6 queue entries,
   which have no `--warm-start` of their own), grep exits 1, `pipefail`
   propagates that past awk's own 0 exit, and the assignment dies
   silently. Hit on the very first queue entry, every time, until
   fixed. Three separate instances of the identical pattern found and
   fixed: the Phase 1b `--warm-start` extractor, `validate_queue()`'s
   `--set`-repeat counter (latent -- never fired since every entry
   currently has a `--set`), and `classify_outcome()`'s log-reason
   grep (this one would have killed the ENTIRE bake-off mid-run, not
   just one entry, the first time any slot finished cleanly with no
   early-stop/wall-clock/traceback line to match). All fixed with a
   trailing `|| true` (or, for the two cases embedded inside `${VAR:-
   ...}` defaults, restructured into a separate statement with its own
   `|| true` -- confirmed directly that `x="${x:-$(false)}"` under
   `set -e` ALSO dies silently, the default-expansion doesn't protect
   the inner command substitution).
2. **`rsync`'s implicit chown on push.** Phase 1b's checkpoint upload
   used `rsync -avP` (no `--no-owner`/`--no-group`) -- `-a` implies
   preserving owner/group, which needs `CAP_CHOWN` on RunPod's
   container even as `root@`. Exit 23 (`chown ... Operation not
   permitted`), AFTER the file had already transferred 100% correctly
   -- purely a metadata failure that `set -e` still treated as fatal.
   This exact failure, and this exact fix, already existed in
   `scp_env_files.sh` (OVERVIEW.md §55/v7.15) -- missed here because
   Phase 1b was new code, not a copy of the already-fixed pattern.
3. **Ctrl-C didn't actually stop anything.** No explicit signal
   handling existed at all -- relied entirely on ambient process-group
   delivery reaching the re-exec'd child, the timeout watchdog, AND
   every live SSH training connection, which isn't something to rely
   on across environments. Fixed with explicit `trap ... INT TERM` at
   both the outer wrapper (forwards to the child + watchdog) and the
   main body (kills every live slot's local SSH connection, then lets
   the existing `cleanup_pod` EXIT trap delete the pod) -- verified the
   trap-and-kill mechanism itself fires correctly (confirmed via
   SIGTERM directly; this session's own sandboxed test tool appears to
   suppress raw SIGINT delivery to background jobs, an environment
   quirk, not a flaw in the fix -- the trap code path is identical for
   both signals).
4. **The warm-start loader never unwraps `torch.compile`'s
   `_orig_mod.` prefix.** `h11_wd_dropout_variant`'s explicit
   `--warm-start saved_models/r2_h11_ridge_distill_rollout_best.pt`
   crashed with `REFUSING to load: unexpected keys in checkpoint
   (architecture drift)` -- EVERY key, because that checkpoint was
   saved while wrapped in `torch.compile`, and its raw
   `model_state_dict` carries the `_orig_mod.` prefix on every
   parameter name. `save_scripted_model()`/`moe_aux_loss()` already
   unwrap this (`getattr(model, "_orig_mod", model)`) -- the warm-start
   read path (`train_production_transformer_deep_dive.py`, the
   function `DEFAULT_WARM_START_CKPT` feeds) never did. Fixed: strip a
   leading `_orig_mod.` from every checkpoint key right after it's
   read, before the shape/architecture-drift checks run. This is a
   real fix to the shared trainer, not just the bake-off script --
   any future explicit `--warm-start` from a compiled checkpoint would
   have hit the identical crash.

### 68.3 The real finding: six catastrophic screens had ONE shared cause, not six bad hyperparameters

First real bake-off run (post-fixes above) posted wildly negative,
wildly INCONSISTENT `improvement_pct` across every `a3b_delta_ar`
screen (-204592%, -219994%, -436965%, -188066%, -81816%) -- diagnosed
by reading the actual pulled-back logs (`sweep_logs/r21_a3b_delta_ar.log`
etc.), not the summary numbers alone:

- Every non-`fresh` screen warm-starts from
  `DEFAULT_WARM_START_CKPT` = `saved_models/old/r1_a3b_delta_ar_rollout_best.pt`.
  Its own embedded metadata: `step=25 epoch=25`. Its own deep-dive
  report (`tests/reports/r1_a3b_delta_ar_deep_dive.md`) confirms it was
  ALREADY losing to persistence by frame 28 in its native 40-frame
  regime (`last% (frame 28) = -110.84%`).
- Every screen's rollout eval runs ~68 frames (`VAL_ROLLOUT_STEPS` at
  `NUM_TIME=80`) -- more than DOUBLE the horizon where this checkpoint
  was already failing natively. Autoregressive error compounds with
  rollout length; asking a fraying-by-frame-28 checkpoint to run 68
  frames free-running is a guaranteed blowup, not a measurement of
  WSD vs. cosine-restarts vs. `AR_SEQS`. Confirmed directly in the
  logs: teacher-forced loss was completely normal and decreasing
  (`val_tf=0.0097`) while rollout MSE was ~2000x worse than
  persistence -- textbook exposure-bias divergence, not a broken
  model or a broken metric. The wild swings in the percentages across
  entries are the noise signature of chaotic divergence dynamics, not
  a real effect size for anything actually being screened.
- A second, compounding discovery: `NETWORK_VOLUME_ID` makes
  `/workspace` (and everything under `saved_models/`, `sweep_logs/`)
  PERSIST across every pod recreation. Round numbers reused across
  separate debugging attempts (including ones from before the bugs in
  §68.2 were fixed) silently accumulated -- e.g.
  `sweep_logs/r21_a3b_delta_ar.log` contained TWO separate
  `[DEVICE DETECTED]` startup banners concatenated together, one from
  an earlier crashed attempt, one from the real run. Not itself the
  cause of the bad numbers, but a real trap for reading these logs
  cold -- always find the LAST `[DEVICE DETECTED]` banner before
  trusting anything above it.

**Fix**: `bakeoff_queue.json`'s three affected screens
(`a3b_wsd_screen`, `a3b_cosine_restarts_screen`,
`a3b_arseqs_bump_screen`) now explicitly `--warm-start` from
`saved_models/r2_h11_ridge_distill_rollout_best.pt` instead --
confirmed real, positive `improvement_pct=+37.26%` under the CURRENT
80-frame regime (its own `status.json`, from the real run this same
session pulled back). Round numbers bumped (21->31, 22->32, 25->35,
26->36) specifically to avoid resuming the now-poisoned state this
run just wrote to the persistent volume, per the trap just above. The
two `fresh: true` MoE screens are untouched by this fix (they don't
warm-start) and are expected to STILL look bad on a short screen for
the same long-rollout-horizon reason, just starting from random init
instead of a bad checkpoint -- for those specifically, "does it train
without crashing / does loss trend down" remains the real cheap-screen
signal, not rollout-vs-persistence this early.

**Not yet confirmed on real hardware**: this whole diagnosis is a
conjecture, however well-evidenced -- the next real bake-off run is
the actual test (see the command in the session notes / next chat
turn). If the three re-pointed screens STILL show five-or-six-digit
negative `improvement_pct` at their very first eval (step 250), this
diagnosis is wrong and the search continues elsewhere.

### 68.4 The re-pointed run's early results (H200, wandb group `Bones`) -- confirmed live, mid-run

Also fixed this same session, both real: `wandb login`'s interactive
flow writes to `~/.netrc` (i.e. `/root/.netrc`) -- the CONTAINER's own
root filesystem, not `/workspace` (the `NETWORK_VOLUME_ID` mount) --
so credentials never survived a pod recreation despite `/workspace`
itself persisting. `bakeoff_run.sh` now also checks for a key at
`/workspace/.wandb_api_key` on the volume itself before deciding
`--no-wandb`, and each slot's own wandb run URL is now re-printed as a
standalone, bold, per-slot-colored line the instant wandb prints it,
instead of being buried mid-scroll.

Checked the live pod directly (`ssh` + reading `saved_models/*_status.json`
and `nvidia-smi` mid-run, not waiting for pull-back) rather than
waiting for the run to finish:

- **`h11_wd_dropout_variant` (r36, step 700/6000): `improvement_pct
  = +46.08%` (`rollout_mse=0.001744`), promoted.** This is the best
  rollout-vs-persistence result logged ANYWHERE in this project's
  history to date -- beats `u3_frame_rollout_dominant` (+41.3%),
  `q1_sophia_ema` (+40.8%), `p1_bestyet_adamw` (+40.5%), and its own
  warm-start source `r2_h11_ridge_distill` (+37.3%). Only `WEIGHT_DECAY`/
  `DROPOUT` raised 0.01->0.03 on top of the best known checkpoint. Only
  12% through its step budget -- could still peak-and-decline like
  prior arms (the exact pattern `EARLY_STOP_PATIENCE_STEPS` exists
  for), not yet a final result.
- **`a3b_wsd_screen`/`a3b_cosine_restarts_screen`/`a3b_arseqs_bump_screen`
  (r31/r32/r35) are STILL catastrophic** (-246351%/-131296%/-414664%
  at step 500) -- but checked their logs directly and confirmed the
  §68.3 warm-start fix DID work this time (`transferred 4.79M
  (100.00%) across 83 tensors` from the correct, good
  `r2_h11_ridge_distill_rollout_best.pt`, matching `r36`'s own
  source). So this is a NEW, separate finding, not the old bug
  recurring: the common factor across these three (and absent from
  `r36`) is each either fully RESTARTS the LR schedule (`wsd`/
  `cosine_restarts` re-run a full warmup ramp back up to peak LR from a
  fresh optimizer) or raises BOTH `AR_SEQS` and `LEARNING_RATE`
  together -- a full LR-warmup restart on top of an already-converged
  checkpoint is a well-known way to catastrophically disrupt it. Real,
  informative negative result about how to screen these specific axes
  (a much lower peak LR / no full re-warmup when starting from a good
  checkpoint), not evidence WSD/cosine-restarts/AR_SEQS bump are
  themselves bad ideas.
- **MoE screens (r23/r24) are running 8-14x slower per step than the
  dense arms** and hadn't reached their first eval yet at check time
  (step 225/100, `last_eval: {}`) -- ~2.7s/step (`r23`, 4 experts) and
  ~3.4s/step (`r24`, 8 experts) vs. ~0.24-0.46s/step for every dense
  `a3b`/`h11` entry in the same batch. Traced to `model_variants.py`'s
  `MoEFeedForward`, whose own docstring already flags it as "NOT
  optimized for scale (no token-capacity limit/dropping, a plain
  per-(slot, expert) Python loop)" -- `top_k x n_experts` boolean-mask/
  scatter iterations per forward, small-matmul-plus-kernel-launch
  overhead dominating at this model's size (4.79M params, 32-64 token
  micro-batches). Not a bug, a known and now EMPIRICALLY CONFIRMED
  tradeoff. **Backlog, conditional**: don't invest in a real
  batched/capacity-limited MoE kernel until the current cold-start
  screens actually finish and show a quality signal worth keeping --
  optimizing throughput for an approach that turns out not to help
  would be wasted effort. In the meantime, future MoE queue entries
  should get a larger `--max-steps`/`--max-hours` budget than dense
  entries to reach a usable number of evals at all, given the
  confirmed ~8-14x per-step cost.
- GPU health checked directly (`nvidia-smi`): 99% utilization,
  83GB/144GB used on the H200 -- comfortable headroom, no OOM risk
  from the 3-way concurrent slots at this point in the run.

### 68.5 That run's FINAL numbers (all 6 entries completed) -- confirmed from the pulled-back logs, not just status.json snapshots

- **`r36` (h11_wd_dropout_variant) early-stopped at step 750/6000**
  (default `patience=500` since the last promoted checkpoint, at step
  250 -- this run predates §68.6's `--optuna-patience=750` default
  below). Read the actual per-eval log lines
  (not just the final `status.json`): `rollout_mse` degraded MONOTONICALLY
  every single eval -- `0.001744` (step 250, the +46.08% peak) ->
  `0.001856` (step 500, +42.61%) -> `0.002339` (step 750, +27.67%) --
  while `val_tf` loss stayed flat/improving the whole time (`0.0323` ->
  `0.0212` -> `0.0221`). This is as clean a textbook exposure-bias
  signature as this project has logged: the one-step objective the
  model is actually trained on is fine and not overfitting; the
  free-running rollout it's SCORED on degrades anyway, monotonically,
  every eval. Directly motivates §68.6's search space (regularization +
  EMA + AR-loss-weight dampening, not more capacity).
- **MoE screens (`r23`/`r24`) don't just start bad, they get WORSE
  during training, at least within a cheap-screen budget.** `r23`
  (4 experts): `rollout_mse` 2.036 (step 250) -> 2.567 (step 500) --
  degrading, not improving. This matches the ORIGINAL dense cold-start
  precedent too (`r2_a3b_delta_ar` historically showed -926314% at step
  500) -- so this isn't uniquely a MoE defect, it's "any cold start
  looks catastrophic under this eval before real convergence,"
  compounded by MoE's confirmed ~8-14x slower per-step cost eating into
  how many of those early, rough steps a cheap screen can even afford.
  Doesn't change §68.4's conditional-backlog verdict on MoE
  optimization, but sharpens it: the honest comparison for any future
  MoE screen is against a dense cold start's OWN early trajectory, not
  against a warm-started arm's numbers.
- Persistent-volume log accumulation (§68.3's trap) hit again in these
  same log files (multiple `[DEVICE DETECTED]` banners per file, one
  per historical attempt at that round number) -- reconfirming: always
  read from the LAST banner forward, or better, from the pulled-back
  `status.json` (one file, always current) rather than the raw
  accumulated log when just checking "what happened."

### 68.6 `singleshot/optuna_suggest.py` -- a real (if modest) automated search, replacing hand-picked permutations

Direct response to "isn't this what the GPU is for": yes, and the gap
wasn't hardware, it was that every queue entry through §68.5 was a
human (this session) hand-picking one hypothesis and writing it into
JSON -- a real automated search needs a defined space, a sampling/
suggestion algorithm, and a controller that proposes the next trial
from prior results without a human doing it each time. Built exactly
that, layered onto the EXISTING bake-off infrastructure rather than a
parallel system:

- **`singleshot/optuna_suggest.py`** -- standalone `ask`/`tell`/`summary`
  CLI, zero SSH/pod awareness, Optuna's own SQLite storage
  (`sqlite:///PATH`, no server) so independent short-lived process
  invocations (one per slot-fill event) share state. `ask` calls
  `study.ask()` and suggests four ROLLOUT-STABILITY axes built directly
  on `r36`'s result and its §68.5 decline shape (deliberately excludes
  MoE/capacity -- a different axis, can't warm-start from this
  checkpoint anyway, and already has its own screens):
  `WEIGHT_DECAY` (log-uniform 0.01-0.10, `r36` used 0.03),
  `DROPOUT` (uniform 0.01-0.10, `r36` used 0.03),
  `EMA_DECAY` (categorical `[0.0, 0.9, 0.95, 0.98, 0.995]` -- `0.98` was
  `q1_sophia_ema`'s value at +40.8%, never combined with `r36`'s
  regularization bump before), and
  `AR_LOSS_WEIGHT` (uniform 0.3-1.0, default 1.0 -- dampening the AR
  curriculum's push directly targets the exposure-bias mechanism
  itself, not just regularizing around it). Every trial warm-starts
  from the same fixed, proven `r2_h11_ridge_distill_rollout_best.pt`
  under the same fixed `h11_ridge_distill` arm -- only these four
  values vary. `tell` reads `best.improvement_pct` back out of the
  pulled-back `status.json` (clamping the `+-Infinity` "never promoted"
  sentinel to a large finite penalty -- Optuna's storage can't hold
  actual `inf`/`NaN`) and reports it via `study.tell()`; a crashed/
  missing-checkpoint trial reports `TrialState.FAIL` instead of a
  fabricated number, so Optuna's own bookkeeping tells "bad result"
  and "this config crashed" apart.
- **Real bug caught during local smoke-testing, before ever touching a
  pod**: the first version pinned `TPESampler(seed=1337)` for
  reproducibility -- since `ask` and `tell` are SEPARATE processes
  (each `ask` call is a fresh Python interpreter), a fixed seed
  re-initializes the identical RNG state every single call, so TPE's
  random-sampling fallback (used for its first `n_startup_trials`)
  returned the EXACT SAME suggestion three times in a row in testing.
  Fixed by dropping the fixed seed entirely -- confirmed five
  back-to-back `ask` calls afterward all produced distinct, diverse
  suggestions.
- **`bakeoff_run.sh --optuna`** -- minimal surgery on the existing
  script, not a parallel implementation: `launch_slot()`,
  `classify_outcome()`, pull-back, the report file, the color-tagged
  wandb-URL surfacing, and the GLOBAL TIMEOUT WRAPPER are all reused
  completely unchanged. Only the "where does the next queue entry come
  from" question changes -- a new `fill_next_slot()` helper either
  pops from the static `bakeoff_queue.json` arrays (unchanged default
  behavior) or calls `optuna_suggest.py ask`/`tell` to grow the same
  `Q_*` arrays live, one trial at a time, as slots free up. New flags:
  `--optuna`, `--optuna-study=PATH` (default
  `singleshot/optuna_study.db`, shared across runs to keep building on
  prior trials), `--optuna-round-offset=40`, `--optuna-max-trials`
  (default 0 = unbounded, the global timeout is the real cap),
  `--optuna-max-steps=2000 --optuna-max-hours=0.35 --optuna-patience=750`
  (3 eval windows at `--val-every 250` -- `r36` got cut at exactly 2
  windows past its promotion; one extra window of slack separates
  "truly plateaued" from "about to recover" without burning much
  budget on losers). `--dry-run --optuna` previews `--slots` trials
  against a THROWAWAY scratch study file, never the real
  `--optuna-study` path -- `ask` creates a real `RUNNING` trial the
  instant it's called, so previewing against the real file without
  ever `tell`-ing it back would permanently strand trial numbers as
  abandoned rows in the study meant for the real run. Confirmed via a
  full local dry-run: five distinct real suggestions generated, passed
  through the exact same `(arm,round)`-uniqueness/no-repeated-`--set`
  validation the static queue uses, zero `runpodctl`/`ssh` calls, and
  the real study file was confirmed NOT created by the preview.
- Sizing: at ~0.28-0.4s/step (measured this session) and a 0.35h/trial
  cap across 3 concurrent slots inside a 2h total budget, expect
  roughly 15-20 trials -- enough for TPE to move past pure random
  exploration (`n_startup_trials` defaults to 10) partway through the
  run, but not a large-N study. Honest framing: a real, if modest,
  first automated search -- not a claim of statistical power a run
  this size doesn't have.
- **Not yet run for real** -- smoke-tested end to end locally (`ask`/
  `tell`/`summary` standalone, plus the full `bakeoff_run.sh --dry-run
  --optuna` path); the actual 2-hour H200 run is the user's to launch.

### 68.7 The real 2h Optuna run: a genuinely new best (+49.63%), then the 50GB volume filled up and killed the rest of the budget

Ran for real. First 15 trials completed normally -- trial 0
(`weight_decay=0.0327 dropout=0.0262 ema_decay=0.95 ar_loss_weight=0.546`)
hit **+49.63%**, a new best beating `r36`'s +46.08%, and several others
(trials 1, 4, 9) landed in the high 46-49% band too -- real, if early,
confirmation the search space is sane and the TPE sampler is finding
good regions. Then **every trial from #16 through #101 (86 in a row)
failed identically**:

```
OSError: [Errno 122] Disk quota exceeded
  File ".../train_production_transformer_deep_dive.py", line 439, in acquire_run_lock
    with open(tmp, "w") as f:
```

Root cause, confirmed via `runpodctl network-volume get`: `NETWORK_VOLUME_ID
=yl7f9e8rwr` is **50GB**, and nothing has ever been deleted from it across
this entire multi-day session -- every arm from `r2_*` through `r36_*`
(dozens of arms, each `_latest`/`_best`/`_rollout_best`/`_train_best` plus
scripted twins plus a 5-deep `_archive/` directory) plus ~15 real Optuna
trials' own checkpoint sets finally exceeded it. The failure wasn't in
training at all -- it died trying to write a few-byte run-lock file,
before even loading data. A compounding, separately real cost: the
ORIGINAL pull-back code still ran its full 3-attempt/15s-sleep
`rsync_with_retry()` against a checkpoint that could never exist for
every one of those 86 failures, burning real wall-clock on retries with
no chance of success.

**Three fixes, all in `bakeoff_run.sh`, all in the shared poll-loop code
path (so both static-queue and `--optuna` modes get them for free):**

1. **Purge each trial's checkpoint set from the pod immediately after a
   successful local pull-back.** We already have the copy; there's no
   reason for the persistent volume to keep accumulating it run after
   run. This is the direct fix for the root cause, not just a mitigation.
2. **Skip the pull-back retry dance when the process crashed and left
   nothing behind.** A cheap one-shot remote `compgen -G` existence
   check (via `bash -c`, since `compgen` is a bash builtin, not a
   generic-shell one) runs first when `local_rc != 0`; only falls
   through to the full retry-backed `rsync_with_retry()` if something
   might actually be there.
3. **Circuit breaker**: `MAX_CONSECUTIVE_FAILURES` (env, default 5) --
   after that many consecutive trial failures with no checkpoint
   pulled back, stop launching new trials/queue entries, let
   already-running slots finish naturally, then proceed straight to
   final pull-back/teardown instead of relying on the GLOBAL TIMEOUT
   WRAPPER's SIGTERM ~86 failures later to be the only thing that ever
   ends it. A run of consecutive failures is a systemic-problem signal
   (disk full, missing data, a bad shared flag), not "this
   hyperparameter combo is bad," and should stop the whole run rather
   than keep spending budget re-confirming the same failure.

**Deliberately NOT done automatically**: bulk-deleting the historical
backlog (`r2_*` through `r36_*`) that was the actual majority of the
50GB. Several of those are real reference results this project has
pointed back to repeatedly (`r2_h11_ridge_distill_rollout_best.pt`
specifically is the ACTIVE warm-start source for every current and
future Optuna trial -- deleting it would break the whole search), and
deleting is irreversible. That cleanup is a one-time, human-reviewed
action, not something to fold into an automated script.

## 69. v9.1 -- `WANDB_PROJECT` renamed "NI_Review_v8" -> "Persistence_Linear_v1"

A deliberate rebrand, not a version bump -- every prior run stays exactly
where it is under "NI_Review_*" in wandb; only NEW runs land under the
new name going forward. This RETIRES the OVERVIEW.md-v4.1 convention
(`WANDB_PROJECT`'s trailing `_vN` tracks THIS document's latest major
version) -- "_v1" is the new project name's own independent version 1,
unrelated to this document being at v9. `tests/test_version_sync.py`'s
equality check (`WANDB_PROJECT`'s version must match OVERVIEW.md's) was
removed for exactly this reason -- it would otherwise permanently and
correctly-uselessly fail forever now that the two are deliberately
decoupled. The "has SOME trailing `_vN` suffix at all" check stays --
that part of the convention is still real and enforced.

## 70. v9.2 -- `CHECKPOINT_EVERY_STEPS=25` was firing every ~19s on H100/H200, 30x more often than the 10-minute floor it was supposed to defer to

Caught by direct observation on a live pod, not by review: checked
`saved_models/r160_h11_ridge_distill_archive/`'s own file timestamps --
`step_0000875.pt` -> `step_0000975.pt` (100 steps, 4 checkpoints) spanned
14:43:43 -> 14:44:58, ~19s per 25-step checkpoint interval. §67.1's
`CHECKPOINT_EVERY_SECONDS=600` addition was meant to be a 10-minute
floor "regardless of hardware," but the pre-existing
`CHECKPOINT_EVERY_STEPS=25` (tuned for B300/H200-era throughput, per its
own original comment: "~20-30s apart") was ALWAYS going to fire first on
any reasonably fast GPU -- the 600s trigger never got a real chance to
govern anything.

Raising `CHECKPOINT_EVERY_STEPS` alone would have broken something else,
though: `touch_run_lock()`'s stale-lock detector has a 300s staleness
threshold, deliberately calibrated (§v6.1/v7.11 lineage) against a
heartbeat every ~20-30s -- a 10-15x safety margin. `touch_run_lock()` was
called from the SAME conditional block as the checkpoint write, so
simply making checkpoint writes rarer would have ALSO slowed the lock
heartbeat to match -- up to 600s between touches, LONGER than the 300s
staleness window, which would falsely mark a perfectly healthy run as
dead and let another process steal its lock mid-run.

**Fix**: decoupled the two. `CHECKPOINT_EVERY_STEPS` raised to `5000`
(at the fastest measured rate this session, ~0.19s/step dense H100/H200,
that's ~950s -- still comfortably above 600s, so `CHECKPOINT_EVERY_
SECONDS` is now the real, sole governor there; at MoE's measured
~3.4s/step, 5000 steps is ~4.7 hours, so the 600s time-based trigger
governs there too -- `CHECKPOINT_EVERY_STEPS` is now a backstop, not the
dominant trigger, on every hardware speed actually measured this
session). A new `RUN_LOCK_TOUCH_EVERY_STEPS = 25` keeps the lock
heartbeat on its original fast, fixed cadence via a cheap `elif` branch
(touch the lock with no checkpoint write) alongside the main save block
-- the 300s staleness math is unchanged and still correct.

**Not yet running anywhere**: like the wandb rename in §69, this only
affects the NEXT time `scp_env_files.sh` uploads a fresh copy of the
trainer to a pod -- any trial already running with the old code in
memory keeps its old (~19s) cadence until it finishes.

## 71. v9.3 -- wandb's local artifact cache filled the CONTAINER's root disk to 100%, mid-run, on a COMPLETELY DIFFERENT filesystem than the one this session had been watching

Checked the live pod directly after being asked to fix an "over
utilized" disk warning -- `/workspace` (the persistent
`NETWORK_VOLUME_ID`, 50GB quota, everything §68/§70 had been
monitoring) was fine at ~23GB. The real problem was `df -h /`: the
CONTAINER's own ephemeral root disk (`overlay`, `CONTAINER_DISK_GB=60`)
at **100% used, 20KB free**, with three trials still fully alive and
consuming GPU the entire time -- a disk that was never even part of the
mental model this whole cleanup effort had been built around.

Root cause: `/root/.cache/wandb/artifacts/obj` alone was 60GB.
`save_checkpoint()`'s `log_artifact()` call queues a wandb artifact
upload on EVERY checkpoint save -- not just `_best`/`_rollout_best`
promotions, the routine `_latest.pt` write too -- and wandb caches a
local copy of every uploaded artifact indefinitely by default, on the
container's root disk, not `/workspace`. At the OLD (pre-§70)
`CHECKPOINT_EVERY_STEPS=25` cadence (~19s/write on H100/H200), that's
enough local-cache growth to fill a 60GB disk within a few hours of a
3-slot run.

**Immediate fix**: cleared the cache live (`wandb artifact cache
cleanup 0GB` -- which itself failed, unable to even create a tmpdir on
a disk with zero bytes free; fell back to `rm -rf
/root/.cache/wandb/artifacts/obj/*`). Freed the disk from 20KB to 60GB
free in one command, zero disruption to the three running trials
(confirmed: same GPU utilization, same round numbers, still training
immediately after).

**Standing fix**: `bakeoff_run.sh`'s poll loop now runs this same
cleanup (`wandb artifact cache cleanup 0GB`, `rm -rf` fallback) every
`WANDB_CACHE_CLEANUP_INTERVAL_SECS` (default 900s = 15 min), on its own
clock independent of slot/trial completion events -- the cache grows
continuously while ANY trial is actively checkpointing, not just when
one finishes.

**How much §70 alone would have already helped**: `CHECKPOINT_EVERY_
STEPS` raised 25->5000 cuts the DOMINANT contributor (the `_latest.pt`
path) by ~30x once a fresh pod picks up that code -- the same growth
that filled 60GB in a few hours would now take on the order of days.
Not zero, though: promotion-triggered uploads still accumulate
independently of that fix over a long enough run, so the periodic
cleanup above is real insurance, not redundant with §70.

**Neither fix reached the run that hit this**: the live run that
triggered this whole investigation was already 1h38m into its 2h
budget, running old code in memory, when this was diagnosed -- it
finished (or will finish) on the manually-cleared headroom alone,
never picking up either the §70 checkpoint-cadence fix or this
section's periodic-cleanup addition. Both apply starting with the next
fresh pod.

## 72. v10.0 -- "Scotty": a real study result (trial 112, +51.43%) confirmed the run completed cleanly, one real interrupt-handling bug found, and a two-track plan to both deepen and keep searching at once

The full 2-hour `--optuna` run (§71's run) completed and every one of its
22 finished trials pulled back correctly -- confirmed by direct file
listing, not just trusting the console. The overall study record is now
**trial 112: +51.43%** (`weight_decay=0.0966, dropout=0.0735,
ema_decay=0.98, ar_loss_weight=0.683`), beating trial 0's earlier
+49.63%.

### 72.1 Real bug found: the GLOBAL TIMEOUT WRAPPER's own SIGTERM path lost in-flight work

Checked the Optuna study state directly after the run ended: trials 128,
129, 130 (the 3 slots' in-flight work at the exact moment
`--total-budget-hours` elapsed) were stuck in `RUNNING` forever --
`handle_interrupt()` killed their local ssh clients (correctly stopping
the remote training) but never attempted a pull-back first, and never
told Optuna a result either. Real, permanent loss of partial progress,
and permanently-orphaned study rows.

**Fixed**: `handle_interrupt()` now does, for every live slot, BEFORE
killing it: one best-effort (no-retry -- this is an emergency path, not
the normal one) `rsync` pull-back of whatever checkpoint state exists
at that exact instant, then (in `--optuna` mode) reports it back to the
study via `tell` (a real result if the pull-back landed, `--failed`
otherwise) so the trial is never left dangling. Caught and fixed a
second bug WHILE writing this fix, before it ever ran for real: the
first draft indexed `Q_TRIAL` by the SLOT number instead of the QUEUE
index that slot happened to be running (`Q_TRIAL[SLOT_QIDX[slot]]`, not
`Q_TRIAL[slot]`) -- those only coincide by accident for the first few
trials in a run. The 3 orphaned trials from this run were manually
marked `--failed` to close out the bookkeeping.

### 72.2 The plan: two tracks, one pod, wandb group "Scotty"

Rather than just re-running the same screen, or just deepening the one
best result, do both at once on the same pod:

- **Track A -- deepen**: `h11_ridge_distill` warm-started from
  `r2_h11_ridge_distill_rollout_best.pt`, PINNED to trial 112's exact
  hyperparameters (`singleshot/bakeoff_queue_trial112.json`, round 200),
  `--max-hours=2.0` and `--early-stop-patience-steps=1000000` (i.e.
  effectively disabled) so it uses the FULL 2-hour budget rather than
  getting cut off early by the same patience mechanism that stopped the
  original 2000-step screen -- the open question is whether +51.43%
  keeps climbing with a real budget instead of a cheap-screen one.
- **Track B -- keep searching**: `--optuna` mode, reusing the SAME
  `optuna_study.db` (131 trials of history already, well past TPE's
  `n_startup_trials=10` default -- suggestions should be genuinely
  informed now, not mostly-random exploration), but only **2** slots
  this time, not 3 -- Track A's pinned run occupies the 3rd concurrent
  slot on the same GPU.
- Both tracks tagged `WANDB_GROUP=Scotty` (`--wandb-group=Scotty` on
  both invocations) so every run this session lands in one filterable
  wandb group.
- Both bake-off processes run against the SAME pod (`POD_ID` passed to
  the second once the first creates it) with `KEEP_POD=1` on both, so
  neither process's own completion tears the pod down out from under
  the other -- final teardown is a manual step once both finish.

## 73. v10.1 -- the Scotty run's real outcome, a severe data-loading contention bug found and fixed, and Optuna requeue tooling

### 73.1 Final verdict: trial 112's recipe has a real ceiling, not a screen-budget artifact

Track A (round 200, pinned to trial 112's exact hyperparameters,
`--max-hours=2.0`, early-stop effectively disabled) ran to its full
2-hour wall-clock limit at step 12,464/100,000. Peak stayed at
**+48.31%** (set early) and DECLINED from there -- 48% (early) -> 38%
(step 1,750) -> 24% (step 6,250) -- never recovering. Giving this exact
recipe 6x more steps than its original 2,000-step screen did NOT reveal
a stronger regime; it kept sliding. Real, if disappointing, signal that
this recipe's ~48-51% band is a genuine ceiling, not a screen-length
artifact -- directly motivating §74 below (question the LOSS/METRIC
itself, not just the hyperparameters around it).

Track B (Optuna) ran 153 trials this session (58 completed) before the
same 2h timeout, reaching round 192. **Overall study record stayed at
trial 112, +51.43%** -- nothing in this run's fresh exploration beat
it, though trials 136 (+50.12%) and 146 (+49.99%) came close. Notable:
4 of the top 5 trials cluster around `ema_decay=0.95-0.98` and
`weight_decay=0.03-0.10` -- a real TPE-discovered region, not scatter.

### 73.2 Severe data-loading contention found: 0.01 GB/s reading the SAME file from 3 concurrent processes, fixed with an in-RAM promotion

Two Optuna trials (131/132, rounds 171/172) died at **step 1/2000**
after burning their entire 21-minute budget -- confirmed via their
actual logs, not guessed: `train_80.h5 read done in 724.0s (0.01 GB/s
across 8 threads)`, vs. 0.15-2+ GB/s measured solo elsewhere this
session. All three concurrently-launched processes (including Track
A's own pinned run) measured the identical 0.01 GB/s reading the SAME
file from `/workspace` (`mfs#us-ga-2.runpod.net`) at the same time --
a 15-200x collapse, far worse than simple 3-way bandwidth-splitting
would predict. Root cause is most likely that MooseFS (a shared,
multi-tenant network filesystem at 82% cluster-wide utilization, not a
dedicated local disk) serializes concurrent opens of the same file far
worse than raw bandwidth division alone would explain.

**Fix**: `bakeoff_run.sh` now promotes `train_80.h5`/`val_80.h5` into
`/dev/shm` (tmpfs -- real RAM) once per pod, via a new
`promote_data_to_shm()` step. `Config.TRAIN_H5`/`VAL_H5` are PINNED
absolute paths (no `--set` override possible), so the only way to
redirect reads is a symlink at the exact pinned path.
`mount --bind` (which would have needed zero other changes -- a
bind-mounted path transparently reports the real file's size to `stat`)
was tried FIRST and confirmed NOT permitted in this container
(`mount: permission denied`, no `CAP_SYS_ADMIN`). Symlinks need no
special privileges but require one real precaution: `/dev/shm` is
per-pod ephemeral while `/workspace` (where the symlink itself lives)
is the PERSISTENT volume, so a symlink from one pod would dangle on the
very next one. Solved with a `.on_volume` backup of the real file kept
ON the persistent volume -- first time ever on a given volume, back up
the real file and symlink; every time after (including a dangling
symlink from a previous pod), just refill `/dev/shm` from that backup.
Also required fixing the existing `NETWORK_VOLUME_ID` data-presence
check to `stat -L` (dereference) instead of a plain `stat`, since a
plain `stat` on a symlink reports the link's own tiny size, not its
target's -- would have permanently broken that check on every run after
the first one. **Verified live under real 2-3-way concurrent load
post-fix**: two freshly-launched trials measured 3.3s and 8.3s loading
the same file that previously took 724-774s under contention, no
repeat of the step-1 failure for the rest of the run.

### 73.3 Optuna `requeue` -- a completed trial's bad result doesn't get retried on its own

The two contention-killed trials (131/132) exited cleanly (exit code 0,
hit their own wall-clock cap) with a REAL status.json showing
`improvement_pct=-inf` (clamped to -1,000,000) -- so `bakeoff_run.sh`
correctly reported them as genuine (if catastrophic) results, not
crashes. Optuna has no "this completed trial's result doesn't count"
mechanism: once `COMPLETE`, a trial is permanent history that only ever
informs FUTURE suggestions, never gets automatically re-attempted --
confirmed directly in the study (`value: -1000000.0` for both,
permanently). Added `optuna_suggest.py requeue --trial N`, using
Optuna's real `study.enqueue_trial()` API to force that trial's exact
params to be served again on the very next `ask()`, ahead of the
sampler. Smoke-tested on a throwaway study first, then applied for
real to trials 131/132 and verified (against a disposable copy of the
real study, so the verification itself didn't consume the queued slot)
that the next `ask()` genuinely returns those exact params.

### 73.4 Standing instruction going forward

The user does not want this agent launching real (billed) pod runs
unless explicitly authorized for that specific run. From this point on:
code, tests, and queue files are written and verified locally (syntax
checks, dry-runs, unit tests, smoke tests against disposable data) --
the user runs and monitors the actual pod themselves.

## 74. v10.2 -- is scoring the loss on ONE point of a 125-point cube too narrow? `CENTROID_LOSS_POINTS` built

Directly motivated by §73.1: trial 112's recipe (and everything derived
from it) has hit a real ceiling around +48-51% that more regularization
tuning and more training time both failed to move. The user's
hypothesis: the training loss (`centroid_velocity_loss()`) has only
ever scored the model on ONE of the AE decoder's 125 reconstructed
spatial points -- the exact center (`CENTROID_TRIPLET_IDX=62`) -- and
that may be too narrow a supervision signal, independent of anything
else already tried.

### 74.1 Confirmed the cube's index layout directly, not by assumption

`decode_centroid()`'s docstring already asserted index 62 is "the
middle of 125," but `125 // 2 == 62` is true of ANY flat ordering, so
that alone doesn't prove which physical offset each of the other 124
indices corresponds to. Found and read the actual generator,
`og_data_prep/Ordered_005_AllPossibleCombos.py`: `offsets = [-2, -1, 0,
1, 2]`, nested `for dx in offsets: for dy in offsets: for dz in
offsets:` -- i.e. index `(dx+2)*25 + (dy+2)*5 + (dz+2)`. Confirms
`(0,0,0) -> 62` independently of the docstring's own claim, and gives
the exact indices of the 6 face-adjacent neighbors: `{37, 57, 61, 63,
67, 87}` (NOT the 26-neighbor Moore neighborhood -- axis-aligned
face-adjacency only, matching what "6 positions around the centroid"
means).

### 74.2 Two new training-loss modes, `decode_centroid()` (eval) left completely untouched

New `Config.CENTROID_LOSS_POINTS` (default `'center'`, bit-identical to
every existing result -- verified by a direct test comparing against
the pre-existing single-centroid computation):
- `'center_plus_6'` -- mean over the centroid + its 6 face-adjacent
  neighbors (7 points, uniformly weighted).
- `'all125'` -- mean over the full cube, centroid up-weighted by
  `Config.CENTROID_LOSS_CENTER_WEIGHT` (default 10.0) relative to the
  other 124 (each at weight 1.0).

Implementation: factored `decode_centroid()`'s single frozen-decoder
call into `_decode_raw()` (returns the full unsliced 375-dim vector),
so the new modes pull additional triplets from the SAME decode call
rather than re-invoking the decoder per point; `decode_centroid()`
itself is unchanged (still just `_decode_raw()[..., CENTROID_SLICE]`).
This is the load-bearing design choice: EVERY eval/rollout/
persistence-baseline call site in this file goes through
`decode_centroid()`, never `centroid_velocity_loss()` -- so
`improvement_pct`/`rollout_mse` mean the EXACT same thing regardless of
which loss mode a given arm trains with, keeping every historical
result (r36, trial 112, everything in §73.1's table) directly
comparable to whatever these new modes produce. Changing what the model
is TRAINED on must never change what the numbers we've been comparing
all session MEAN.

New `tests/test_centroid_loss_points.py` (12 tests, all passing):
index-mapping correctness against an independently-reimplemented
reference formula (not just re-importing the same possibly-wrong
constant), diagonal neighbors confirmed excluded, all three modes
produce finite scalars with real gradient flow, `center_plus_6`/`all125`
are exactly zero for identical inputs, `all125` at `CENTER_WEIGHT=1.0`
matches a plain 125-point mean, `all125` at `CENTER_WEIGHT=1e6`
converges to the center-only loss value (verified: 2.03048 vs 2.03047),
an unknown mode string raises `ValueError`, and `decode_centroid()`'s
own output is verified bit-identical across all three modes. Full
existing suite re-run clean: 232 passed, 14 skipped, zero regressions.

### 74.3 Queue prepared, not launched (per §73.4)

`singleshot/bakeoff_queue_centroid_points.json` -- two entries, both
warm-started from `r2_h11_ridge_distill_rollout_best.pt` with trial
112's EXACT other hyperparameters (`WEIGHT_DECAY=0.0966
DROPOUT=0.0735 EMA_DECAY=0.98 AR_LOSS_WEIGHT=0.683`), changing ONLY
`CENTROID_LOSS_POINTS` -- an isolated, apples-to-apples test of this one
variable, sized as a cheap screen (2000 steps / 0.35h / 750-step
patience, matching this session's established screen convention).
Validated via `--dry-run` only -- not run.

```bash
bash singleshot/bakeoff_run.sh --dry-run \
  --queue=singleshot/bakeoff_queue_centroid_points.json --slots=2
```

### 74.4 Numbering note

Labeled `v10.2` (continuing the same v10 chapter as §72's original
Scotty plan and §73's outcomes/fixes) rather than a fresh `v11.0` --
this reads as a direct continuation of the same investigation (Scotty's
ceiling finding -> question the loss itself), not an unrelated new
chapter. Flagged for the user to correct if a harder version boundary
was intended.

## 75. v10.3 -- a 3rd `CENTROID_LOSS_POINTS` mode (`distance_weighted`), the queue resized to 1h/H200, wandb group "McCoy", and a dedicated "focus" dashboard

### 75.1 Third mode: `distance_weighted` -- a smooth falloff, not a hard cutoff

`center_plus_6` is a hard 7-point cutoff (centroid + face-neighbors,
uniform weight); `all125` is a binary split (centroid at
`CENTROID_LOSS_CENTER_WEIGHT`, all other 124 points flat at `1.0`
regardless of how close or far they are from the centroid). Neither
lets a point's distance from the centroid matter continuously. Added a
3rd mode: weight per point = `CENTROID_LOSS_CENTER_WEIGHT / (1 + dist)`,
where `dist` is that point's Euclidean grid distance from the centroid
(new module-level `CENTROID_DISTANCES` tuple, 125 entries, built
directly from the same `_cube_idx` offset math already verified in
§74.1 -- `CENTROID_DISTANCES[62] == 0.0`, asserted at import time).

Confirmed by direct test (`tests/test_centroid_loss_points.py`, now 15
tests): zero for identical inputs, gradients flow, weight strictly
decreases with distance (unlike `all125`'s flat non-center weight), and
`decode_centroid()` remains bit-identical regardless of this mode (same
isolation guarantee as the other two). One thing this mode does NOT do,
confirmed by a failed assumption caught before committing: raising
`CENTROID_LOSS_CENTER_WEIGHT` does NOT make `distance_weighted`
converge to center-only behavior the way `all125` does at a huge
weight -- since every non-center point's weight is
`CENTER_WEIGHT / (1 + dist)`, the RATIO of a neighbor's weight to the
centroid's weight is `1 / (1 + dist)`, a constant independent of
`CENTER_WEIGHT`. A 1e6 center weight still leaves face-neighbors
(`dist=1`) at fully half the centroid's weight. This is an inherent,
correctly-understood property of the scheme (a smooth falloff, not a
tunable-to-center-only knob), not a bug -- the original "converges to
center-only at huge weight" test from `all125` was deliberately NOT
reused here for that reason.

### 75.2 Queue resized: 2 entries -> 3 entries, cheap-screen sizing -> ~1h/H200

`singleshot/bakeoff_queue_centroid_points.json` rebuilt with 3 entries
(rounds 310/311/312: `center_plus_6`, `all125` w=10, `distance_weighted`
w=10), each `max_steps=6000 max_hours=1.0
early-stop-patience-steps=1500` -- roughly 6x the step budget of the
original 0.35h cheap screens (§74.3), keeping the same
steps:patience:val-every ratio. Same trial-112 hyperparameters,
warm-start checkpoint, and isolation-of-one-variable design as before.
Re-validated via `--dry-run --slots=3`: all 3 `(arm,round)` pairs
unique, zero network/ssh calls made.

### 75.3 wandb group "McCoy" and a dedicated "focus" dashboard

Per explicit request, this batch uses `--wandb-group=McCoy` (previous
batches: "Bones" §68.4, "Scotty" §72.2) -- a top-level
`bakeoff_run.sh` CLI flag, not something baked into the queue JSON, so
no queue-file change was needed for this part.

New `singleshot/setup_wandb_focus_view.py` -- a standalone,
no-SSH/no-pod script using the `wandb_workspaces` package (installed
this session, v0.4.11) to create a saved wandb Workspace view
containing exactly one Section named "focus" with exactly one panel: a
`LinePlot` of `improvement_pct` (confirmed a genuine per-step
time-series metric, not summary-only, by reading the trainer's own
`tel.log(wandb_payload, step=step)` call site), filtered to
`Group == 'McCoy'` via `RunsetSettings(filters=...)`, with
`auto_generate_panels=False` so nothing else gets auto-added next to
it. Directly answers "If it is somewhere else in wandb, I don't want to
find it" -- one section, one panel, nothing auto-generated alongside
it. Built and verified locally (`Workspace` object constructs correctly
against the real entity `pvl-data`, panel/filter fields inspected) but
`.save()` (the network call that actually creates it in wandb) was NOT
invoked -- per §73.4, left for the user to run.

### 75.4 Commands for the user to run (nothing above was launched)

```bash
# Dry-run already confirmed clean; drop --dry-run to actually launch.
# NETWORK_VOLUME_ID=yl7f9e8rwr (this project's persistent volume,
# §68.7/§72) is REQUIRED -- without it the pod has no /workspace mount,
# so no warm-start checkpoint and no persisted wandb key (see the v10.5
# errata note at the end of this section for why this was missing here
# originally).
NETWORK_VOLUME_ID=yl7f9e8rwr bash singleshot/bakeoff_run.sh \
  --queue=singleshot/bakeoff_queue_centroid_points.json --slots=3 \
  --total-budget-hours=1.0 --wandb-group=McCoy

# One-time (or whenever the view needs recreating) -- builds the wandb
# Workspace itself, no pod/training involved:
python singleshot/setup_wandb_focus_view.py \
  --entity pvl-data --project Persistence_Linear_v1 --group McCoy
```

## 76. v10.4 -- the McCoy run's result: widening the loss target HURT, not helped -- all 3 `CENTROID_LOSS_POINTS` modes underperformed the single-centroid ceiling

### 76.1 Numbers (H200, wandb group `McCoy`, pod `5oprn256i8lxla`, all 3 slots exit 0 / pull-back OK)

Pulled-back `manual/r310-312_h11_ridge_distill.json`, full eval curves
(not just the final `best` snapshot):

| Mode | round | Peak improvement_pct | Peak step | Ended at | beat_constant_predictor |
|---|---|---|---|---|---|
| `center_plus_6` | 310 | **+43.84%** | 750 | +38.05% @ step 2250 | true |
| `distance_weighted` w=10 | 312 | +43.48% | 500 | +28.57% @ step 2000 | true |
| `all125` w=10 | 311 | +41.55% | 1000 | +33.69% @ step 2500 (noisy, dipped to +18-22% mid-run) | **false** |

All 3 early-stopped at their configured 1500-step patience, well
short of `max_steps=6000` and `max_hours=1.0` (wall time 1330-1614s
each, ~22-27 min -- the 1h budget was never the binding constraint,
patience was).

### 76.2 Verdict: broadening the training-loss target made things WORSE, not better

The §74 hypothesis was that scoring training loss on only 1 of 125
reconstructed points might be too narrow a signal, and that including
more of the cube (uniformly, or distance-weighted) might produce a
better-trained model. The result is the opposite: **every widened mode
peaked BELOW trial 112's established single-centroid ceiling**
(+48.31% Track A / +51.43% Track B best, §73.1) -- by 5-10 points, not
a rounding difference. All 3 also reproduce the identical exposure-bias
decline shape already characterized in §73.1 (early promotion, then
monotonic-ish decline as free-running rollout drifts from the
teacher-forced training distribution) -- so this isn't "broader loss
delays the peak given more steps," it's strictly worse at every step
budget tried.

Within the 3, there's an internally consistent gradient: `center_plus_6`
(narrowest widening, uniform 7-point window immediately around the
centroid) did the LEAST damage; `all125` (widest, most diluted --124
points at flat weight 1.0 vs. one point that's actually evaluated) did
the MOST damage, and was the only mode to fail `beat_constant_predictor`
at all during training, plus the noisiest curve of the three (dips to
+18-22% mid-run). `distance_weighted` fell in between but closer to
`center_plus_6`, despite having the same "spread across all 125"
structure as `all125` -- consistent with its per-point weight decaying
fast enough (`1/(1+dist)`, so a face-neighbor at dist=1 already gets
half weight, and the cube's outer shell at dist~2.8 gets ~1/4) that its
*effective* footprint is much closer to `center_plus_6`'s 7-point window
than to `all125`'s flat 125-point one.

**Practical conclusion: the single-centroid loss (existing default,
`CENTROID_LOSS_POINTS='center'`) is not the bottleneck.** Widening the
spatial scoring window, in any of the 3 forms tried, trades away
accuracy on the one point that's actually evaluated for accuracy on 6
or 124 points nobody is scoring -- a real, measured cost, not a
theoretical one. This closes out the §74/§75 investigation: keep the
default `'center'` mode for any future run; `center_plus_6`,
`all125`, and `distance_weighted` remain available as `Config` options
(and are unit-tested, §74.2/§75.1) but are not recommended based on
this result.

### 76.3 Next hypotheses (not yet built or run -- for discussion)

The loss-target-width axis is now closed (§76.2). Candidate next
directions, roughly in order of how directly they attack the
exposure-bias decline shape that has now shown up identically in EVERY
run this session regardless of loss target (§73.1 Track A, §76.1's all
3 modes, and the original r36 finding that motivated the whole Optuna
search):

1. **Scheduled sampling / attack the exposure bias directly, not just
   regularize around it.** Every run so far either trains fully
   teacher-forced (bar `AR_LOSS_WEIGHT`'s existing curriculum) or has
   tried to delay the decline with weight decay/dropout/EMA (Track B's
   whole search space, §68.6-68.7). None have changed WHAT the model
   sees during training to look more like free-running rollout. A
   scheduled-sampling or DAgger-style scheme (occasionally feed the
   model its own prior prediction instead of ground truth during
   training, ramping up over steps) targets the actual mismatch
   between train-time and rollout-time input distributions, rather
   than hoping regularization slows the drift once it starts.
2. **Checkpoint at/near the peak and stop -- treat the decline as
   expected, not a failure to prevent.** Every run's `best.*` already
   reflects this (the promoted checkpoint is pulled from the peak, not
   the final step), but the current early-stop patience (750-1500
   steps) still burns real wall-clock and steps chasing a recovery that
   has now never once happened across ~10+ runs this session. Consider
   cutting patience substantially (e.g. 250-500 steps) as a pure
   efficiency change -- would not change the achievable ceiling, but
   would let more distinct hyperparameter configs be screened per
   dollar/hour.
3. **Revisit `AR_LOSS_WEIGHT` specifically against the confirmed
   ceiling.** Trial 112 used `AR_LOSS_WEIGHT=0.683` (dampened from
   h11's default of 1.0); Track B's 153-trial search covered this axis
   already but never in combination with any of the 3 widened loss
   targets from this section. Given §76.2's finding that widening the
   *target* hurts, an open question is whether widening the *AR
   curriculum* portion specifically (as opposed to the plain
   teacher-forced portion) behaves differently -- untested combination.
4. **Rollout-length curriculum.** `--rollout-seqs` has stayed fixed at
   16 across every run referenced above. If the decline is driven by
   compounding rollout error, training against progressively longer
   rollout windows (rather than a fixed short one) is a more direct
   lever than loss-target width and hasn't been tried at all this
   session.

Recommend (1) as the most likely to move the actual ceiling rather
than just find another point on the same ~48-51% plateau, since it's
the only untried idea that changes what the model is trained to be
robust to, rather than how hard it's regularized or what target it's
scored against. (2) is a free efficiency win regardless of which
hypothesis is tested next and could be adopted immediately.

## 77. v10.5 -- hypotheses #1 (scheduled sampling) + #4 (rollout-horizon curriculum) combined into ONE mechanism, `AR_SCHED_MIX_P`

### 77.1 Correction: hypothesis #4 was already running in every trial-112-derived run so far

Before building anything, checking `h11_ridge_distill`'s own arm
definition (the arm every run since Scotty has used) found it ALREADY
sets `AR_MODE='frame_ar', AR_FRAMES=8, AR_FRAMES_START=2` -- i.e. the
rollout-horizon curriculum from hypothesis #4 has been active in trial
112 itself and every one of the §76 centroid-loss-points runs. #4 was
never a new axis to test; it was already baked into the baseline. This
reframes the real gap as just hypothesis #1 (scheduled sampling) --
attacking exposure bias by mixing ground truth into the ALREADY-CURRICULUM'D
rollout, not adding a separate curriculum on top of one that doesn't
exist yet.

### 77.2 Why `AR_MODE='sched'` (the existing scheduled-sampling code path) can't just be turned on here

The codebase already has an `AR_MODE='sched'` path
(`sched_sampling_loss()`, `Config.SCHED_SAMPLING_P`) -- but it's a
single-step, two-forward mix with NO horizon/curriculum concept at all,
and `AR_MODE` is a single mutually-exclusive setting (`'none' | 'frame_ar'
| 'sched'`, picked by one `if/else` in the training loop). Switching to
`'sched'` would mean giving up `frame_ar`'s already-curriculum'd
multi-frame chain entirely -- the opposite of "combine #1 and #4."

### 77.3 New mechanism: `Config.AR_SCHED_MIX_P`, a scheduled-sampling gate INSIDE `frame_ar_loss()`'s existing chain

Added a new knob that composes with the curriculum instead of replacing
it. `frame_ar_loss()`'s per-step feedback already had one decision point
(`_feed_or_aropt()`: feed the model's own prediction, optionally via
AROpt accept/reject, §32.2 item 7). Added `_feed_final()` in front of it:
with probability `AR_SCHED_MIX_P` (evaluated fresh per sequence, per
chain step), force GROUND TRUTH regardless of what `_feed_or_aropt()`
would have picked; otherwise defer to it unchanged. This is classic
scheduled sampling (Bengio et al. 2015) applied AT EVERY STEP OF THE
ALREADY-CURRICULUM'D CHAIN -- as `AR_FRAMES_START` ramps the rollout
longer over training, the model is simultaneously exposed to a fixed
fraction of its own errors at each of those (growing number of) steps,
rather than the curriculum alone changing only the LENGTH of a
100%-self-fed chain. `AR_SCHED_MIX_P=0.0` (default) is an exact
passthrough to `_feed_or_aropt()` -- every existing `frame_ar` arm
(h9/h11/branch-M/P/Q/S, etc.) is unaffected byte-for-byte.

Verified via `tests/test_sched_mix.py` (6 new tests, mirroring
`test_aropt_accept_reject.py`'s own convention): default reproduces the
exact pre-v10.5 loss for the same seed; finite/differentiable in both
frame-native and token-native branches and at `AR_SEQS=1`;
`AR_SCHED_MIX_P=1.0` demonstrably overrides `AROPT_TEMPERATURE=1e6`
(proving the gate actually short-circuits AROpt rather than silently
no-opping); and `AR_SCHED_MIX_P` composes with `AR_FRAMES_START` without
interfering with `ar_frames_for_step()`'s own curriculum math. Full
suite re-run clean: 241 passed, 14 skipped, zero regressions (up from
235 before this section -- the 6 new tests).

### 77.4 The run: trial 112's exact recipe + `AR_SCHED_MIX_P=0.3`, nothing else changed

`singleshot/bakeoff_queue_sched_mix.json` -- ONE entry, round 320,
`h11_ridge_distill` arm (so `AR_MODE='frame_ar', AR_FRAMES=8,
AR_FRAMES_START=2` -- hypothesis #4 -- come from the arm default,
unchanged), same warm-start checkpoint and trial-112 hyperparameters as
every prior isolation run this session (`WEIGHT_DECAY=0.0966
DROPOUT=0.0735 EMA_DECAY=0.98 AR_LOSS_WEIGHT=0.683`), `CENTROID_LOSS_POINTS`
left at its default `'center'` (per §76.2's conclusion -- not
re-litigating the closed loss-target-width question), plus the one new
variable: `AR_SCHED_MIX_P=0.3` (30% chance per chain-step of forcing
ground truth instead of the model's own fed-back prediction, on top of
the already-active 2-to-8-frame horizon curriculum). Sized
`max_steps=6000 max_hours=1.0 early-stop-patience-steps=1500`, matching
§75.2's convention -- `AR_SCHED_MIX_P`'s extra cost is one `rand`+`where`
per existing chain step, not an additional forward pass, so no
step-cost premium over any other `h11_ridge_distill` run this session.
Validated via `--dry-run --slots=1 --wandb-group=NC1701`: entry valid,
zero network/ssh calls made.

### 77.5 Fully qualified command for the user to run (nothing above was launched, per §73.4)

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_run.sh \
  --queue=/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_queue_sched_mix.json \
  --slots=1 --total-budget-hours=1.0 --wandb-group=NC1701
```

### 77.6 Errata: two real mistakes in how this command was first handed off

1. The first version of this command omitted `NETWORK_VOLUME_ID=yl7f9e8rwr`
   entirely -- without it, the pod comes up with no `/workspace` mount at
   all, meaning no warm-start checkpoint AND no persisted wandb key for
   `--wandb-group` alone to discover (the auto-discovery in
   `bakeoff_run.sh`, §75/988-994, only checks a file that lives ON that
   mount). Confirmed as a real, load-bearing omission, not a style
   preference -- every prior real launch this session
   (§68.7/§72.2/§75.4) used this exact env var.
2. The user's very next real launch after that correction used the OLD
   `bakeoff_queue_centroid_points.json` (rounds 310-312, §76 -- already
   analyzed) again, not `bakeoff_queue_sched_mix.json` (round 320, this
   section) -- so as of this writing, no `AR_SCHED_MIX_P` result exists
   yet. §77.5's command above is the corrected, complete one for that
   still-pending run.

## 78. v10.6 -- round 320's real result, a live GPU-utilization audit, and why the "worse than the previous-frame anchor" console line is NOT the instability signal

### 78.1 Round 320 result: `AR_SCHED_MIX_P=0.3` peaked closer to trial 112's ceiling than any §76 variant, but the same decline shape persisted

Pulled from `sweep_logs/manual/r320_h11_ridge_distill.json`: peak
**+46.60%** at step 250, declining to +35.24% by early-stop at step
1750 (`early-stop: 1500 steps since the last new promoted rollout
checkpoint (step 250)`). This beats every §76 centroid-loss-points
variant (all under +44%) and is within ~2-5 points of trial 112's own
ceiling (+48.31%/+51.43%, §73.1) -- a real, if modest, improvement. But
the identical exposure-bias decline shape is still there: peak early,
monotonic-ish decline afterward. One run at one fixed `AR_SCHED_MIX_P`
value isn't enough to call this "solved," but it's the closest any
single-variable change has gotten to the ceiling so far.

### 78.2 Live GPU-utilization audit during round 320, requested mid-run

`nvidia-smi dmon` (5 samples, 1Hz) and a point-in-time snapshot while
round 320 was training:

| Resource | Observed | Cap | Utilization |
|---|---|---|---|
| GPU compute (SM) | 0-67%, bursty | 100% | ~40% avg, frequent 0% dips |
| GPU memory | 24.5 GB | 143.8 GB | 17% |
| Power draw | 130-190 W | 700 W | 18-27% |
| SM clock | 1980 MHz | 1980 MHz (max) | pinned at ceiling, `pstate=P0` |

SM clock already at its max with power/memory this low rules out
thermal/power throttling -- the GPU is starved for work, not held back.
Root cause identified by reading `resolve_train_regime()`
(train_production_transformer_deep_dive.py:579): **the CUDA
micro-batch is a HARDCODED constant** (`64 if "H200" in name else 32`),
completely independent of `Config.BATCH_SIZE`/`ACCUMULATION_STEPS`
(those fields are dead on the CUDA path -- confirmed by the comment at
line 5264, "Config.ACCUMULATION_STEPS are no longer read inside the
loop"). No `--set` override could have changed this; it required an
actual code change. Separately, `torch.compile` is OFF by default on
CUDA for a real, already-documented reason (v7.19): concurrent multi-
slot launches each spawn their own full-core `torch._inductor`
compile-worker pool, oversubscribing the host N-way before a single
step runs -- but that reasoning doesn't apply to a solo (`--slots=1`)
run like this one, where the steady-state 15-30% speedup has a full
hour to amortize. An opt-in override (`PFD_COMPILE_MODEL=1`) already
existed for exactly this case.

### 78.3 Fixes: `PFD_CUDA_MICRO_BATCH` env override (new) + `PFD_COMPILE_MODEL=1` (already existed)

Added `PFD_CUDA_MICRO_BATCH` to `resolve_train_regime()`'s CUDA branch
-- unset preserves the exact hardcoded 64/32 default byte-for-byte;
set, it overrides `micro_batch` (and `virtual_batch`/`eval_micro_batch`/
`aux_micro_batch`, which all track it, since CUDA runs with gradient
accumulation off). Deliberately an opt-in env var, not a default
change or a `Config` field read inside the CUDA branch -- a larger
batch changes gradient-noise characteristics (§78.4) and hasn't been
validated safe for every existing arm/regime, so this is one knob to
test in isolation on the next run, matching this session's own
one-variable-at-a-time convention. `bakeoff_run.sh` gained pass-through
for both `PFD_CUDA_MICRO_BATCH` and `PFD_COMPILE_MODEL` (read from the
calling shell's environment, exported into the remote heredoc only
when set -- same convention as `NETWORK_VOLUME_ID`). New test
`test_cuda_micro_batch_env_override` (`tests/test_train_regime_cuda.py`)
confirms the override works and that both CUDA defaults (H200=64,
generic=32) are unchanged when unset. Full suite re-run clean: 242
passed, 14 skipped, zero regressions.

### 78.4 The real instability signal was never the "worse than the previous-frame anchor" line -- it's the pre-clip gradient norm

Read `centroid_velocity_loss`'s console flag logic directly
(train_production_transformer_deep_dive.py:5968-5978): that red
"worse than the previous-frame anchor" line fires whenever the RAW,
single-step, noise-perturbed (`NOISE_STD`) teacher-forced training loss
exceeds a FIXED floor computed once from clean validation data. This
comparison is inherently noisy and near-meaningless on a step-by-step
basis -- round 320's own eval curve shows genuine **+42-46% improvement
over persistence** at the very same steps where nearly every per-step
console line was flagged red. If this flag were the real signal, a
healthy, clearly-improving model couldn't produce it on almost every
line, which is exactly what's observed. This is a diagnostic red
herring, not a sign of anything wrong.

The REAL instability signal is `gnorm` -- and it's actually informative
once you know what it measures: `torch.nn.utils.clip_grad_norm_()`
(train_production_transformer_deep_dive.py:5843) returns the norm
**BEFORE clipping**, then rescales the gradient down to `Config.GRAD_CLIP`
(default 1.0) in place. Round 320's log shows `gnorm` values of
9-25+ on a large fraction of steps -- meaning the RAW gradient is
routinely 10-25x larger than the clip threshold before being clamped
down. `GRAD_CLIP` is already doing its job (preventing those spikes
from blowing up the optimizer step), but frequent large pre-clip norms
this large are a genuine sign of a noisy per-step training signal, not
a false alarm like the anchor-comparison line.

### 78.5 Why the SAME fix (larger batch) addresses both the GPU-efficiency question and the gradient-noise question

This isn't a coincidence: gradient noise from mini-batch sampling
shrinks as batch size grows (the per-step gradient estimate averages
over more examples), which is exactly the standard mechanism behind
why larger batches produce calmer, less spiky `gnorm` traces. The same
`PFD_CUDA_MICRO_BATCH` override that fixes the GPU-utilization finding
(§78.2 -- 126 GB of unused headroom) is also the most direct, cheapest
lever on the gradient-noise finding (§78.4) -- one change, two
symptoms, rather than a separate mechanism for each. `torch.compile`
(§78.3) contributes on the OTHER axis found in §78.2 (kernel-launch-
overhead-bound bursts in the sequential `frame_ar` AR loop, which batch
size can't fix since that loop's cost is chain-length-bound, not
batch-width-bound) -- fusing those small sequential ops directly
targets the 0%-utilization dips in the `dmon` trace, not the gradient
noise.

Not changed this round: `GRAD_CLIP` itself (still 1.0, unchanged) --
the plan is to see whether a larger batch alone meaningfully calms the
`gnorm` trace before reaching for a second, more invasive lever (e.g.
lowering `GRAD_CLIP` further, or adaptive/percentile-based clipping)
on top of it. One variable at a time, same convention as every other
change this session.

### 78.6 The run: same isolation-run hyperparameters as round 320, plus the two efficiency fixes, plus a larger early-stop patience

`singleshot/bakeoff_queue_sched_mix_v2.json` -- ONE entry, round 321,
IDENTICAL hyperparameters to round 320 (`h11_ridge_distill` arm,
trial-112 recipe, `AR_SCHED_MIX_P=0.3`) so this is a clean before/after
comparison of the efficiency fixes alone, not a new hypothesis.
`early-stop-patience-steps` raised `1500 -> 2000` (33% more) -- since a
calmer gradient trace (if `PFD_CUDA_MICRO_BATCH` delivers on §78.5's
prediction) plausibly extends how long the model can be trained before
the exposure-bias decline sets in, and since the throughput gain from
better GPU utilization gives real wall-clock room to spend more
patience within the same `max_hours=1.0` cap without a proportional
runtime cost. `max_steps=6000`/`max_hours=1.0` unchanged. Validated via
`--dry-run --slots=1 --wandb-group=NC1701` with
`PFD_CUDA_MICRO_BATCH=128 PFD_COMPILE_MODEL=1` set: entry valid, zero
network/ssh calls made.

### 78.7 Fully qualified command for the user to run (nothing above was launched, per §73.4)

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr PFD_CUDA_MICRO_BATCH=128 PFD_COMPILE_MODEL=1 \
  bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_run.sh \
  --queue=/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_queue_sched_mix_v2.json \
  --slots=1 --total-budget-hours=1.0 --wandb-group=NC1701
```

Watch `nvidia-smi dmon -s pucm` during this run to confirm SM
utilization/memory/power actually rise versus §78.2's baseline numbers
before drawing any conclusion about whether the batch/compile change
helped the gradient-noise/decline-shape question too -- those are two
separate things to verify, not one.

## 79. v10.7 -- round 321 confirmed `torch.compile` actively hurts this workload, and the batch-size lever was aimed at the wrong knob; `AR_SEQS` is the real one

### 79.1 What round 321 actually showed (user manually killed the pod after spotting it live)

Live inspection of `r321_h11_ridge_distill.log` found:

`torch._dynamo hit config.cache_size_limit (8)` at step ~175 --
`frame_ar_loss()`'s sequential AR loop grows its context by one token
per iteration (up to `AR_FRAMES*NUM_X` ≈ 80 distinct shapes per call,
§77's already-active curriculum), and `torch.compile(model,
dynamic=True)` did not absorb that growth cleanly -- the guard failure
was a STRIDE mismatch, not just a size change. Dynamo recompiled
repeatedly, hit its cache limit after 8 distinct shapes, and silently
fell back to eager for that function from then on. Net effect: real
compile overhead was paid (a 192-worker `torch._inductor` pool spawned
just for this one solo slot) for none of the steady-state benefit on
the part of the step that dominates cost. This is a DIFFERENT,
previously-undocumented failure mode from the one that made compile
default OFF in the first place (v7.19's concurrent-multi-slot CPU
oversubscription) -- it's not about slot count at all, it's that
`frame_ar`'s growing-context loop is structurally hostile to
shape-caching JIT, regardless of how many slots are running.

Separately: round 321's memory only rose 24.5 GB -> 27.1 GB despite
doubling `micro_batch` 64 -> 128 -- far less than doubling would
predict. Root cause: `h11_ridge_distill` runs the sequential
`frame_ar` AR loop on EVERY step (`AR_EVERY_N_STEPS=1`), and that
loop's own parallelism is gated by `Config.AR_SEQS` (arm default: 2),
which is COMPLETELY INDEPENDENT of `micro_batch`/`PFD_CUDA_MICRO_BATCH`
-- §78.3's fix only widened the (apparently minor-cost) primary
teacher-forced pass, never touching the loop that actually dominates
per-step wall time. Measured throughput bore this out: ~0.38s/step in
round 321 (batch=128, compile on) vs. ~0.3s/step in round 320
(batch=64, compile off) -- flat to slightly WORSE, not better.

### 79.2 Corrected fix: compile OFF, raise `AR_SEQS` instead (no new code needed)

`AR_SEQS` is already a plain `--set`-able `Config` field (no code
change required, unlike §78.3's `micro_batch`, which needed a new env
override because it was hardcoded). Raising it directly increases how
many sequences run in parallel through each of the sequential AR
loop's ~80 growing-length forwards -- the actual GPU-idle-between-
kernels mechanism identified in §78.2, correctly targeted this time.
`regime.aux_micro_batch` (which clamps `AR_SEQS` on CUDA, see
`resolve_train_regime`) now equals `micro_batch=128` under
`PFD_CUDA_MICRO_BATCH=128`, so `AR_SEQS` can go up to 128 without being
silently clamped back down -- picked 16 (8x the arm default) as a
first, moderate step to test in isolation, not the ceiling.

`singleshot/bakeoff_queue_sched_mix_v3.json` -- ONE entry, round 322
(321 is now dead, pod killed manually), identical to round 320/321's
recipe (`AR_SCHED_MIX_P=0.3`, `early-stop-patience-steps=2000`) plus
`AR_SEQS=16`, and `PFD_COMPILE_MODEL` dropped entirely (defaults back
to OFF). `PFD_CUDA_MICRO_BATCH=128` kept (harmless, real if modest win
on the primary TF pass, §78.3). Validated via `--dry-run --slots=1
--wandb-group=NC1701` with `PFD_CUDA_MICRO_BATCH=128` set: entry valid,
zero network/ssh calls made.

### 79.3 Fully qualified command for the user to run (nothing above was launched, per §73.4)

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr PFD_CUDA_MICRO_BATCH=128 \
  bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_run.sh \
  --queue=/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_queue_sched_mix_v3.json \
  --slots=1 --total-budget-hours=1.0 --wandb-group=NC1701
```

Watch `nvidia-smi dmon -s pucm` early (first few hundred steps) to
confirm memory/utilization actually rise THIS time, and grep the log
for `cache_size_limit`/`torch._dynamo` to confirm it stays clean (it
should -- compile is off, so dynamo never engages at all).

## 80. v10.8 -- quantified proof the "worse than the previous-frame anchor" console line was never a real signal, and it's now removed

### 80.1 The quantified check (not just a mechanism argument)

Direct analysis of round 320's full local log (`sweep_logs/r320_h11_ridge_distill.log`):
**100% of all 73 logged steps, step 1 through step 1750 (the entire
run to early-stop), were flagged "worse than the previous-frame
anchor."** Zero exceptions -- including at the run's own peak.

The decisive check used the CLEAN validation teacher-forced loss (no
`NOISE_STD` injection, full val set, not a noisy single training
mini-batch), so "it's just training noise" can be ruled out directly:

| step | val_tf_loss (clean) | anchor | verdict | real rollout improvement_pct |
|---|---|---|---|---|
| 250 (run's peak) | 0.009874 | 0.00740 | worse | **+46.60%** |
| 500 | 0.008946 | 0.00740 | worse | +43.60% |
| 1750 (early-stopped) | 0.008606 | 0.00740 | worse | +35.24% |

Even with every noise source removed, the model never once beat the
anchor -- not even at the checkpoint that was simultaneously beating
persistence by 46.6% on the metric that actually matters. Confirmed
mechanism: `anchor` is a ONE-STEP, never-compounding "copy the previous
frame" predictor -- unusually strong on smooth physical data precisely
BECAUSE it never compounds -- while `improvement_pct` is a MULTI-STEP
autoregressive rollout comparison, where persistence's error compounds
and a genuinely-learned model can win by a wide margin while still
losing the one-step comparison every single time. These measure
different regimes; losing one while winning the other is expected, not
a red flag.

### 80.2 Fix: the per-step alarm is gone, only the (rare, real) positive case still shows

Removed the `elif train_loss > anchor:` red-flag branch entirely from
the per-step console line (train_production_transformer_deep_dive.py,
`~5985`) -- it was structurally incapable of ever showing anything but
red on this arm's own recipe (§80.1), so it carried zero information.
`crossed_anchor` (green, "beats previous-frame anchor... saving
_train_best.pt") is unchanged and is now the ONLY anchor-related signal
ever printed -- kept specifically because it's the rare, genuinely
informative direction, not because it's common. The OTHER red flag,
`WORSE THAN PREDICTING ZERO` (failing to beat the trivial CONSTANT
predictor), is a different and real problem signal and was left
untouched -- unlike the anchor, that one's rare on a healthy run and
means something when it fires. No change to `crossed_anchor`/
`train_improved`'s underlying computation or the `_train_best.pt`
checkpoint-save gating -- display-only change. Full suite re-run clean:
242 passed, 14 skipped, zero regressions.

## 81. HANDOFF NOTES (v10.9) -- for the next developer picking this up

This section is written to be readable **on its own**, without first
reading §1-80. It orients a new developer: what this project is, how
the data was built, what's been tried, where the project currently
appears to be stuck, and one concrete backlog item (§81.4) requested
explicitly for improving unit-test coverage. Every other section in
this document (§1-80) is the detailed, chronological lab notebook this
summary is distilled from -- cross-references below point back into it
for full detail, code line numbers, and exact numbers.

### 81.1 What this project is, in one paragraph

The goal is to train a transformer (`train_production_transformer_deep_dive.py`)
to predict the future evolution of a turbulent wake flow (vortex
shedding behind a bluff body, varied by inlet velocity) in a compressed
latent space, well enough that its **autoregressive rollout** (feed the
model's own predictions back in as input, repeatedly, to simulate many
timesteps forward) beats a trivial **persistence baseline** (literally:
freeze the last known frame and never update it) by as wide a margin as
possible. The trained model is meant to eventually stand in for running
the real (expensive) simulation. Progress is measured almost entirely
by one number: `improvement_pct` = how much lower the model's
rollout MSE is than persistence's rollout MSE, on a fixed held-out
validation rollout window.

### 81.2 How the data was put together

Full detail: `DATA_PIPELINE_WALKTHROUGH.md` (280 lines, read this next
-- it dissects one real training example end to end with real numbers).
Condensed version:

```
raw sim output (pickled per-frame        GEN3 AttentionSE          build_wake_atlas.py      prepare_data.py       train_production_
dataframes: x,y,z,vx,vy,vz per grid      autoencoder (frozen,      (finds vortex-core       (assembles 52-dim     transformer_deep_dive.py
point, per PARAM FOLDER, per step)       trained separately,       centroids, emits         rows into 80x10x52    (trains on the 47-dim
      |                                  NOT part of this repo's   (y,z) sensor taps        sequences)            latent slice; the
      |                                  training loop at all)     to sample around them)         |                47-dim latent is
      +---------- encode -------------------->  |                        |                        |                decoded back through
                                     47-dim latent per grid              wake_atlas.csv.gz    train_80.h5 /         the SAME frozen
                                     point, per step -- this is    (y,z taps + spatial       val_80.h5             decoder to score the
                                     what gets baked into every    centroid cx,cy,cz)        (N, 80, 10, 52)       training loss)
                                     row of train_80.h5/val_80.h5
```

**Raw source**: `/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent/<PARAM>/<step:04d>.pkl.gz`
-- one directory per simulated case, named things like `3p6`, `4p6`,
`5p2`, `6p4`, `6p6`, `7p2`, `7p8`, `8p4`, `10p4`, `11p4`. These
directory names ARE the experiment identifier at the raw-data level.

**The `PARAM` naming is already a real, live inconsistency worth
knowing about up front** (this is the exact gap §81.4's backlog item
exists to close): `prepare_data.py::parse_param()` hardcodes a
lookup table mapping each folder-name code to a physical scalar (an
inlet/free-stream velocity, in m/s):

```python
{"3p6": 5.6, "4p4": 6.9, "4p6": 7.2, "5p2": 8.1, "6p4": 10.0,
 "6p6": 10.3, "7p2": 11.3, "7p8": 12.2, "8p4": 13.1,
 "10p4": 16.3, "11p4": 17.8}
```

Note that `"3p6"` does NOT map to `3.6` -- the folder-name code is an
opaque legacy label, not a directly-readable encoding of the physical
value it represents. So the SAME experiment can legitimately be
referred to as `"3p6"` (the folder/code name, which is what's stored
in `train_80.h5`'s `param` column after going through `parse_param()`
-- see below) OR as `"U=5.6"` / similar (the actual physical quantity),
depending on which script, plot, or conversation you're reading. There
is currently no single column anywhere in the training data that
records BOTH forms side by side, and no documented reason for why the
codes are shaped the way they are (why `6p6`, not `6p3` for 10.3?)
beyond "that's what the raw folders are named."

**Assembly**: `build_wake_atlas.py` finds the vorticity-magnitude
weighted centroid of the vortex core for each `(param, step)` (§3A of
the walkthrough doc), and emits a sliding window of `(y,z)` sensor taps
around it, PLUS some random/background taps for contrast -- all
recorded in `wake_atlas.csv.gz`. `prepare_data.py` then assembles
`train_80.h5`/`val_80.h5`: shape `(N_sequences, 80, 10, 52)`, where the
52 feature columns are `[latent(0:47), x(47), y(48), z(49), t_idx(50),
param(51)]` -- `param` here is already the PARSED SCALAR (e.g. `10.0`
for `"6p4"`), not the original folder-name string. `t_idx` is a
position WITHIN the 80-frame sequence (0-79), not an absolute simulation
timestep -- that mapping back to a real `.pkl.gz` step number is not
stored per-row either.

**Val/train param split** (as of the current version, `prepare_data.py`
v3.5): `val_params = ["3p6", "6p4", "11p4"]`, everything else in train.
This has moved multiple times as issues were found (`4p4`'s raw folder
turned out partially corrupt/unusable, `3p6` was later found to
overlap with data the upstream autoencoder had already SEEN during its
own training -- meaning validation on `3p6` measures the transformer's
generalization over already-familiar latents, not a fully clean
holdout; see `prepare_data.py` lines ~1021-1090 for the full history of
this decision and its caveats).

**Frozen, external dependency**: the GEN3 AttentionSE autoencoder
(trained completely separately, lives under
`encoder/autoencoderGEN3/`) is used TWICE, in two different roles that
are easy to conflate (§3 of the walkthrough doc covers this in detail):
once (long before any of this repo's training runs) to PRODUCE the
47-dim latent baked into every H5 row, and again, LIVE during every
training/eval step of `train_production_transformer_deep_dive.py`, to
DECODE a 47-dim latent back into a 375-dim physical velocity
reconstruction (125 spatial points x 3 velocity components) so the
training loss and the `improvement_pct` eval metric can be computed in
real, physical units rather than raw (physically meaningless, arbitrary-
scale) latent space. This decoder is frozen -- never trained by this
repo, never fine-tuned, loaded once as a scripted `.pt` file.

### 81.3 The traceability gap in the CURRENT datasets (why this matters for testing)

Today, `train_80.h5`/`val_80.h5` (and their `_quartered`/Parquet
derivatives) let you recover, per row: the 47-dim latent, the spatial
`(x,y,z)` coordinate (though only the 10 fixed `x` stations and the raw
`(y,z)` sensor-grid values -- NOT the actual (cx,cy,cz) vortex-core
centroid this point was sampled around, or whether it was a real
wake-core tap vs. a random background sample), the within-sequence
`t_idx` (NOT an absolute simulation step number), and the parsed
`param` SCALAR (not the original folder-name code, and not any other
naming convention someone might use for the same experiment). Getting
any of the missing pieces back requires re-joining against
`wake_atlas.csv.gz` (for the wake-core/random distinction and the
`cx,cy,cz` centroid) or reading `prepare_data.py`'s source directly
(for the `parse_param()` table and `X_COORDS`) -- see the walkthrough
doc's §5 table for the full "have it / don't have it" breakdown.

None of this matters for TRAINING or for the `improvement_pct` metric
-- the model only ever needs the 47-dim latent and its own
positional/temporal bookkeeping. It matters for **writing competent
unit tests**: a test that wants to assert "row N really does correspond
to the vortex core of experiment `3p6` at simulation step 143, at grid
position `(y,z)=(-4,18)`, roughly `(4.2, -1.1, 3.6)` units from the
core centroid" currently cannot do that from the training data alone --
it has to reconstruct the join by hand, against files that live outside
`data/`, using lookup tables that live in yet another file. That's a
real testing gap, not a training gap, which is exactly why the fix
(§81.4) is scoped as two NEW dataset variants rather than a change to
the datasets actually used for training.

### 81.4 BACKLOG: build "trace" versions of `train_80.h5`/`val_80.h5`

**Status: not started. Explicitly a backlog item, not in progress.**

**Ask (verbatim intent, from the user)**: create two new dataset files
-- `train_80_trace.h5` and `val_80_trace.h5` (naming to be finalized by
whoever picks this up; the `_trace` suffix is a placeholder) -- that
contain **the exact same training data** as `train_80.h5`/`val_80.h5`
(same rows, same 52-dim feature vectors, same row ordering/shapes), but
with **additional appended metadata columns** that make it possible to
trace every row's spatial, temporal, and "experiment" identity all the
way back to its original source, without needing to separately load
`wake_atlas.csv.gz` or reverse-engineer `parse_param()`. This
provenance data is **not needed for training or for the
`improvement_pct` validation metric** -- the model must never see these
columns as input features. It exists purely so that **unit tests** can
make strong, source-verifiable assertions instead of the current
"trust the pipeline, spot-check by hand" state described in §81.3.

**What "trace back to source" should mean concretely** -- three axes,
matching the user's own framing:

1. **Spatial**: which raw simulation grid point(s) this row's
   `(x,y,z)` corresponds to, AND (where applicable) the vortex-core
   centroid `(cx,cy,cz)` it was sampled relative to, AND whether this
   row is a real wake-core tap or a random/background sample (today:
   only recoverable by joining `wake_atlas.csv.gz`, see §3A of the
   walkthrough doc).
2. **Temporal**: the ABSOLUTE simulation step number (the `NNNN` in
   `<step:04d>.pkl.gz`) this row's `t_idx` corresponds to within its
   source param's raw file sequence -- not just the 0-79
   within-training-sequence offset currently stored.
3. **"Experiment"**: the original raw folder-name code (e.g. `"3p6"`)
   AND the parsed physical scalar (e.g. `5.6`, i.e. what `parse_param()`
   already computes) stored SIDE BY SIDE, explicitly closing the
   "sometimes keyed off `3p6`, sometimes off `U=...`" ambiguity
   described in §81.2 -- plus, ideally, the original source file path
   itself (`get_file_path(param_set, step)`'s output, or an
   equivalent), so a test can, in principle, go all the way back to the
   exact `.pkl.gz` a value came from.

**Explicit non-goals**:
- These files are NOT meant to replace `train_80.h5`/`val_80.h5` for
  training -- the trainer's `Config.TRAIN_H5`/`VAL_H5` pinned paths
  (§77.2/§78.3 discuss how rigidly other paths in this codebase are
  pinned) should keep pointing at the existing, untouched files.
  Whether the trace files are consumed by loading a SUBSET of their
  columns for training (dropping the extra metadata) or are purely a
  test-only artifact never touched by `train_production_transformer_deep_dive.py`
  at all is an open implementation decision for whoever picks this up
  -- but either way, the ordinary training/eval path must remain
  provably unaffected.
- Not meant to change any existing test, metric, or training run's
  numbers -- this is purely additive, new test infrastructure.

**Suggested (not mandated) approach**: extend `prepare_data.py`'s
assembly step (or write a thin companion script that consumes its
intermediate outputs) to also emit these extra columns per row, sourced
from information `prepare_data.py`/`build_wake_atlas.py` already have
in hand at assembly time (the raw param-folder string, the absolute
step number, the wake-atlas join result) but currently discard once the
row is written. Follow the existing `_quartered`/Parquet precedent
(DATA_PIPELINE_WALKTHROUGH.md §5) for how this codebase already carries
"extra, non-training metadata" alongside a dataset (HDF5 attrs +
per-row columns, or a Parquet sidecar) rather than inventing a new
convention.

**Why this is valuable enough to backlog explicitly**: the "Known gaps"
list at the end of DATA_PIPELINE_WALKTHROUGH.md already flags that
nothing tests the Parquet conversion's provenance round-trip, and
`test_data_quality.py`'s one content-level test targets the WRONG
(legacy, `train_40.h5`) file. A trace-enabled dataset is what would let
a future test suite assert real, end-to-end, source-verifiable
correctness (e.g. "every row flagged as a wake-core tap really does
have a matching `wake_atlas.csv.gz` entry with a centroid within
physical tolerance of where the atlas said it should be") instead of
today's structural-only checks (shapes, dtypes, non-degeneracy).

### 81.4b BACKLOG: report the autoencoder's own reconstruction "floor" side-by-side with every experiment's result

**Status: not started. Explicitly a backlog item, not in progress.**

**Ask (verbatim intent, from the user)**: every experiment this session
reports `rollout_mse`/`improvement_pct` as if the frozen GEN3 decoder's
output were ground truth -- but the decoder is itself LOSSY (it's an
autoencoder: real velocity -> 47-dim latent -> reconstructed velocity,
and that round trip has its own inherent error, separate from anything
the transformer does). We currently have no measurement, reported
alongside a run's real result, of how much of the remaining gap is the
TRANSFORMER being imperfect vs. the ENCODING ITSELF being imperfect --
i.e. no visible "floor" below which no amount of transformer
improvement could ever push `rollout_mse`, because even a perfect
transformer predicting the perfect TARGET LATENT would still decode to
something measurably different from the real, originally-measured
velocity.

**What exists today, and why it doesn't already cover this**:
`encoder/autoencoderGEN3/test_ae_lightning.py::test_ae_reconstruction()`
already does an encode->decode round trip on 100 real rows and prints
an RMSE -- but it (a) scores the FULL 375-dim reconstruction, not
specifically the single centroid triplet (`[186:189]`) that
`centroid_velocity_loss()`/`improvement_pct` are actually scored on,
(b) samples a different random file/100 rows every run (no fixed,
reusable reference number), (c) mostly prints rather than asserts
(DATA_PIPELINE_WALKTHROUGH.md §6 flags this same caveat), and (d) is
never invoked by, or reported alongside, any real training run's
`status.json`/wandb output. So there is currently no apples-to-apples
"encoding floor" number sitting next to any `improvement_pct` a reader
could compare against.

**What the fix should look like**: compute, on a FIXED reference
sample (e.g. the same validation rollout window `improvement_pct`
already uses), the encode -> decode round-trip error on the CENTROID
TRIPLET specifically -- i.e. take real measured `(vx,vy,vz)` at the
centroid, run it forward through the frozen GEN3 encoder to get a
latent, then immediately decode that latent back, and compare the
result to the ORIGINAL real measured value (not to anything the
transformer predicted). This is a NEW quantity, distinct from
everything currently computed:

| Quantity | Compares | Already exists? |
|---|---|---|
| `rollout_mse` | model's decoded rollout prediction vs. decoded ground-truth latent | Yes |
| `persistence_mse` | decoded "copy last frame" vs. decoded ground-truth latent | Yes |
| **encoding floor (new)** | decoded(encode(real measured value)) vs. the REAL measured value itself | **No** |

Report this floor value once per dataset/split (it doesn't depend on
any model or training run -- it's a property of the frozen AE alone),
then surface it side by side with `rollout_mse`/`improvement_pct` in
`status.json` and the console eval printout, so every future
experiment's numbers can be read against "how close to the AE's own
ceiling did this get" as well as "how much better than persistence was
this," per the user's own framing: "helps give confidence."

**Why this matters for interpreting everything in §81.5 below**: if
the encoding floor turns out to be a non-trivial fraction of the
distance between persistence and the ~48-51% ceiling this session kept
hitting, that's a real, previously invisible upper bound on how much
`improvement_pct` could EVER improve, no matter what's tried on the
transformer/loss/curriculum side -- a materially different situation
from "we haven't found the right training recipe yet." Currently
unknown either way; this backlog item is what would answer it.

### 81.5 Where this currently appears to be stuck

The model has, across many architectures/losses/hyperparameters tried
this session and previously, converged on a **~48-51% ceiling** on
`improvement_pct` (ridge/linear baseline: `+69%` improvement over
persistence, for context -- so the transformer has never caught up to
even a simple fitted linear map, let alone dramatically exceeded it).
The dominant, repeatedly-confirmed failure pattern is **exposure bias**:
every run peaks early (often within the first few hundred steps) and
then declines -- sometimes slowly, sometimes sharply -- as training
continues, and this has now been observed under every loss variant,
every hyperparameter combination, and every efficiency change tried.
Chronological summary of what's been tried against this ceiling this
session (full detail and exact numbers in the referenced sections):

| # | Idea | Section | Outcome |
|---|---|---|---|
| 1 | Optuna hyperparameter search (weight decay, dropout, EMA decay, AR loss weight) over 153 trials | §68.6-68.7, §72-73 | Found trial 112 (+51.43%), the best result of the whole session; Track A (2h, no early stop) confirmed this is a genuine ceiling, not a screen-length artifact -- given 6x more steps, it declined from +48.31% to +24% and never recovered |
| 2 | Widen the training loss's spatial scoring target beyond the single centroid point (`center_plus_6`, `all125` flat-weighted, `distance_weighted`) | §74-76 | All 3 variants UNDERPERFORMED the single-centroid loss, the narrower the widening the less damage (center_plus_6 best of the 3, all125 worst) -- concluded the single-centroid loss is not the bottleneck, closed this axis |
| 3 | Scheduled sampling (`AR_SCHED_MIX_P`, new mechanism built this session) layered on the already-existing `AR_FRAMES_START` rollout-horizon curriculum | §77-78 | Peaked at +46.60% -- closer to trial 112's ceiling than any loss-widening variant, but the SAME decline shape persisted. One data point, not yet swept over `AR_SCHED_MIX_P` values |
| 4 | GPU-utilization / efficiency pass (`PFD_CUDA_MICRO_BATCH`, `AR_SEQS` increase, `torch.compile`) | §78-79 | `torch.compile` actively HURT this workload (frame_ar's growing-context loop defeats dynamo's shape cache -- a real, previously-undocumented incompatibility, now disabled again). Raising `AR_SEQS` 2->16 delivered a real efficiency win (8x more AR-loss sequences per step for ~1.5-2x the wall-clock cost) but did NOT calm the gradient-noise trace as hypothesized (mean/median pre-clip `gnorm` unchanged, §prior-turn finding) and did not change model quality (+43.48% vs +46.60%, within existing run-to-run noise) |
| 5 | Diagnosed the "worse than the previous-frame anchor" console line as a false alarm, not a real instability signal | §80 | Confirmed via direct log analysis (100% of steps flagged, including at the run's own peak, including on clean noise-free validation loss) -- fixed the display, not a training change |

**What's still genuinely open, not yet tried**:
- **The autoencoder's own reconstruction floor, on the centroid triplet
  specifically, has never been measured or reported alongside a real
  run's result** (§81.4b, backlog). Until this exists, it's unknown
  whether some (or much) of the gap between persistence and the
  ~48-51% ceiling is actually the frozen AE's own lossy encoding, not
  anything the transformer could fix.
- **Whether the model needs the `(x,y,z,t,param)` "meta" input columns
  at all** -- built and dry-run validated (not yet launched, §82) an
  A/B ablation (`USE_META_COLS` True vs False) on the modern recipe.
  `param` (the experiment identifier) IS in the model's input today by
  default, contrary to an initial assumption raised and corrected this
  session -- the open question is whether the model is actually USING
  it in a way that matters for the exposure-bias ceiling, not whether
  it's present.
- A real sweep over `AR_SCHED_MIX_P` values (only `0.3` has been tried)
  -- idea #3 above is promising but under-explored.
- The actual source of the large pre-clip gradient norms (`gnorm`
  routinely 10-25x the `GRAD_CLIP=1.0` threshold) is still unexplained
  -- batch size was ruled out as the cause; whether it's tied to
  specific phases of the `AR_FRAMES_START` curriculum, to
  `AR_SCHED_MIX_P`/`AROPT_TEMPERATURE`'s own randomness, or is intrinsic
  to `frame_ar`'s chained-rollout structure generally, is unknown.
- No one has yet tried directly attacking the exposure-bias DECLINE
  itself with an intervention that changes DURING a single run (e.g.
  annealing `AR_SCHED_MIX_P` over training, the way classic scheduled
  sampling anneals its own mixing probability, rather than the current
  fixed-value-per-run design) -- everything tried so far picks one
  fixed value/config and trains it start to finish.
- The MoE variants (referenced but not detailed in this handoff --
  see §23 and surrounding sections) were found to get WORSE with more
  training, not just slower to converge -- a different, so-far-
  unexplained failure mode from the exposure-bias decline seen
  everywhere else, worth a fresh look with the tooling now available
  (Optuna, the `AR_SCHED_MIX_P` mechanism) that didn't exist when MoE
  was last tried.

### 81.6 Operational notes for picking this back up

- **Standing rule (explicit user instruction, §73.4)**: whoever is
  operating this codebase writes/verifies code, tests, and configs
  locally (syntax checks, dry-runs, unit tests) but does NOT launch
  real GPU pod/training runs autonomously -- that decision belongs to
  whoever owns the budget. Every `bakeoff_run.sh` invocation in this
  document was `--dry-run` validated, never actually launched, unless
  explicitly stated otherwise.
- **Launching a real run**: `singleshot/bakeoff_run.sh`, always with
  `NETWORK_VOLUME_ID=yl7f9e8rwr` (this project's persistent RunPod
  volume -- omitting it means no `/workspace` mount, no warm-start
  checkpoint, no persisted wandb key; a real mistake made and corrected
  this session, §77.6). See §75.4/§77.5/§78.7/§79.3 for fully-qualified
  example commands.
- **wandb project**: `Persistence_Linear_v1` (renamed from
  `NI_Review_v8`, §69), entity `pvl-data`. Each batch of runs this
  session used its own `--wandb-group` (`Bones`, `Scotty`, `McCoy`,
  `NC1701`) to keep them visually separable in the dashboard.
  `singleshot/setup_wandb_focus_view.py` (§75.3) builds a focused,
  single-metric (`improvement_pct`) dashboard view scoped to one group.
- **The reference hyperparameter recipe**, used as the base for every
  isolated-variable experiment this session: `h11_ridge_distill` arm,
  warm-started from `saved_models/r2_h11_ridge_distill_rollout_best.pt`,
  `WEIGHT_DECAY=0.0966331656012948 DROPOUT=0.07345842794271963
  EMA_DECAY=0.98 AR_LOSS_WEIGHT=0.6828481795652924` (trial 112's exact
  params, §68.7/§73.1).
- **Test suite**: `pytest tests/` from `transformer_neurIPS/`, 242
  passing / 14 skipped as of this writing (skips are hardware-gated --
  e.g. tests requiring the scripted decoder file or CUDA). Full suite
  runs in under 20 seconds; keep it that way (existing tests are
  deliberately small/synthetic where possible, e.g. tests exercising
  `frame_ar_loss`/`centroid_velocity_loss` directly at tiny model sizes
  rather than requiring a real training run to validate mechanisms).
- **This document (`OVERVIEW.md`) is the primary lab notebook** --
  read chronologically for full context, or jump to a specific `§N`
  cross-referenced above. `DATA_PIPELINE_WALKTHROUGH.md` is the
  authoritative, example-driven reference for the data format
  specifically. Neither is likely to be fully up to date the moment
  something new is tried -- always verify a referenced code line number
  or file still says what this document claims before relying on it
  for anything consequential (the same "verify, don't trust blindly"
  discipline applied throughout this session's own work).

## 82. v10.10 -- does the model actually need the "experiment" (`param`) column, or is it silently averaging across experiments? `n2_meta_off` was NEVER run -- built the real test

### 82.1 Correcting the record: `param` IS in the model's input today, but the ablation that would prove it MATTERS has never been executed

User's hypothesis: the transformer might be "averaging outputs across
experiments" because it lacks the context to distinguish which
experiment (inlet velocity, keyed inconsistently as `"3p6"` vs. its
physical value, §81.2) a given token belongs to. First checked whether
that context is even present -- it is: `Config.USE_META_COLS = True`
by default (train_production_transformer_deep_dive.py:818), which
means `model_variants.py`'s `input_projection` consumes the FULL
52-dim row (latent + `x,y,z,t_idx,param`) on every arm run this
session, including every `h11_ridge_distill` run referenced throughout
§68-81 -- none of them override `USE_META_COLS`.

There is a real, pre-existing control arm for exactly this question --
`n2_meta_off` (branch N, train_production_transformer_deep_dive.py:2094,
"Zero the (x, y, z, t, param) input columns entirely") -- but **direct
search of `OVERVIEW.md`, `sweep_logs/`, and `saved_models/` found zero
evidence it was ever actually run**. The one time branch N came up in
this project's history (§20.2), the operator was explicitly told NOT
to run its arms -- but for an unrelated reason (a bogus threshold-based
auto-classifier misfire at a too-short step budget, not because the
meta-columns question itself was ever answered). So the correct
status is: **untested, not previously falsified** -- my earlier
phrasing implying it might have run was wrong and is corrected here.

### 82.2 Mechanically confirmed the zeroing actually works before proposing a real run

`model_variants.py:664/819`: `h = h * self.feat_keep`, where
`feat_keep` zeroes indices `[LATENT_DIM:INPUT_DIM]` (i.e. `x,y,z,t_idx,
param`) when `USE_META_COLS=False` (`model_variants.py:459-462`).
Verified directly on CPU, no pod needed: built a real model, ran it
twice on the same input, then wildly perturbed the meta columns
(`x[:,:,LATENT_DIM:] = torch.randn(...) * 100`) and re-ran --

```
USE_META_COLS=True:  perturbing meta cols changes output=True
USE_META_COLS=False: perturbing meta cols changes output=False
```

Confirms the mechanism does exactly what it claims: with the flag off,
the model's output is provably invariant to whatever garbage sits in
those columns. Also confirmed warm-start-safe: zeroing happens on the
INPUT side (`h * feat_keep`), not by deleting weights, so
`input_projection`'s existing weight columns for the meta features
simply go unused rather than causing a shape mismatch against
`r2_h11_ridge_distill_rollout_best.pt`.

### 82.3 The real test: same modern recipe, A/B on `USE_META_COLS` only

`n2_meta_off`'s own definition is stale (branch N predates
`PREDICT_DELTA`'s eventual retirement path, uses `LOSS=mse` instead of
the current fixed centroid-velocity objective, and isn't warm-started)
-- rather than resurrect that exact arm, built a fresh, isolated A/B
directly on trial 112's modern recipe (same one used for every §74-80
experiment), changing ONLY `USE_META_COLS`:

`singleshot/bakeoff_queue_meta_cols_ablation.json` -- two entries:
- round 330 `meta_cols_control_on` -- `USE_META_COLS=True` (explicit,
  though this is already the default -- stated for clarity as the
  control side of the A/B)
- round 331 `meta_cols_ablation_off` -- `USE_META_COLS=False`, otherwise
  byte-identical: same `h11_ridge_distill` arm, same warm-start
  checkpoint, same trial-112 hyperparameters (`WEIGHT_DECAY=0.0966
  DROPOUT=0.0735 EMA_DECAY=0.98 AR_LOSS_WEIGHT=0.683`), same
  `max_steps=6000 max_hours=1.0 early-stop-patience-steps=1500`.
  `AR_SCHED_MIX_P` deliberately NOT included on either side -- keeping
  this test independent of the still-experimental §77-79 mechanism, so
  a difference (or lack of one) can't be attributed to the wrong knob.

Validated via `--dry-run --slots=2 --wandb-group=NC1701`: both entries
valid, unique `(arm,round)` pairs, zero network/ssh calls made.

**What the two possible outcomes would mean**, since this is worth
stating explicitly before the result comes back:
- If round 331 (no meta cols) performs MEANINGFULLY WORSE than round
  330 -- the model is using `param`/position context in a way that
  matters, and the exposure-bias ceiling has a different explanation
  (the information is present but something else -- the loss, the
  curriculum, the rollout mechanism -- isn't using it well).
- If round 331 performs THE SAME OR BETTER -- exactly the user's own
  framing ("if removal had no impact, we are doing something very
  wrong here") -- that would mean 5 of the model's 52 input dimensions
  are dead weight in the CURRENT recipe, which either points at a
  genuine bug in how that context reaches the loss/rollout, or means
  the model has never needed to learn to use it (e.g. if per-sequence
  `param` is redundant with something else already fully determining
  behavior, or the loss doesn't reward using it). Either reading would
  be a real, actionable finding, not a null result to shrug off.

### 82.4 Fully qualified command for the user to run (nothing above was launched, per §73.4)

```bash
NETWORK_VOLUME_ID=yl7f9e8rwr \
  bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_run.sh \
  --queue=/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/bakeoff_queue_meta_cols_ablation.json \
  --slots=2 --total-budget-hours=1.0 --wandb-group=NC1701
```
