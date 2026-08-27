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
