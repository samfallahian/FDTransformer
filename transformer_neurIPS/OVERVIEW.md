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
