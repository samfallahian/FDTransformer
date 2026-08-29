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

### 17.5 Known stale tests found during this pass (not fixed)

Surfaced by running the full suite against the freshly-rebuilt data
files; left as-is because fixing them was out of scope for this pass —
tracked here so they aren't mistaken for new regressions:

- `tests/test_data_files_present.py` — hard-codes a single `NUM_X = 26`
  applied to both the 40-frame (legacy, correctly 26) and 80-frame
  (v3.1+, actually 10) cohorts. `test_80_files_present` currently fails
  (`10 != 26`) against a correctly-built file. Fix: make the expected
  x-dimension per-`(num_time, filename)` entry instead of one global
  constant.
- `tests/test_data_files_size_parity.py` — asserts 40- and 80-frame
  on-disk byte sizes stay within 2% of each other. That invariant
  predates both the v3.1 x-window cut and the v3.4 wake-atlas density
  increase; `train_80.h5` is now legitimately ~6x the size of
  `train_40.h5` (different `NUM_X`, ~8x more sequences), not evidence of
  truncation. The invariant this test encodes no longer holds and needs
  to be retired or rewritten against a different premise, not patched.
- `tests/test_model_vs_baseline.py` — two `setUpClass` errors evaluating
  v1.0 checkpoints against `val_40.h5`; pre-existing (reproduces against
  both the old `SEQ_LEN=2080` and the new `SEQ_LEN=800`), caused by the
  test not scoping `Config` back to the v1.0 shape before loading the
  legacy file. Unrelated to this pass's `NUM_X` fix.
