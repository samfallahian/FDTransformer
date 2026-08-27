# `encoder_neurIPS/` — Overview

The autoencoder half of the NeurIPS submission. Learns a 375 → 47 latent representation of the per-cube physical state that the transformer in `../transformer_neurIPS/` then rolls out autoregressively. Everything in this folder is scoped to the encoder itself: dataset construction, model variants, training loops, saved checkpoints, and the AE-only unit tests. Held-out cases (`4p4`, `6p4`) are excluded from every training pool here so that downstream evaluation on those cases remains clean.

## 1. Layout

```
encoder_neurIPS/
├── build_neurIPS_dataset.py     # source cubes → encoder train/val pickle
├── models.py                    # AE architecture + SE-residual blocks + variants
├── train_neurIPS.py             # multi-model round-based training driver
├── train_production.py          # single-model production trainer (round_production)
├── train_simultaneous.py        # joint training driver (simultaneous_training)
├── saved_models/
│   ├── round_1/ … round_5/      # multi-round variant sweeps
│   ├── round_production/        # frozen production AE (feeds the transformer)
│   ├── simultaneous_training/
│   └── training_state.pkl       # resumable state for round-based runs
├── unit_tests/
│   ├── test_data.py             # dataset sanity checks
│   └── test_gen3_vs_winner.py   # "Winner beats GEN3" PASS/FAIL gate
├── plots/                       # loss curves / diagnostic figures
└── wandb/                       # W&B run logs
```

## 2. Data pipeline — `build_neurIPS_dataset.py`

- Reads raw cubes via `EfficientDataLoader` at the project root.
- Source: `/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data`; destination: `/Users/kkreth/PycharmProjects/data/encoder_neurIPS`.
- `is_excluded(path, excluded_terms)` filters out any file whose path matches an excluded term. The default excluded set is `{"4p4", "6p4"}` — the held-out cases used later for reversal evaluation and for the `U* = 10` persistence baseline.
- `build_neurips_dataset(...)` splits the surviving files into train/val, serializes them as pickles, and logs counts / timing.

## 3. Model — `models.py`

- `ORIGINAL_DIM = 375` (flattened per-cube physical vector), `LATENT_DIM = 47`.
- `BaseNeurIPSAE` — encoder / decoder MLP with reconstruction MSE plus a small latent regularizer (`+ 0.00005 · ‖z‖²`) to keep the latent bounded without collapsing it.
- Building blocks: `ResidualBlock`, `SEBlock` (squeeze-and-excitation gate), `SEResidualBlock`.
- `create_model_variant(idx)` — enumerates the architecture variants swept in `round_*`. The production ("winner") variant is the one that the transformer's `load_autoencoder(device)` binds to at rollout time.

## 4. Training drivers

### 4.1 `train_neurIPS.py` (round-based sweep)
- `get_device()` cuda → mps → cpu.
- `load_data()` loads the pickle produced by the dataset builder.
- `train_one_model(model_idx, device, train_loader, val_loader, round_num, og_performance=None, dry_run=False)` — trains one variant, tracks best-val, checkpoints under `saved_models/round_<n>/`.
- `main(dry_run=False, force_restart=False)` — drives rounds 1…5, resumable via `saved_models/training_state.pkl`.

### 4.2 `train_production.py` (production AE)
- `BATCH_SIZE = 4096`, `LEARNING_RATE = 5e-5` (both stepped down from an initial `12288` / `1e-4` for stability — see the inline "Choice 1" comments).
- `run_unit_test()` invokes `unit_tests/test_gen3_vs_winner.py` and gates release on the `"Winner beats GEN3!"` message. The production checkpoint saved under `saved_models/round_production/` is the one downstream code loads.

### 4.3 `train_simultaneous.py` (joint / simultaneous)
- Same `BATCH_SIZE = 4096`, `LEARNING_RATE = 5e-5`.
- `train_one_epoch(...)` / `validate(...)` — thin per-epoch loop.
- Writes to `saved_models/simultaneous_training/`.

## 5. Unit tests

- `unit_tests/test_data.py` — shape, dtype, and split-integrity checks on the built dataset.
- `unit_tests/test_gen3_vs_winner.py` — regression gate against the prior "GEN3" AE. `train_production.py` refuses to promote a checkpoint that fails this test.

## 6. How this folder plugs into the rest of the repo

- The AE trained here is loaded by `transformer_neurIPS/tests/test_model_vs_baseline.py::load_autoencoder` and reused unchanged inside `transformer_neurIPS/persistence_formal_documentation.py` (see `Documentation/persistence_formal_documentation.md`). Every physical-space metric the transformer publishes — MAE / RMSE / L2 in m/s — flows through this decoder.
- The 40-frame transformer sequences in `transformer_neurIPS/data/val_40.h5` carry the AE latents in their first `LATENT_DIM = 47` columns, so any change to `ORIGINAL_DIM` / `LATENT_DIM` here forces a rebuild there.

---

## 7. Versioning

### v1.0 — everything above this line

The state of `encoder_neurIPS/` at the time of this document is frozen as **v1.0**. Concretely:

- 375 → 47 SE-residual autoencoder, `train_production.py` promoted via the GEN3 gate.
- Held-out cases `4p4`, `6p4` excluded from the AE training pool.
- Downstream transformer at `transformer_neurIPS/` is the **40-timestep** model (12 context / 28 forecast at 120 Hz → 100 ms / 233.3 ms / 333.3 ms window). All persistence-baseline numbers, reviewer follow-ups, and `Documentation/persistence_formal_documentation.md` are v1.0 artifacts.

### v2.0 — planned (80-timestep migration)

Direct quote of the plan (author's git note):

> OK, so this SHOULD be all the code for the working 40 time step model. All things being the same....the 80 time step model should do well here... I'm going to next:
>
> Recreate the 80 time step version of the test and validation input files for the training.
>
> Refactor the training accordingly to size for 80 time steps...ensure we first load the weights from the 40 we have (as much as possible), then see how we progress over a FEW epochs. This will CRUSH my M4!!
>
> After that, it will be run (possibly in many different permutations) on a:
> H200

Operational breakdown:

1. **Regenerate 80-frame sequences.** Rebuild `transformer_neurIPS/data/{train,val}_80.h5` (12 context + 68 forecast = 80 frames = 566.7 ms at 120 Hz) using the same AE checkpoint promoted here in v1.0. The AE itself is unchanged — only the transformer's sequence length grows. This is what unlocks the 300 ms post-reversal sustain claim and the long-horizon persistence-gap figure that a 233.3 ms window cannot substantiate (see `Documentation/persistence_baseline_design.md` and the 40-vs-80 first-principles analysis).
2. **Refactor the transformer trainer for `NUM_TIME = 80`.** Update `Config.NUM_TIME`, positional embedding capacity, and any hard-coded 40s in `train_production_transformer_deep_dive.py` / `model_variants.py`.
3. **Warm-start from the 40-step weights.** Load as much of the v1.0 transformer checkpoint as `load_state_dict(..., strict=False)` will accept; only positional / length-dependent tensors should re-initialize. Track exactly which keys land in `missing_keys` / `unexpected_keys` and treat non-benign ones as failures the way `BENIGN_MISSING_KEYS` already handles the AE.
4. **Short local shakedown on M4 (MPS).** A few epochs only — enough to confirm the warm-start is not diverging and the causality probe still passes on the longer window. Expect memory pressure: `rollout_frames` peak scales with batch × sequence length, so `PFD_BATCH_SIZE`-style device-dependent defaults will need to be re-tuned for 80 frames.
5. **Full training on H200.** Multiple permutations (warm-start vs. cold-start, `NUM_TIME = 80` vs. staged 40 → 80 curriculum, etc.). Rerun `persistence_formal_documentation.py` with `PFD_HORIZON_FRAMES` extended to the new 68-frame horizon and update `Documentation/persistence_formal/` accordingly.

**Scope note.** v2.0 does **not** re-train the autoencoder. `models.py`, `train_production.py`, `saved_models/round_production/`, and the GEN3 gate all stay pinned at v1.0 unless a v2.0 evaluation surfaces a decoder-side bottleneck. Any AE-side change would be v2.1 and require re-running the encoder unit tests before promotion.
