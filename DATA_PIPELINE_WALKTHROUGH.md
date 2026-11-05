# Data Pipeline Walkthrough — `a3b_delta_ar` / v1.0

Scope: only the pipeline that feeds the model currently being trained
(`train_production_transformer_deep_dive.py`, arm `a3b_delta_ar`, `train_80.h5`/
`val_80.h5`, `layout=N_NT_NX_C`, `NUM_TIME=80`, `NUM_X=10`, `INPUT_DIM=52`,
`LATENT_DIM=47`). All numbers below are read directly out of the real files
in this repo — none are illustrative placeholders. See `OVERVIEW.md` for the
full versioned history this walkthrough is a cross-section of.

---

## 1. The big picture

```
raw sim output                 GEN3 AttentionSE            build_wake_atlas.py         prepare_data.py            train_production_
(x,y,z,vx,vy,vz per            autoencoder (frozen,        (vorticity-core             (assembles the             transformer_deep_dive.py
 grid point, per               trained separately)         centroids -> (y,z)          52-dim rows into                (trains on
 param, per step)                    |                       taps to sample)            80x10x52 sequences)         52-dim rows)
      |                               v                            |                            |                            |
      |                    375-dim reconstruction        wake_atlas.csv.gz            train_80.h5 / val_80.h5    the 47-dim latent slice
      |                    (125 x (vx,vy,vz) cube)      (y,z taps + spatial          data: (N, 80, 10, 52)      is decoded back through
      +----- encode ------------->                       centroid cx,cy,cz)                                    the SAME frozen decoder
                                   |                                                                            to score the training loss
                              47-dim latent -------------------------------------------------------------------------+
                              (per grid point,                                                                       |
                               per step)                                                                    "centroid velocity" loss
                                                                                                          (vx,vy,vz at the cube's center)
```

Two upstream artifacts are **frozen inputs** to this arm, not things it
computes itself:

- The **47-dim latent** already baked into every row of `train_80.h5`/
  `val_80.h5` — produced by a separately-trained GEN3 AttentionSE encoder,
  long before `prepare_data.py` ever runs.
- The **frozen scripted decoder**,
  `encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`
  — used only at *training/eval* time, to turn a latent back into a
  physical velocity for the loss (§4). It never touches `prepare_data.py`.

---

## 2. The 52-column feature vector

Every row of the `data` dataset — shape `(N_sequences, 80, 10, 52)` — packs
one grid point at one timestep into 52 floats, laid out exactly like this
(`prepare_data.py:949-955`, `train_production_transformer_deep_dive.py:711-722`):

| Columns  | Name          | What it is                                                             |
|----------|---------------|-------------------------------------------------------------------------|
| `0:47`   | `latent`      | GEN3 AttentionSE encoder's 47-dim bottleneck for this grid point/step. Opaque — no per-dimension meaning is documented anywhere in this pipeline. |
| `47`     | `x`           | streamwise coordinate, one of the fixed 10 values in `X_COORDS = [-18,-14,-10,-6,-2,1,5,9,13,17]` |
| `48`     | `y`           | cross-plane coordinate (int, stored as float32)                        |
| `49`     | `z`           | cross-plane coordinate (int, stored as float32)                        |
| `50`     | `t_idx`       | position within the 80-frame sequence (`0..79`), filled in at sequence-assembly time, not extraction time |
| `51`     | `param`       | the physical run parameter as a scalar, via a fixed `parse_param()` lookup (e.g. `"6p4" -> 10.0`) |

`47 (latent) + 5 (x, y, z, t_idx, param) = 52 = Config.INPUT_DIM`. This 5/47
split is also the reason a training example can't be *fully* interpreted
from `train_80.h5` alone — the 5 extra columns tell you *where* a token is,
but not what its 47 latent numbers physically mean; that requires the
frozen decoder (§4).

**Real row** — `val_80_quartered.h5`, `sequence_index=0`, `t=0`, `x_index=0`:

```
latent[0:47] = [-0.00792, 0.02075, -0.02538, -0.00523, 0.01075, 0.02153,
                 0.00021, -0.00841, 0.00103, 0.00278, 0.01446, -0.04377,
                 0.02203, 0.00854, -0.00915, -0.01382, -0.01477, 0.03059,
                -0.00107, -0.00820, 0.00577, -0.01318, -0.00806, 0.00471,
                 0.01973, 0.00431, -0.00396, -0.00736, 0.01945, -0.01400,
                 0.00181, 0.01386, -0.00346, 0.02061, -0.00818, -0.02547,
                -0.02258, 0.02882, -0.02847, -0.00990, -0.01305, -0.00178,
                 0.00608, 0.00148, 0.01005, -0.02020, -0.00267]
x, y, z, t_idx, param = -18.0, 3.0, -1.0, 0.0, 10.0    # param 10.0 == "6p4"
```

---

## 3. Two different "centroids" — do not conflate them

This pipeline uses the word "centroid" for two unrelated things, computed
by different code, in different spaces:

### 3A. The wake-core spatial centroid `(cx, cy, cz)` — where to *sample*

Computed once, offline, by `build_wake_atlas.py::find_vortex_core_region()`
(lines 114-201, a self-contained copy of `identify_training_wake.py:21-61`).
For each `(param, step)`:

1. Pivot the raw `(x, y, z, vx, vy, vz)` point cloud onto a regular grid.
2. Compute vorticity via `np.gradient`:
   `omega_z = d(vy)/dx - d(vx)/dy` (and the `x`/`y` analogues).
3. `mag = sqrt(omega_x^2 + omega_y^2 + omega_z^2)`, restricted to `|x| <= 30`.
4. The **vortex core** = every grid point with `mag >= 0.90 * peak_val`.
5. The **centroid** is the vorticity-magnitude-**weighted mean position**
   of those core points — not a distance calc, a weighted arithmetic mean
   in the raw simulation `(x, y, z)` Euclidean space:

   ```
   cx = sum(core_x * weight) / sum(weight)      (same for cy, cz)
   ```

Real example, `wake_atlas.csv.gz`, `param="6p4"`, core near `(y_grid=-4,
z_grid=18)`, `step=1`:

```
cx = 16.21567499721026
cy = -4.815537245520996
cz = 14.03884311380249
```

The centroid itself is **not** what gets written into `train_80.h5` rows —
`build_wake_atlas.py` snaps the *individual core points* (not the centroid)
onto the real `(y, z)` sensor grid and emits a sliding window of neighbour
taps around each (`compute_neighbour_offsets`, default `radius_yz=8,
stride_yz=4`). Those `(y_grid, z_grid)` taps are what `prepare_data.py`
actually samples; `cx/cy/cz` ride along only as provenance columns in the
atlas row.

A related, but currently **orphaned**, table lives at
`cube_centroid_mapping/df_all_possible_combinations_with_neighbors.csv`
(repo root) — 378 columns = `centroid_x, centroid_y, centroid_z` + 125 × 3
`nbr_dx_{-2..2}_dy_{-2..2}_dz_{-2..2}_{x,y,z}` offset columns. No `.py` file
in the repo currently reads it (`grep -rl` over `*.py` for either name
returns nothing) — it appears to predate the AE's 125-point cube layout
below and was never wired into the live pipeline.

### 3B. "Centroid velocity" — the actual training target

This is the sense that dominates `OVERVIEW.md` and the trainer, and is
**not** a spatial location at all — it's a decoded physical velocity.

`train_production_transformer_deep_dive.py:684-691`:

```python
DECODED_DIM = 375
N_TRIPLETS = 125
CENTROID_TRIPLET_IDX = 62            # 0-based, middle of 125
CENTROID_SLICE = slice(186, 189)     # -> (vx, vy, vz)
```

The frozen GEN3 decoder reconstructs, from any 47-dim latent, a 375-dim
vector laid out as `[vx_0, vy_0, vz_0, ..., vx_124, vy_124, vz_124]` — the
velocity at each of 125 points of a 5×5×5 physical cube around that token.
Index 62 is the cube's **center point** (`dx=dy=dz=0` in the
`cube_centroid_mapping` layout from §3A) — `decode_centroid()`
(lines 2143-2169) slices out just `[186:189]`, i.e. `(vx_62, vy_62, vz_62)`.
`centroid_velocity_loss()` (lines 2186-2204) decodes both the predicted and
the target latent this way and scores the L2 error between the two
`(vx, vy, vz)` triplets — this is the real training objective for
`a3b_delta_ar` (this exact wiring was a live bug in earlier arms, fixed at
`OVERVIEW.md` v3.7/§18: the loss *looked* wired to this target but wasn't).

**In one sentence**: §3A's centroid tells `prepare_data.py` *where in
(y,z) to sample*; §3B's centroid is *what the model is scored on*, decoded
from the 47-dim latent that was already sitting at that sampled location.
They share a name and nothing else.

---

## 4. One training example, dissected end to end

**`val_80_quartered.h5`, `sequence_index=37`** (chosen over index 0 because
it lands on a real wake-atlas hit rather than a background/random tap).

**Step 1 — read the coordinate straight out of the row** (columns 47-51):

```
x = -18.0   (X_COORDS[0])
y = -4.0
z = 18.0
param = 10.0  -->  parse_param inverse lookup  -->  "6p4"
```

**Step 2 — join against `wake_atlas.csv.gz`** (`param=="6p4" & y_grid==-4
& z_grid==18`) — a match exists:

```
is_neighbour = True
step = 1
cx, cy, cz = 16.21567499721026, -4.815537245520996, 14.03884311380249
```

So this token is a **sliding-window neighbour tap** around a real vortex
core whose weighted centroid sits at `(16.22, -4.82, 14.04)` in the sim's
physical frame — this row's own `(y,z)=(-4,18)` is one of the offset taps
around that core, not the core's exact location. (Contrast: `sequence_index=0`
from §2 has `(y,z)=(3,-1)` under `param="6p4"`, which has **zero** matching
rows anywhere in `wake_atlas.csv.gz` for that param — i.e. it is provably a
random/background sample, and that distinction is only recoverable by
joining against the atlas file, never from the H5 row alone.)

**Step 3 — the raw 52-dim row**, `t_offset=0, x_index=0`:

```
latent[0:47] = [ 0.00635, -0.01416,  0.00231, -0.00034,  0.00355, -0.00872,
                 0.00861,  0.00442, -0.00523, -0.01681, -0.00108, -0.00226,
                -0.03996,  0.01464, -0.00063,  0.00142,  0.00354,  0.09527,
                -0.01728, -0.00366, -0.00810, -0.00632,  0.00189,  0.00176,
                -0.01146, -0.00358,  0.00990,  0.01117, -0.01169,  0.00080,
                -0.02116, -0.00677,  0.01991,  0.00646, -0.01000,  0.00568,
                 0.00074, -0.01931,  0.02810, -0.01350,  0.00636,  0.01591,
                 0.01849, -0.01922,  0.00434, -0.00927, -0.00666]
x, y, z, t_idx, param = -18.0, -4.0, 18.0, 0.0, 10.0
```

**Step 4 — what the 47-dim latent decodes to, structurally.** The 47
numbers above are not individually interpretable — no per-dimension
meaning is documented anywhere in this pipeline (they're the frozen GEN3
encoder's opaque bottleneck). What *is* fully determined without running
the decoder is the shape of what comes back out:

```
decode_centroid(latent)  -->  dec(latent.float())[..., 186:189]
```

i.e. run this 47-vector through
`Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`, get a 375-dim
reconstruction (125 `(vx,vy,vz)` triplets forming a physical 5×5×5 cube
around this `(t, x, y, z)` point), and take triplet 62 (`slice[186:189]`)
— the cube's center, i.e. the velocity **at this token's own location**.
That center-velocity triplet is exactly what `centroid_velocity_loss()`
scores this token's prediction against.

---

## 5. Parquet vs. HDF5 vs. "you have to get it elsewhere"

`dataBricks/convert_quartered_h5_to_parquet.py` converts
`train_80_quartered.h5`/`val_80_quartered.h5` into one row per
`(sequence_index, time_index, x_index)` with a 52-float `features` column,
plus every source HDF5 attribute carried as Parquet file metadata.

| Have it in the Parquet file? | Item |
|---|---|
| ✅ Yes | The raw 52-dim feature vector per token (§2) — latent + `x,y,z,t_idx,param`, fully self-describing since layout/dim attrs travel as metadata (`input_dim`, `num_time`, `num_x`, `layout`, ...). |
| ✅ Yes (as metadata only) | Provenance breadcrumbs: `sampled_fraction=0.25`, `sampled_seed=1337`, `sampled_from=.../val_80.h5`, `wake_atlas_source`, `wake_atlas_sha256`, `param_list`, `split_version`, `windows_per_coord`. |
| ❌ No — need `wake_atlas.csv.gz` (`transformer_neurIPS/data/wake_atlas.csv.gz`) | Whether a given `(param, y, z)` is a real vortex-core/neighbour tap or a random background sample, and (if core) the spatial centroid `(cx,cy,cz)` and vorticity magnitude near it (§3A). The Parquet metadata records the atlas's *sha256*, not its contents. |
| ❌ No — need `encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt` | Turning any 47-dim latent back into a physical velocity — the 375-dim reconstruction, the center/"centroid velocity" triplet, all of it (§3B, §4). The Parquet file only ever carries the latent, never a decoded value or the decoder weights. |
| ❌ No — need `prepare_data.py` source + the original pickle tree (`source_root` attr) | The `parse_param()` name↔scalar table (to turn `param=10.0` back into `"6p4"` without guessing), `X_COORDS`, and the wake/random plan-generation logic itself if you want to reproduce *why* a `(y,z)` was chosen rather than just that it was. |
| ❌ No — need the un-quartered `train_80.h5`/`val_80.h5` | The full dataset — the Parquet/quartered files are a 25% random subsample (`sampled_fraction=0.25`); this is the actual training data only in miniature. |
| ❌ No — need `train_80_enriched.h5`/`val_80_enriched.h5` (from `enrich_h5_with_velocity.py`) | Pre-decoded target `centroid_velocity` values, if you'd rather not run the decoder yourself. These enriched files exist today only as companions to the un-quartered `train_80.h5`/`val_80.h5` — not generated for the quartered/Parquet files. |

---

## 6. Unit tests that check we're doing what we think we're doing

| Test | File | Invariant checked |
|---|---|---|
| `TestAtlasLoader.test_atlas_loader_parses_expected_columns` | `tests/test_wake_atlas.py:92` | `load_wake_atlas()` parses atlas CSV columns correctly and dedupes `(param,y,z)`. |
| `TestAtlasLoader.test_include_neighbours_toggle` | `tests/test_wake_atlas.py:112` | `include_neighbours=False` keeps only true core taps, drops sliding-window neighbours. |
| `TestMissingAtlasBehaviour.test_missing_atlas_hard_fails_without_env_var` | `tests/test_wake_atlas.py:155` | No silent fallback to a stale hardcoded coordinate list — hard `SystemExit` if the atlas is missing. |
| `TestMissingAtlasBehaviour.test_missing_atlas_falls_back_with_env_var_and_matches_legacy_24` | `tests/test_wake_atlas.py:161` | The explicit legacy-fallback opt-in reproduces the original 24 hardcoded taps exactly. |
| `TestSlidingWindowNeighbourGeneration.test_offsets_stride_4_radius_8` | `tests/test_wake_atlas.py:176` | `compute_neighbour_offsets(8,4)` yields exactly the 25 `(dy,dz)` combinations in `{-8,-4,0,4,8}^2`. |
| `TestSlidingWindowNeighbourGeneration.test_neighbour_emission_respects_probe_grid` | `tests/test_wake_atlas.py:187` | Neighbour taps that fall off the real probe grid are dropped, not silently kept as phantom rows. |
| `TestRandomPlanExclusion.test_random_plan_pool_is_probe_minus_atlas` | `tests/test_wake_atlas.py:226` | Random-plan pool == `probe_yz − atlas_yz` exactly — a wake coordinate can never masquerade as "random." |
| `TestBuilderRestartSafety.*` | `tests/test_wake_atlas.py:277,290` | Re-running `build_wake_atlas.py` skips already-built shards; deleting one shard recomputes only that one. |
| `TestAutoMergeFromShards.test_loader_auto_merges_shard_tree` | `tests/test_wake_atlas.py:331` | Atlas loader can reconstruct the merged view in-memory from per-shard files if no merged `.csv.gz` exists yet. |
| `RandomPlanGridMissesRealDataGrid.test_random_plan_grid_regular_data_grid_irregular` | `tests/test_sequence_dropoff_diagnosis.py:183` | Regression test for the historical bug: a regular random-plan sampling grid missing the irregular real-data grid. |
| `ExtractReturnsNoneForMissingYZ.test_present_yz_returns_data_missing_yz_returns_none` | `tests/test_sequence_dropoff_diagnosis.py:111` | `extract_from_file` returns real data for a present `(y,z)`, `None` for an absent one — never fabricates. |
| `RandomPlanRuntimeGuards.test_random_plan_hit_rate_on_wake_grid` | `tests/test_sequence_dropoff_diagnosis.py:142` | Random-plan hit-rate against the real wake grid stays within expected bounds. |
| `DropoffAttributionEndToEnd` / `RandomPlansShouldMostlySurvive.test_overall_drop_is_small` | `tests/test_sequence_dropoff_diagnosis.py:203,391` | End-to-end sequence dropoff during assembly is correctly attributed and stays small. |
| `TestPrepareData::test_dtypes_and_content` | `tests/test_data_quality.py:12` | Columns 48/49 (`y`,`z`) are integer-valued despite float32 storage; latent columns aren't degenerate (zero-fraction `<1%`). **Targets legacy `train_40.h5` (`NUM_X=26`)**, not the `train_80.h5`/`NUM_X=10` files this arm trains on — same column-index assumptions, stale file target. |
| `TestDataFileIntegrity.test_train_80_integrity` / `test_val_80_integrity` | `tests/test_data_files_size_parity.py:128,131` | `FEATURE_DIM=52` matches `data.shape[-1]`, `NUM_X=10`, no compression, byte-size within a sane band — directly on the files this arm uses. |
| `TestSequenceFilesPresent.test_80_files_present` | `tests/test_data_files_present.py:83` | `train_80.h5`/`val_80.h5` exist on disk before any dependent test runs. |
| `RandomPlanRuntimeGuards` / `ExtractionTimeoutGuards` / `StreamedH5WriteGuards` | `tests/test_prepare_data_runtime_guards.py:28,72,161` | Operational guardrails (wall-clock deadlines, streamed writes) around the assembly pipeline, not layout semantics. |
| `test_ae_reconstruction` | `encoder/autoencoderGEN3/test_ae_lightning.py:14` | Encode→decode round trip on 100 real 375-dim rows stays within a sane RMSE. **Caveat**: this script prints its checks rather than `assert`ing most of them — runs cleanly under pytest collection without proving much on failure. |
| `TestCentroidAvailabilitySample.test_train_80_centroid_sample` / `test_val_80_centroid_sample` | `tests/test_centroid_availability.py:120,123` | The exact `(47)→(375)→[186:189]` path (§3B/§4): decoder output width is 375, the centroid slice is finite (no NaN/Inf), `max\|centroid velocity\| < 1e3`, and the sample spans more than one distinct `(param,y,z)`. This is the direct test of the mechanism this whole document is explaining. |
| `test_token_native_roundtrip` / `test_frame_native_roundtrip` / etc. | `tests/test_scripted_save.py:119,175,140,198,243` | The **transformer's own** scripted-checkpoint round trip — a different model than the GEN3 AE decoder, included for completeness. |

### Known gaps (no test currently covers these)

- No test asserts on the newer H5 attrs individually (`wake_atlas_source`,
  `wake_atlas_sha256`, `wake_atlas_rows`, `split_version`, `sampled_*`) —
  only shape/compression are checked (`test_data_files_size_parity.py`).
- **Nothing tests `dataBricks/convert_quartered_h5_to_parquet.py`** — no
  test file exists under `dataBricks/` at all, so the Parquet schema and
  the `transformer_neurips_provenance` metadata round-trip (verified by
  hand while building that script) has no regression coverage yet.
