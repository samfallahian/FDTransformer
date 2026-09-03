# Deep-dive sweep report -- round 2  (`round2_20260903_013205`)

## VERDICT

**Branch `S`: Control already works -- scale and refine**

- best arm = v3_h9_moreseqs at +46.49% vs persistence
- beat the previous-frame anchor on the training objective: 0/4
- The plateau is broken; the remaining work is scaling and refinement.

Recommended next command:

```
python sweep_deep_dive.py --round 2 --branch S --max-parallel 1 --max-steps 360
```

which runs:

- `s1_capacity_xl` -- E768/L12 at the winning settings.
- `s2_steps_3x` -- Same model, 3x the step budget, lower cosine floor.
- `s3_swiglu` -- SwiGLU feed-forward at the winning settings.
- `s4_lr_low` -- Half the peak LR, longer warmup -- plateau escape by annealing.
- `s5_bigbatch` -- Effective batch 2048 with LR scaled up.
- `s6_e3_scaled` -- e3_ar_long's EXACT winning config (OVERVIEW.md v4.2's H200 shallow sweep, +30.77% vs persistence, the first arm in this whole investigation to post a positive average improvement) at a production-scale step budget instead of the shallow sweep's 2000. NOT one of the generic s1-s5 scaling knobs -- those apply to the CONTROL config's settings, not to the AR mechanism that actually won this round, so running them as suggested would scale the wrong starting point. MAX_STEPS baked in here (matches s2_steps_3x's convention) but a launcher's --max-steps still wins if passed, per apply_arm()'s 'CLI beats the arm' rule.
- `s7_h9_scaled` -- h9_ar_freq1's EXACT winning config (OVERVIEW.md v4.6, +43.78% vs persistence at 400 steps, beating both e3_ar_long and s6_e3_scaled) at a longer step budget. h9 is h1_ar_freq2's config (AR horizon 8, AR_SEQS 2) with AR-loss applied on EVERY step (AR_EVERY_N_STEPS=1) -- the most expensive AR frequency tried, so unlike s6_e3_scaled no MAX_STEPS is baked in here: the right budget depends on how much wall-clock is actually available on the box running it, so run_sweep_h300_scale_h9.sh's own default (sized for a real time budget, not a fixed convention) controls it -- a launcher's --max-steps always wins over an arm's own value regardless, per apply_arm()'s 'CLI beats the arm' rule.

## Run settings

```
run_id             = round2_20260903_013205
round              = 2
max_parallel       = 1
max_steps          = 360
max_hours          = 0.0833333
seed               = 1337
subset_ratio       = 1.0
rollout_seqs       = 64
gpus               = 0
started            = 2026-09-03 01:32:05
finished           = 2026-09-03 01:52:31
```

## Diagnostics (run once, before any training)

_Diagnostics did not complete -- see `diagnostics.log`._

## Arm results

```
arm            status      steps   mins  params     IMPROV%     frame1%       last%    roll MSE    pers MSE     val tf      train  >const?  >anch?
--------------------------------------------------------------------------------------------------------------------------------------------------
v3_h9_moreseqs ok            302    5.0    4.79      +46.49    -5000.88      +55.72    0.001522    0.002844   0.032168   0.023891      yes      NO
v2_h9_clip     ok            328    5.0    4.79      +42.45    -5931.35      +55.87    0.001637    0.002844   0.027813   0.028462      yes      NO
v1_h9_wd       ok            312    5.0    4.79      +42.14    -5772.94      +54.39    0.001646    0.002844   0.020955   0.022585      yes      NO
v4_h9_lrlow    ok            305    5.0    4.79      +41.52    -6348.52      +57.49    0.001663    0.002844   0.022730   0.022802      yes      NO
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 68 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = -6348.5%   '@' = +58.7%

v3_h9_moreseqs |_*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-5000.88  f28=+55.72
v2_h9_clip     |.*##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-5931.35  f28=+55.87
v1_h9_wd       |.*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-5772.94  f28=+54.39
v4_h9_lrlow    |.*##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@%%|  f1=-6348.52  f28=+57.49
```

- `v3_h9_moreseqs`: -5000.9, -1838.5, -935.7, -585.3, -428.1, -333.1, -226.2, -171.9, -120.6, -88.1, -63.2, -44.9, -23.9, -6.2, +4.2, +13.0, +19.2, +25.7, +27.8, +33.6, +35.8, +40.6, +42.8, +45.1, +45.7, +49.0, +51.1, +51.3, +54.1, +53.1, +54.9, +55.6, +56.5, +57.0, +57.0, +57.9, +58.1, +58.1, +56.7, +57.7, +57.7, +57.9, +58.1, +58.2, +57.0, +57.5, +57.6, +56.9, +57.1, +55.8, +55.9, +55.7, +55.7, +55.4, +55.5, +55.6, +55.1, +55.1, +55.2, +56.4, +55.7, +56.4, +57.0, +56.0, +54.4, +55.3, +55.9, +55.7
- `v2_h9_clip`: -5931.4, -2230.7, -1112.0, -745.9, -504.6, -395.2, -297.2, -219.5, -168.8, -132.2, -94.2, -67.4, -51.5, -30.7, -17.8, -8.6, +1.4, +9.1, +13.9, +22.9, +28.3, +28.7, +33.2, +36.5, +42.2, +44.8, +43.9, +49.3, +48.1, +51.8, +51.2, +52.4, +54.1, +52.3, +55.7, +55.3, +56.7, +55.0, +57.8, +57.3, +56.1, +57.4, +56.4, +56.8, +57.1, +57.1, +56.5, +55.7, +56.7, +53.6, +53.9, +55.4, +55.3, +55.8, +54.3, +54.4, +53.5, +54.5, +55.7, +55.2, +53.6, +54.7, +55.7, +55.1, +55.4, +55.9, +55.6, +55.9
- `v1_h9_wd`: -5772.9, -2156.9, -1078.9, -740.5, -481.9, -369.6, -281.7, -205.3, -155.6, -129.4, -84.4, -61.6, -48.2, -28.7, -16.6, -6.6, +1.7, +11.1, +15.3, +25.1, +31.2, +30.8, +35.5, +38.6, +43.5, +45.9, +44.3, +49.7, +48.8, +52.3, +51.0, +51.6, +53.3, +51.4, +55.7, +54.3, +56.0, +53.5, +57.4, +56.8, +55.5, +56.7, +55.5, +56.6, +56.4, +55.6, +54.8, +54.1, +55.2, +50.3, +51.4, +53.4, +54.4, +54.9, +52.9, +53.0, +52.7, +54.0, +55.1, +54.7, +52.4, +53.7, +54.8, +53.7, +54.3, +55.0, +54.4, +54.4
- `v4_h9_lrlow`: -6348.5, -2282.8, -1120.5, -796.2, -531.1, -405.8, -310.7, -231.4, -176.6, -152.1, -102.2, -79.0, -63.2, -49.5, -34.3, -14.8, -9.9, +1.8, +5.9, +21.3, +24.3, +18.9, +29.5, +35.8, +39.1, +43.3, +44.5, +46.7, +48.6, +49.0, +51.4, +52.4, +53.5, +45.9, +54.4, +53.4, +56.0, +51.5, +56.2, +57.7, +56.8, +57.6, +56.3, +58.4, +58.1, +57.1, +56.3, +55.6, +57.8, +47.7, +51.7, +56.0, +56.7, +57.2, +55.3, +55.6, +55.0, +57.3, +57.9, +57.7, +52.7, +56.2, +58.6, +57.3, +57.9, +58.7, +58.3, +57.5

## Training curves (subsampled)

### `v3_h9_moreseqs`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.092926   0.066941    0.005250    -84.63  8.49e-04    3.2
    200   0.042773   0.053125    0.001688    +40.64  4.44e-04    4.1
    300   0.023891   0.032168    0.001522    +46.49  8.94e-05    5.0
```

### `v2_h9_clip`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.104420   0.082226    0.006061   -113.13  8.49e-04    3.0
    200   0.065899   0.046209    0.005028    -76.80  4.44e-04    3.9
    300   0.028462   0.027813    0.001637    +42.45  8.94e-05    4.7
```

### `v1_h9_wd`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.056401   0.083113    0.003369    -18.49  8.49e-04    3.1
    200   0.092260   0.059735    0.004959    -74.36  4.44e-04    4.0
    300   0.022585   0.020955    0.001646    +42.14  8.94e-05    4.9
```

### `v4_h9_lrlow`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.118025   0.080692    0.003574    -25.67  4.24e-04    3.1
    200   0.062774   0.035618    0.002317    +18.54  2.22e-04    4.1
    300   0.022802   0.022730    0.001663    +41.52  4.47e-05    5.0
```

## What each arm was

### `v3_h9_moreseqs`

- **what**: h9's exact config with AR_SEQS 2->4 -- less gradient noise per AR-loss application, same horizon/frequency. h5_ar_moreseqs tested this at freq=8 without a clear benefit; worth re-checking at freq=1 where AR gradient variance matters more.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'DELTA_ANCHOR': 'persistence', 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `v2_h9_clip`

- **what**: h9's exact config with tighter gradient clipping (0.5 vs the default 1.0) -- AR-loss every step means far more frequent backprop through the AR loop than any prior arm; checks whether a tighter clip damps any resulting instability without costing the improvement.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'DELTA_ANCHOR': 'persistence', 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `v1_h9_wd`

- **what**: h9's exact config with weight decay 0.01 -> 0.05 -- an independent regularisation axis from gradient clipping (v2), testing whether damping weight growth improves consistency under AR-loss-every-step.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'DELTA_ANCHOR': 'persistence', 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.05, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `v4_h9_lrlow`

- **what**: h9's exact config at half the peak LR (5e-4 vs 1e-3) -- gentler optimisation under the most AR-gradient-dense regime tried yet; checks whether consistency improves at the cost of some peak improvement, the classic stability/speed tradeoff.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'DELTA_ANCHOR': 'persistence', 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.0005, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

