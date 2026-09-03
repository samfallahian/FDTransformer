# Deep-dive sweep report -- round 2  (`round2_20260903_003105`)

## VERDICT

**Branch `S`: Control already works -- scale and refine**

- best arm = s7_h9_scaled at +31.69% vs persistence
- beat the previous-frame anchor on the training objective: 0/1
- The plateau is broken; the remaining work is scaling and refinement.

Recommended next command:

```
python sweep_deep_dive.py --round 2 --branch S --max-parallel 1 --max-steps 2000
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
run_id             = round2_20260903_003105
round              = 2
max_parallel       = 1
max_steps          = 2000
max_hours          = 12.0
seed               = 1337
subset_ratio       = 1.0
rollout_seqs       = 64
gpus               = 0
started            = 2026-09-03 00:31:05
finished           = 2026-09-03 00:49:15
```

## Diagnostics (run once, before any training)

_Diagnostics did not complete -- see `diagnostics.log`._

## Arm results

```
arm            status      steps   mins  params     IMPROV%     frame1%       last%    roll MSE    pers MSE     val tf      train  >const?  >anch?
--------------------------------------------------------------------------------------------------------------------------------------------------
s7_h9_scaled   ok           2000   18.1    4.79      +31.69    -2150.23      +28.83    0.001943    0.002844   0.008250   0.007779      yes      NO
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 68 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = -2150.2%   '@' = +42.8%

s7_h9_scaled   |.+##%%%%%%%%%%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-2150.23  f28=+28.83
```

- `s7_h9_scaled`: -2150.2, -905.9, -436.2, -267.2, -181.3, -128.9, -83.2, -53.0, -32.3, -12.8, -2.4, +9.6, +16.6, +25.4, +30.4, +32.0, +35.4, +37.2, +38.5, +40.0, +41.0, +41.5, +41.7, +41.8, +41.8, +42.5, +42.2, +42.8, +42.2, +42.2, +41.7, +42.0, +41.4, +41.2, +40.6, +40.0, +39.0, +38.5, +38.1, +38.4, +37.7, +37.0, +36.3, +36.6, +35.3, +34.4, +34.4, +33.2, +32.7, +32.4, +31.2, +30.8, +30.4, +29.0, +29.4, +29.3, +29.2, +29.1, +28.5, +29.3, +28.8, +29.5, +29.8, +29.2, +28.6, +28.7, +28.7, +28.8

## Training curves (subsampled)

### `s7_h9_scaled`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.090755   0.090741    0.002233    +21.48  9.87e-04    3.9
    400   0.068917   0.076973    0.002850     -0.21  9.28e-04    5.5
    600   0.063298   0.062374    0.001954    +31.30  8.24e-04    7.1
    800   0.043054   0.039409    0.002411    +15.21  6.88e-04    8.7
   1000   0.023553   0.022321    0.001861    +34.55  5.34e-04   10.2
   1200   0.022515   0.017462    0.002034    +28.48  3.77e-04   11.8
   1400   0.014527   0.013366    0.001970    +30.73  2.34e-04   13.4
   1600   0.009460   0.010467    0.001915    +32.68  1.19e-04   14.9
   1800   0.008604   0.010061    0.001907    +32.93  4.55e-05   16.5
   2000   0.007779   0.008250    0.001943    +31.69  2.00e-05   18.1
```

## What each arm was

### `s7_h9_scaled`

- **what**: h9_ar_freq1's EXACT winning config (OVERVIEW.md v4.6, +43.78% vs persistence at 400 steps, beating both e3_ar_long and s6_e3_scaled) at a longer step budget. h9 is h1_ar_freq2's config (AR horizon 8, AR_SEQS 2) with AR-loss applied on EVERY step (AR_EVERY_N_STEPS=1) -- the most expensive AR frequency tried, so unlike s6_e3_scaled no MAX_STEPS is baked in here: the right budget depends on how much wall-clock is actually available on the box running it, so run_sweep_h300_scale_h9.sh's own default (sized for a real time budget, not a fixed convention) controls it -- a launcher's --max-steps always wins over an arm's own value regardless, per apply_arm()'s 'CLI beats the arm' rule.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'DELTA_ANCHOR': 'persistence', 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

