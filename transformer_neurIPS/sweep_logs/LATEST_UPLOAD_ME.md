# Deep-dive sweep report -- round 2  (`round2_20260902_230514`)

## VERDICT

**Branch `S`: Control already works -- scale and refine**

- best arm = h1_ar_freq2 at +41.90% vs persistence
- ridge linear frame-map baseline = +69.49% (centroid space)
- beat the previous-frame anchor on the training objective: 0/8
- The plateau is broken; the remaining work is scaling and refinement.

Recommended next command:

```
python sweep_deep_dive.py --round 2 --branch S --max-parallel 1 --max-steps 400
```

which runs:

- `s1_capacity_xl` -- E768/L12 at the winning settings.
- `s2_steps_3x` -- Same model, 3x the step budget, lower cosine floor.
- `s3_swiglu` -- SwiGLU feed-forward at the winning settings.
- `s4_lr_low` -- Half the peak LR, longer warmup -- plateau escape by annealing.
- `s5_bigbatch` -- Effective batch 2048 with LR scaled up.
- `s6_e3_scaled` -- e3_ar_long's EXACT winning config (OVERVIEW.md v4.2's H200 shallow sweep, +30.77% vs persistence, the first arm in this whole investigation to post a positive average improvement) at a production-scale step budget instead of the shallow sweep's 2000. NOT one of the generic s1-s5 scaling knobs -- those apply to the CONTROL config's settings, not to the AR mechanism that actually won this round, so running them as suggested would scale the wrong starting point. MAX_STEPS baked in here (matches s2_steps_3x's convention) but a launcher's --max-steps still wins if passed, per apply_arm()'s 'CLI beats the arm' rule.

## Run settings

```
run_id             = round2_20260902_230514
round              = 2
max_parallel       = 1
max_steps          = 400
max_hours          = 12.0
seed               = 1337
subset_ratio       = 1.0
rollout_seqs       = 64
gpus               = 0
started            = 2026-09-02 23:05:14
finished           = 2026-09-02 23:35:29
```

## Diagnostics (run once, before any training)

```
torch            = 2.14.0+cu130
device           = cuda  gpu = NVIDIA B300 SXM6 AC x1  bf16 = True
train sequences  = 59280
val sequences    = 25410
```

### 1. Was the old attention call actually leaking the future?

```
{
  "outcome": "raised",
  "error": "RuntimeError: Need attn_mask if specifying the is_causal hint. You may use the Transformer module method `generate_square_subsequent_mask` to create this mask."
}
```
`nn.MultiheadAttention(attn_mask=None, is_causal=True)` -- for the module API `is_causal` is only a HINT that `attn_mask` already is the causal mask. `causal: false` here means the previous runs were trained with the future visible; `outcome: raised` means that call could never have run and something else was happening.

### 2. The old ConvBlock padding

```
symmetric_padding_1 (old)    max change in PAST outputs from a FUTURE perturbation = 2.196e+00
left_padding_2 (fixed)       max change in PAST outputs from a FUTURE perturbation = 0.000e+00
```
Non-zero for `symmetric_padding_1 (old)` confirms `padding=1` with `kernel_size=3` let every token see t+1, once per block.

### 3. Causality of each configuration we are about to train

```
a0_control       causal=True  before_cut=0.000e+00 after_cut=2.441e+00
a1_nonorm        causal=True  before_cut=0.000e+00 after_cut=2.020e+00
a2_mse           causal=True  before_cut=0.000e+00 after_cut=2.073e+00
a3_delta         causal=True  before_cut=0.000e+00 after_cut=7.000e+00
a4_frame         causal=True  before_cut=0.000e+00 after_cut=3.451e+00
```

### 4. Sanity floor: what trivial predictors score on the training objective

`tokenization=token` (target std = 0.0171655286103487)

```
zeros            l2norm=0.105299  mse=2.947e-04
mean             l2norm=0.104394  mse=2.881e-04
previous token   l2norm=0.035298  mse=6.371e-05
previous frame   l2norm=0.018608  mse=9.056e-06
```

`tokenization=frame` (target std = 0.017161697149276733)

```
zeros            l2norm=0.334962  mse=2.946e-04
mean             l2norm=0.332291  mse=2.879e-04
previous frame   l2norm=0.059051  mse=8.872e-06
```

Compare each arm's `train` column against these. A train loss ABOVE the zero-predictor means the model learned nothing at all, and the cause is conditioning or optimisation -- not capacity, not exposure bias, not architecture. The previously saved checkpoint recorded train L2 = 0.4266 against a zero-predictor score of ~0.074 and a previous-frame score of ~0.017, i.e. it was ~6x worse than outputting nothing.

### 5. Is there learnable temporal structure beyond persistence?

```
[raw latent space, 470-dim -- NOT comparable to an arm's IMPROV%]
persistence MSE                = 0.00034558018635598447
ridge linear frame-map MSE     = 0.00013600662014857495
linear improvement             = +60.64%
linear improvement, 1 frame    = +12.93%

[decoded centroid velocity space, m/s -- apples-to-apples with IMPROV%]
persistence MSE                = 0.0024857626344838954
ridge linear frame-map MSE     = 0.0007585177890832314
linear improvement             = +69.49%
linear improvement, 1 frame    = +19.12%
fit on                         = 323584 frame transitions
```
The CENTROID figure is the floor a competent model must clear -- it is in the same units evaluate() scores arms in (see `IMPROV%`/`roll MSE`/`pers MSE` below). A ridge regression that beats persistence there while the transformer does not means the transformer is broken. The raw-latent figure above it is informational only: the decoder is a nonlinear map, so a raw-latent improvement is not the same claim as a decoded-velocity improvement.

### 6. Input feature scales

```
latent std (mean over 47 dims) = 0.01538304053246975
columns 47:52 mean             = [-0.5, -1.3345, 0.5222, 39.5, 11.3886]
columns 47:52 std              = [11.0567, 41.7427, 13.0119, 23.0922, 2.9336]
```
Those columns (x, y, z, t, param) went through the same `nn.Linear` as the latents. The size of that mismatch is what `NORMALIZE_FEATURES` removes.

## Arm results

```
arm            status      steps   mins  params     IMPROV%     frame1%       last%    roll MSE    pers MSE     val tf      train  >const?  >anch?
--------------------------------------------------------------------------------------------------------------------------------------------------
h1_ar_freq2    ok            400    4.2    4.79      +41.90    -6302.11      +53.89    0.001652    0.002844   0.019617   0.021609      yes      NO
h4_ar_long_freq4 ok            400    3.8    4.79       +0.92   -1.30e+04      +28.65    0.002818    0.002844   0.014765   0.017751      yes      NO
h3_ar_short_freq4 ok            400    2.9    4.79       -0.22   -1.08e+04      +25.34    0.002850    0.002844   0.015741   0.018562      yes      NO
h2_ar_freq4    ok            400    3.4    4.79      -22.97   -1.38e+04       +9.99    0.003497    0.002844   0.015613   0.017953      yes      NO
h6_ar_fbnoise  ok            400    3.0    4.79      -31.32   -1.48e+04       +5.98    0.003735    0.002844   0.014559   0.017504      yes      NO
h5_ar_moreseqs ok            400    3.0    4.79      -39.53   -1.56e+04       +0.51    0.003968    0.002844   0.013102   0.015953      yes      NO
h7_ar_wd       ok            400    4.0    4.79      -49.88   -1.63e+04      -12.17    0.004262    0.002844   0.014336   0.017758      yes      NO
h8_ar_lrlow    ok            400    3.0    4.79      -53.21   -2.12e+04      -16.85    0.004357    0.002844   0.014776   0.017993      yes      NO
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 68 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = -21184.9%   '@' = +59.1%

h1_ar_freq2    |*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-6302.11  f28=+53.89
h4_ar_long_freq4 |=#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.30e+04  f28=+28.65
h3_ar_short_freq4 |=#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.08e+04  f28=+25.34
h2_ar_freq4    |-*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.38e+04  f28=+9.99
h6_ar_fbnoise  |-*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.48e+04  f28=+5.98
h5_ar_moreseqs |-*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.56e+04  f28=+0.51
h7_ar_wd       |_*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.63e+04  f28=-12.17
h8_ar_lrlow    |.*#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-2.12e+04  f28=-16.85
```

- `h1_ar_freq2`: -6302.1, -2452.1, -1251.3, -816.0, -582.3, -460.2, -337.3, -259.4, -189.9, -148.2, -112.2, -86.9, -65.1, -40.1, -26.2, -13.8, -5.0, +5.6, +9.6, +18.5, +23.6, +27.7, +31.8, +35.4, +39.0, +42.6, +44.6, +47.1, +49.1, +51.4, +51.2, +52.0, +54.1, +54.5, +56.5, +57.1, +58.4, +57.5, +58.8, +58.3, +57.9, +59.1, +58.9, +58.7, +59.0, +58.5, +57.6, +57.7, +58.2, +55.0, +55.6, +56.8, +56.0, +57.4, +55.7, +55.8, +54.9, +55.7, +56.9, +56.5, +55.8, +55.7, +55.6, +54.4, +55.0, +55.3, +55.1, +53.9
- `h4_ar_long_freq4`: -12990.7, -4638.0, -2443.7, -1597.3, -1178.0, -955.5, -716.7, -566.3, -444.9, -359.7, -303.6, -245.5, -205.7, -161.6, -137.8, -114.6, -93.1, -78.6, -66.8, -53.0, -45.2, -33.2, -26.9, -19.2, -13.3, -8.6, -3.1, -0.1, +5.8, +9.2, +13.0, +14.9, +18.0, +21.0, +22.5, +24.2, +27.8, +29.1, +28.7, +29.9, +31.0, +33.0, +33.4, +34.3, +33.9, +34.0, +33.8, +34.2, +35.1, +33.4, +33.8, +34.2, +33.9, +34.7, +33.2, +33.5, +33.3, +32.6, +33.0, +32.8, +31.9, +31.6, +31.4, +30.0, +29.0, +29.8, +29.8, +28.6
- `h3_ar_short_freq4`: -10849.1, -4613.7, -2426.6, -1585.2, -1174.8, -951.1, -717.0, -565.6, -440.9, -360.9, -299.7, -244.9, -207.1, -161.8, -138.1, -115.6, -97.3, -80.4, -68.1, -53.7, -45.0, -35.1, -28.5, -19.8, -13.5, -9.3, -4.8, -0.7, +4.8, +8.7, +11.7, +13.5, +17.2, +19.5, +21.5, +23.3, +26.5, +27.8, +28.5, +29.2, +29.8, +32.4, +32.3, +33.0, +32.7, +32.9, +32.1, +32.5, +33.7, +31.3, +32.1, +32.8, +32.1, +33.1, +31.9, +32.1, +31.2, +31.1, +31.3, +30.6, +28.9, +29.1, +29.1, +27.6, +26.8, +27.5, +26.8, +25.3
- `h2_ar_freq4`: -13763.2, -5935.7, -3143.2, -2058.0, -1531.5, -1235.2, -939.6, -745.5, -585.5, -482.0, -404.4, -336.4, -289.6, -233.6, -202.3, -172.6, -147.3, -127.4, -111.5, -92.3, -80.7, -67.9, -60.8, -49.9, -43.0, -36.6, -30.5, -26.0, -19.6, -12.4, -10.6, -7.8, -2.7, +0.8, +3.5, +5.0, +10.4, +12.5, +14.0, +13.6, +14.9, +18.3, +19.4, +19.2, +19.2, +19.3, +18.5, +19.4, +20.8, +19.3, +19.5, +20.7, +19.6, +22.1, +19.6, +19.9, +18.7, +18.2, +19.3, +17.9, +16.6, +15.6, +14.7, +12.6, +12.3, +13.3, +12.7, +10.0
- `h6_ar_fbnoise`: -14780.4, -6202.0, -3261.5, -2158.2, -1605.4, -1310.7, -990.0, -785.4, -629.5, -515.8, -439.2, -357.6, -309.1, -252.1, -216.9, -188.1, -161.0, -142.2, -123.7, -106.5, -95.6, -80.5, -72.5, -60.9, -51.5, -46.8, -39.9, -35.6, -28.1, -24.5, -17.7, -14.8, -10.1, -6.5, -4.7, -2.8, +2.7, +4.9, +4.7, +7.2, +8.2, +11.6, +12.2, +13.5, +12.9, +13.8, +13.8, +14.3, +15.0, +14.3, +14.6, +14.9, +14.1, +14.8, +13.9, +13.5, +13.1, +11.8, +12.0, +10.8, +9.9, +10.0, +10.1, +8.6, +6.2, +7.2, +7.0, +6.0
- `h5_ar_moreseqs`: -15592.5, -6665.7, -3532.3, -2329.9, -1746.7, -1415.0, -1080.6, -855.9, -687.1, -562.2, -480.8, -395.5, -342.4, -279.9, -241.8, -210.4, -178.2, -159.9, -140.6, -120.6, -109.4, -93.2, -85.0, -72.5, -62.5, -57.3, -50.1, -44.7, -36.7, -31.9, -25.4, -22.5, -17.2, -12.8, -11.2, -8.3, -3.6, -0.3, -0.0, +1.3, +3.0, +6.3, +7.4, +8.1, +8.3, +9.1, +8.9, +9.1, +10.3, +10.3, +10.3, +10.8, +9.5, +11.0, +9.7, +9.3, +9.5, +8.4, +7.7, +7.4, +5.9, +5.0, +4.8, +3.4, +1.8, +3.1, +2.5, +0.5
- `h7_ar_wd`: -16328.1, -7070.2, -3761.4, -2443.4, -1846.0, -1507.7, -1148.2, -890.8, -719.7, -593.9, -504.2, -421.3, -369.7, -304.2, -261.6, -229.3, -197.4, -172.3, -153.7, -133.5, -118.9, -102.5, -94.2, -82.0, -73.8, -69.2, -60.5, -54.9, -46.5, -37.7, -35.4, -31.8, -25.0, -21.9, -19.2, -17.0, -10.4, -7.9, -5.6, -5.2, -3.7, +0.1, +0.9, -1.3, +1.2, +0.9, +0.9, +0.4, +2.1, +2.2, +2.5, +3.1, +0.1, +3.7, +1.3, +0.9, +1.7, +0.0, +0.1, -1.0, -2.9, -5.0, -5.2, -8.8, -9.9, -7.8, -9.7, -12.2
- `h8_ar_lrlow`: -21184.9, -6605.5, -3471.8, -2332.9, -1735.4, -1404.4, -1062.6, -845.3, -693.1, -588.7, -483.1, -400.0, -358.6, -303.8, -261.8, -231.9, -204.0, -178.6, -161.6, -129.0, -118.3, -110.9, -100.6, -84.6, -71.8, -63.7, -63.7, -53.7, -47.6, -36.5, -35.9, -34.6, -27.6, -38.2, -25.4, -24.1, -11.6, -15.4, -6.2, -9.8, -8.3, -0.0, -1.8, -2.6, -3.3, -2.8, -6.1, -4.0, -2.0, -11.6, -7.9, -2.5, +0.4, +5.1, -5.4, -7.1, -7.4, -2.1, -0.8, -3.0, -12.3, -9.8, -7.4, -18.1, -16.4, -6.8, -11.9, -16.8

## Training curves (subsampled)

### `h1_ar_freq2`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.095452   0.080716    0.002794     +1.76  5.34e-04    3.2
    400   0.021609   0.019617    0.001652    +41.90  2.00e-05    4.2
```

### `h4_ar_long_freq4`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.056318   0.060718    0.005091    -79.02  5.34e-04    3.0
    400   0.017751   0.014765    0.002818     +0.92  2.00e-05    3.8
```

### `h3_ar_short_freq4`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.046753   0.043169    0.017167   -503.67  5.34e-04    2.5
    400   0.018562   0.015741    0.002850     -0.22  2.00e-05    2.9
```

### `h2_ar_freq4`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.094016   0.057909    0.010077   -254.34  5.34e-04    2.8
    400   0.017953   0.015613    0.003497    -22.97  2.00e-05    3.3
```

### `h6_ar_fbnoise`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.072923   0.054300    0.005840   -105.37  5.34e-04    2.6
    400   0.017504   0.014559    0.003735    -31.32  2.00e-05    3.0
```

### `h5_ar_moreseqs`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.065049   0.046243    0.009938   -249.45  5.34e-04    2.5
    400   0.015953   0.013102    0.003968    -39.53  2.00e-05    2.9
```

### `h7_ar_wd`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.081135   0.096875    0.006009   -111.30  5.34e-04    3.6
    400   0.017758   0.014336    0.004262    -49.88  2.00e-05    4.0
```

### `h8_ar_lrlow`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.043888   0.045793    0.007084   -149.12  2.67e-04    2.6
    400   0.017993   0.014776    0.004357    -53.21  1.00e-05    2.9
```

## What each arm was

### `h1_ar_freq2`

- **what**: e3's 8-frame horizon at 4x its AR-loss frequency (every 2 steps).
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h4_ar_long_freq4`

- **what**: a4b_ar_very_long's 14-frame horizon, but at e3's proven-better frequency band instead of a4b's every-16-steps -- tests whether a4b only lost because it was under-applied, per OVERVIEW.md 23.2's explanation, not because 14 frames is worse.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 14, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h3_ar_short_freq4`

- **what**: Shorter horizon (4 frames) at high frequency (every 4 steps) -- brackets e3 from below.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 4, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h2_ar_freq4`

- **what**: e3's 8-frame horizon at 2x its AR-loss frequency (every 4 steps).
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h6_ar_fbnoise`

- **what**: e3's config plus noise on the fed-back prediction during the AR loop -- practice on a noisy version of its own errors, not just the clean rollout.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.005, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h5_ar_moreseqs`

- **what**: e3's config with AR_SEQS=8 instead of 2 -- less gradient noise per AR-loss application, same horizon/frequency.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h7_ar_wd`

- **what**: e3's config plus heavier regularisation (weight decay 0.1, dropout 0.05) -- damp amplifying modes on top of AR training instead of instead of it.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.05, 'WEIGHT_DECAY': 0.1, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `h8_ar_lrlow`

- **what**: e3's config at half the peak LR -- r4_lr_sweep (Branch R) showed aggressive LR causes catastrophic rollout divergence WITHOUT AR training; checks whether AR training is more or less LR-sensitive.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'frame_ar', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 8, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.0005, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

