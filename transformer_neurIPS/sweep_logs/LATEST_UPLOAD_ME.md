# Deep-dive sweep report -- round 2  (`round2_20260902_220830`)

## VERDICT

**Branch `R`: A linear map beats the transformer -- the model or the framing is broken**

- best arm = r5_mse_nonorm at -8.53% vs persistence
- ridge linear frame-map baseline = +60.64%
- beat the previous-frame anchor on the training objective: 4/5
- A ridge regression beats the transformer by 2x. The model or the framing is broken, not the task.

Recommended next command:

```
python sweep_deep_dive.py --round 2 --branch R --max-parallel 1 --max-steps 2000
```

which runs:

- `r1_frame` -- Frame tokenisation: match the linear baseline's own factorisation.
- `r2_frame_delta_mse` -- Frame + delta + MSE -- as close to the linear baseline as a net gets.
- `r3_tiny` -- Deliberately tiny (E128/L2): if small beats large, this is an optimisation failure.
- `r4_lr_sweep` -- Peak LR 3e-3 with a long warmup.
- `r5_mse_nonorm` -- MSE objective, feature normalisation OFF -- isolates the normalisation change.

## Run settings

```
run_id             = round2_20260902_220830
round              = 2
max_parallel       = 1
max_steps          = 2000
max_hours          = 12.0
seed               = 1337
subset_ratio       = 1.0
rollout_seqs       = 64
gpus               = 0
started            = 2026-09-02 22:08:30
finished           = 2026-09-02 22:30:35
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
symmetric_padding_1 (old)    max change in PAST outputs from a FUTURE perturbation = 1.457e+00
left_padding_2 (fixed)       max change in PAST outputs from a FUTURE perturbation = 0.000e+00
```
Non-zero for `symmetric_padding_1 (old)` confirms `padding=1` with `kernel_size=3` let every token see t+1, once per block.

### 3. Causality of each configuration we are about to train

```
a0_control       causal=True  before_cut=0.000e+00 after_cut=2.098e+00
a1_nonorm        causal=True  before_cut=0.000e+00 after_cut=2.266e+00
a2_mse           causal=True  before_cut=0.000e+00 after_cut=2.635e+00
a3_delta         causal=True  before_cut=0.000e+00 after_cut=7.000e+00
a4_frame         causal=True  before_cut=0.000e+00 after_cut=2.972e+00
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
persistence MSE                = 0.00034558018635598447
ridge linear frame-map MSE     = 0.00013600662014857495
linear improvement             = +60.64%
linear improvement, 1 frame    = +12.93%
fit on                         = 323584 frame transitions
```
This is the floor a competent model must clear. A ridge regression that beats persistence while the transformer does not means the transformer is broken. A ridge regression that also gets ~0% means persistence is simply strong at this dt.

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
r5_mse_nonorm  ok           2000    4.6    4.79       -8.53    -9452.28      +28.66    0.003086    0.002844   0.050896   0.046826      yes      NO
r4_lr_sweep    ok           2000    4.1    4.79    -1608.63   -1.19e+05    -1250.96    0.048590    0.002844   0.004455   0.004487      yes     yes
r1_frame       ok           2000    3.6    5.00   -2.90e+04       +8.66   -2.32e+04    0.827749    0.002844   0.007382   0.007197      yes     yes
r3_tiny        ok           2000    3.5    0.42   -4.01e+04   -9.21e+05   -3.50e+04    1.141897    0.002844   0.004673   0.004632      yes     yes
r2_frame_delta_mse ok           2000    3.6    5.00   -1.00e+07      +17.31   -2.21e+07  284.768732    0.002844   0.006699   0.006459      yes     yes
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 68 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = -22107539.7%   '@' = +28.8%

r5_mse_nonorm  |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@%%%|  f1=-9452.28  f28=+28.66
r4_lr_sweep    |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-1.19e+05  f28=-1250.96
r1_frame       |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=+8.66  f28=-2.32e+04
r3_tiny        |%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-9.21e+05  f28=-3.50e+04
r2_frame_delta_mse |%%%%%%%%%%%%%%%%%###############************++++++++=====---____....|  f1=+17.31  f28=-2.21e+07
```

- `r5_mse_nonorm`: -9452.3, -3980.9, -2088.1, -1392.8, -1046.3, -847.0, -664.3, -534.3, -429.7, -339.4, -298.5, -240.5, -210.2, -165.3, -139.6, -121.5, -100.1, -90.2, -83.8, -74.3, -59.9, -51.0, -44.8, -37.9, -28.1, -21.2, -18.3, -12.6, -9.8, -7.1, -5.0, -2.7, +1.1, +2.5, +4.5, +7.6, +11.3, +13.1, +12.7, +12.7, +12.9, +15.6, +18.8, +19.4, +18.5, +22.0, +21.5, +21.1, +21.9, +23.3, +23.4, +23.9, +25.7, +22.1, +25.4, +25.4, +22.9, +23.6, +23.9, +24.3, +24.8, +27.1, +28.5, +28.5, +28.8, +28.7, +27.7, +28.7
- `r4_lr_sweep`: -119383.4, -59891.0, -33427.4, -22610.0, -17955.0, -14619.9, -11684.2, -9453.2, -8055.5, -6872.7, -6107.8, -5159.0, -4726.7, -4088.5, -3724.3, -3430.8, -3149.6, -2970.6, -2732.0, -2482.3, -2378.9, -2220.1, -2122.0, -2013.2, -1906.3, -1857.0, -1806.8, -1747.0, -1670.8, -1609.0, -1549.6, -1497.2, -1454.7, -1409.2, -1390.3, -1355.8, -1275.4, -1253.3, -1220.9, -1211.8, -1179.7, -1131.9, -1130.3, -1107.7, -1118.0, -1101.8, -1081.9, -1085.6, -1067.6, -1062.3, -1065.2, -1042.7, -1048.4, -1049.5, -1070.4, -1062.9, -1072.7, -1098.5, -1115.1, -1153.4, -1177.5, -1191.7, -1199.2, -1201.2, -1222.1, -1237.7, -1238.5, -1251.0
- `r1_frame`: +8.7, -443226.1, -352077.0, -313719.6, -270383.4, -232314.2, -191271.5, -158996.2, -136749.8, -115642.7, -105195.0, -90786.0, -82772.1, -73724.0, -66779.0, -61676.8, -56235.1, -53228.8, -49435.4, -45928.2, -43641.3, -41336.6, -39669.5, -37280.1, -35614.9, -34733.5, -33385.2, -32717.6, -31335.2, -30488.0, -28988.0, -28446.2, -27530.3, -26522.4, -26269.2, -25518.7, -24469.2, -23904.8, -23652.9, -23289.8, -22403.7, -21798.5, -21761.6, -21608.1, -21264.3, -21176.9, -20982.6, -20587.2, -20565.3, -20208.1, -20023.6, -20135.4, -19935.9, -19825.2, -20167.0, -20194.3, -20505.0, -20806.6, -21257.0, -21575.6, -21894.0, -22370.6, -22512.3, -22347.4, -22574.1, -22636.4, -22894.5, -23233.4
- `r3_tiny`: -920605.8, -468052.6, -322273.0, -216231.0, -153898.3, -178134.7, -207095.7, -194969.5, -171277.5, -182277.9, -155213.6, -136402.9, -122728.2, -108558.4, -99382.6, -85695.4, -81460.6, -79240.1, -65112.4, -59418.6, -62049.1, -58585.2, -57987.5, -52776.7, -52515.6, -46894.6, -46606.9, -47058.3, -42313.5, -40375.8, -37138.2, -35324.8, -39196.2, -38442.9, -34993.7, -34058.9, -32858.3, -32999.6, -33027.8, -33583.1, -31432.7, -29313.2, -31066.5, -30234.7, -30374.0, -28485.9, -29983.0, -28630.8, -28772.6, -28300.7, -29369.2, -28444.6, -29168.2, -28700.4, -28831.3, -27256.5, -28397.6, -30936.6, -30330.0, -32709.3, -32011.1, -32809.4, -32501.9, -30878.9, -31764.7, -31391.1, -34104.7, -34956.3
- `r2_frame_delta_mse`: +17.3, -2075.6, -463629.1, -820572.3, -869433.0, -987350.8, -1093166.3, -1222433.9, -1381051.2, -1515302.8, -1723274.3, -1839430.8, -2043387.0, -2176295.7, -2337024.6, -2509228.1, -2659828.2, -2873937.2, -3015717.4, -3140059.4, -3337672.8, -3479668.3, -3724219.1, -3854061.6, -4025058.8, -4287552.5, -4456592.9, -4701623.3, -4859298.6, -5096517.7, -5250477.7, -5512758.1, -5701877.5, -5878670.2, -6178980.3, -6391747.4, -6476203.6, -6656711.8, -6984030.9, -7302421.5, -7458276.4, -7612045.7, -7916751.5, -8268686.7, -8561785.0, -8902060.9, -9263788.5, -9538314.6, -9894917.4, -10223904.4, -10557072.3, -10928805.4, -11383008.1, -11813887.0, -12429018.3, -12957064.9, -13535925.4, -14323610.5, -15033131.2, -15914455.5, -16693346.9, -17513684.0, -18215987.2, -18832373.1, -19610215.1, -20321002.8, -21175594.9, -22107539.7

## Training curves (subsampled)

### `r5_mse_nonorm`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.104615   0.104764    0.002702     +4.98  9.87e-04    2.6
    400   0.086522   0.088947    0.002679     +5.81  9.28e-04    2.8
    600   0.059963   0.075344    0.002094    +26.37  8.24e-04    3.0
    800   0.063131   0.066850    0.003627    -27.53  6.88e-04    3.2
   1000   0.067798   0.055702    0.002360    +17.00  5.34e-04    3.3
   1200   0.053134   0.058714    0.002179    +23.38  3.77e-04    3.5
   1400   0.051265   0.052682    0.002884     -1.42  2.34e-04    3.7
   1600   0.045522   0.051894    0.002832     +0.40  1.19e-04    3.9
   1800   0.045562   0.050923    0.002906     -2.20  4.55e-05    4.1
   2000   0.046826   0.050896    0.003086     -8.53  2.00e-05    4.6
```

### `r4_lr_sweep`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.145758   0.218926    0.123151  -4230.52  3.00e-03    2.3
    400   0.050805   0.046942    0.052885  -1759.67  2.91e-03    2.5
    600   0.030712   0.038463    0.036182  -1172.32  2.66e-03    2.7
    800   0.028461   0.039010    0.033446  -1076.10  2.27e-03    2.9
   1000   0.019657   0.021571    0.023902   -740.48  1.79e-03    3.0
   1200   0.016139   0.017238    0.033346  -1072.59  1.27e-03    3.2
   1400   0.016728   0.015429    0.033372  -1073.51  7.95e-04    3.4
   1600   0.011703   0.008689    0.039140  -1276.34  4.04e-04    3.6
   1800   0.004816   0.005481    0.043798  -1440.14  1.49e-04    3.8
   2000   0.004487   0.004455    0.048590  -1608.63  6.00e-05    4.1
```

### `r1_frame`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.055762   0.054992    0.471750 -16488.82  9.87e-04    2.2
    400   0.047915   0.039003    0.570584 -19964.25  9.28e-04    2.4
    600   0.030050   0.029666    0.544695 -19053.89  8.24e-04    2.5
    800   0.020917   0.022707    0.567719 -19863.52  6.88e-04    2.6
   1000   0.020951   0.014629    0.697120 -24413.81  5.34e-04    2.8
   1200   0.014179   0.013787    0.701749 -24576.60  3.77e-04    2.9
   1400   0.008601   0.010362    0.759619 -26611.57  2.34e-04    3.0
   1600   0.007815   0.008288    0.778237 -27266.26  1.19e-04    3.2
   1800   0.007529   0.007517    0.819871 -28730.29  4.55e-05    3.3
   2000   0.007197   0.007382    0.827749 -29007.32  2.00e-05    3.6
```

### `r3_tiny`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.039360   0.038294    1.116929 -39176.16  9.87e-04    2.5
    400   0.052186   0.055776    0.789331 -27656.38  9.28e-04    2.6
    600   0.024868   0.021918    0.638550 -22354.26  8.24e-04    2.7
    800   0.026594   0.017413    1.116602 -39164.68  6.88e-04    2.8
   1000   0.021279   0.025186    1.210763 -42475.77  5.34e-04    2.9
   1200   0.008984   0.006975    1.166004 -40901.87  3.77e-04    3.0
   1400   0.005981   0.007410    1.164870 -40861.98  2.34e-04    3.1
   1600   0.005844   0.005625    1.162947 -40794.36  1.19e-04    3.2
   1800   0.004862   0.004795    1.157949 -40618.61  4.55e-05    3.3
   2000   0.004632   0.004673    1.141897 -40054.16  2.00e-05    3.5
```

### `r2_frame_delta_mse`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    200   0.016437   0.016047   52.033219 -1829618.19  1.97e-03    2.2
    400   0.011779   0.010688  205.259380 -7217727.90  1.86e-03    2.4
    600   0.009714   0.010069  361.951615 -12727720.12  1.65e-03    2.5
    800   0.008190   0.009037  252.796019 -8889326.44  1.38e-03    2.6
   1000   0.007818   0.007347  191.160820 -6721960.16  1.07e-03    2.8
   1200   0.007294   0.007051  199.618659 -7019375.19  7.54e-04    2.9
   1400   0.006859   0.007114  244.412213 -8594514.73  4.67e-04    3.0
   1600   0.006615   0.006799  257.307227 -9047960.47  2.39e-04    3.2
   1800   0.006724   0.006727  278.926559 -9808191.82  9.10e-05    3.3
   2000   0.006459   0.006699  284.768732 -10013628.49  4.00e-05    3.6
```

## What each arm was

### `r5_mse_nonorm`

- **what**: MSE objective, feature normalisation OFF -- isolates the normalisation change.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': False, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'mse', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `r4_lr_sweep`

- **what**: Peak LR 3e-3 with a long warmup.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.003, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `r1_frame`

- **what**: Frame tokenisation: match the linear baseline's own factorisation.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'frame', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `r3_tiny`

- **what**: Deliberately tiny (E128/L2): if small beats large, this is an optimisation failure.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 128, 'N_LAYERS': 2, 'N_HEADS': 4, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

### `r2_frame_delta_mse`

- **what**: Frame + delta + MSE -- as close to the linear baseline as a net gets.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'frame', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': True, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'mse', 'LEARNING_RATE': 0.002, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 8}`

