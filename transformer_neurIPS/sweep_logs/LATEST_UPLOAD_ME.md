# Deep-dive sweep report -- round 2  (`round2_20260901_162228`)

## VERDICT

**Branch `N`: Models are far below a trivial baseline -- fix conditioning before anything else**

- best arm = e6_sched_noise at -3454.62% vs persistence
- ridge linear frame-map baseline = +60.64%
- beat the previous-frame anchor on the training objective: 0/2
- No arm beat the previous-frame anchor even in-sample, and the best rollout is -3454.6% (far below parity). The models are broken, not the task.
- e6_sched_noise: best train 0.01172 vs anchor floor 0.0074004
- a5b_wd_heavy: best train 0.012531 vs anchor floor 0.0074004

Recommended next command:

```
python sweep_deep_dive.py --round 2 --branch N --max-parallel 1 --max-steps 400
```

which runs:

- `n1_delta_mse` -- Delta + MSE + LR 2e-3: the combination that removes both the output-scale and the objective-geometry problems.
- `n2_meta_off` -- Zero the (x, y, z, t, param) input columns entirely -- position still arrives via the embeddings.
- `n3_lr_low` -- LR 1e-4: rule out silent divergence at 1e-3.
- `n4_frame_delta` -- Frame tokenisation + delta + MSE: the smallest, most directly supervised version of the problem.
- `n5_tiny` -- E128/L2 + delta + MSE: if a tiny model works and a big one does not, this is an optimisation failure, not capacity.

## Run settings

```
run_id             = round2_20260901_162228
round              = 2
max_parallel       = 1
max_steps          = 400
max_hours          = 12.0
seed               = 1337
subset_ratio       = 0.3
rollout_seqs       = 16
gpus               = 0
started            = 2026-09-01 16:22:28
finished           = 2026-09-01 18:52:14
trainer_git_head   = 27ffbb2
```

## Diagnostics (run once, before any training)

```
torch            = 2.12.0.dev20260312
device           = mps  gpu = None xNone  bf16 = None
train sequences  = 17784
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
symmetric_padding_1 (old)    max change in PAST outputs from a FUTURE perturbation = 2.527e+00
left_padding_2 (fixed)       max change in PAST outputs from a FUTURE perturbation = 0.000e+00
```
Non-zero for `symmetric_padding_1 (old)` confirms `padding=1` with `kernel_size=3` let every token see t+1, once per block.

### 3. Causality of each configuration we are about to train

```
a0_control       causal=True  before_cut=0.000e+00 after_cut=3.064e+00
a1_nonorm        causal=True  before_cut=0.000e+00 after_cut=1.913e+00
a2_mse           causal=True  before_cut=0.000e+00 after_cut=2.730e+00
a3_delta         causal=True  before_cut=0.000e+00 after_cut=7.000e+00
a4_frame         causal=True  before_cut=0.000e+00 after_cut=2.706e+00
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
persistence MSE                = 0.0003455801863559845
ridge linear frame-map MSE     = 0.00013600662014857449
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
e6_sched_noise ok            400   56.1    4.79    -3454.62   -2.82e+05    -2787.47    0.114964    0.003234   0.010321   0.011916      yes      NO
a5b_wd_heavy   ok            400   92.9    4.79    -5572.84   -3.61e+05    -5087.33    0.183472    0.003234   0.010904   0.012647      yes      NO
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 68 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = -361088.3%   '@' = -2200.6%

e6_sched_noise |_+##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%@%%%%%%%%%%%%%%%%%%|  f1=-2.82e+05  f28=-2787.47
a5b_wd_heavy   |.=*##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%|  f1=-3.61e+05  f28=-5087.33
```

- `e6_sched_noise`: -281981.6, -146488.2, -82933.3, -52783.7, -46009.1, -34980.3, -25149.7, -21616.6, -17714.5, -14776.6, -13883.1, -12417.4, -10330.5, -8811.5, -8022.8, -7309.0, -6966.4, -6650.9, -6137.8, -5488.3, -5326.2, -4527.9, -4353.0, -3997.1, -3838.4, -3719.2, -3385.2, -3445.7, -3266.2, -3426.7, -2944.7, -2892.3, -2892.5, -2579.7, -2720.0, -2549.8, -2625.0, -2377.3, -2579.9, -2483.3, -2431.7, -2380.5, -2308.2, -2339.1, -2328.5, -2322.3, -2374.9, -2300.6, -2324.7, -2200.6, -2250.6, -2374.4, -2334.6, -2424.8, -2424.2, -2358.2, -2465.0, -2567.1, -2568.9, -2591.4, -2601.1, -2702.9, -2765.4, -2801.3, -2836.8, -2848.7, -2808.5, -2787.5
- `a5b_wd_heavy`: -361088.3, -189431.4, -107240.4, -72250.2, -61207.3, -46286.8, -35140.3, -30293.2, -25244.7, -20958.3, -19344.7, -17669.4, -15799.8, -13872.5, -11961.4, -10940.7, -10999.4, -10015.5, -9405.9, -8186.6, -7516.8, -7300.2, -6761.6, -6215.1, -5877.6, -5773.5, -5479.6, -5317.5, -5161.1, -5108.5, -4838.3, -4701.5, -4649.7, -4387.0, -4339.4, -4299.5, -4337.3, -4213.5, -4177.7, -4139.8, -4099.4, -3909.8, -3871.5, -3906.5, -3930.6, -3998.8, -4110.7, -4033.4, -3932.4, -4127.8, -4010.0, -3973.5, -4038.8, -4002.6, -4148.2, -4148.6, -4291.8, -4350.9, -4332.5, -4616.6, -4806.6, -4953.4, -5018.0, -4930.3, -4897.0, -4976.8, -5091.9, -5087.3

## Training curves (subsampled)

### `e6_sched_noise`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.104213   0.132878    0.143863  -4348.18  8.81e-04   16.2
    200   0.030133   0.042843    0.095057  -2839.11  5.34e-04   32.3
    300   0.017294   0.018467    0.135586  -4092.24  1.72e-04   43.4
    400   0.011916   0.010321    0.114964  -3454.62  2.00e-05   56.1
```

### `a5b_wd_heavy`

```
   step      train     val_tf    roll MSE   IMPROV%        lr    min
    100   0.072958   0.056826    0.178376  -5415.29  8.81e-04   21.1
    200   0.041651   0.053934    0.136724  -4127.43  5.34e-04   37.1
    300   0.028205   0.018338    0.169054  -5127.06  1.72e-04   64.8
    400   0.012647   0.010904    0.183472  -5572.84  2.00e-05   92.9
```

## What each arm was

### `e6_sched_noise`

- **what**: Scheduled sampling + noise on the fed-back prediction itself (not ground truth): simulates 'the thing you're about to condition on is already wrong', not just generic input noise. sched's 2-forward cost (no sequential chain) makes this the only AR-family arm that runs on MPS/CPU unmodified -- see resolve_train_regime().
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'sched', 'AR_LOSS_WEIGHT': 1.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.005, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.01, 'WEIGHT_DECAY': 0.01, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 4}`

### `a5b_wd_heavy`

- **what**: Weight decay 0.1 + dropout 0.1: damp the amplifying modes.
- **config**: `{'VARIANT': 'base', 'TOKENIZATION': 'token', 'EMBED_SIZE': 256, 'N_LAYERS': 6, 'N_HEADS': 8, 'PREDICT_DELTA': False, 'NORMALIZE_FEATURES': True, 'USE_ROPE': False, 'NOISE_STD': 0.0005, 'AR_MODE': 'none', 'AR_LOSS_WEIGHT': 0.0, 'AR_FRAMES': 2, 'AR_FEEDBACK_NOISE_STD': 0.0, 'LOSS': 'l2norm', 'LEARNING_RATE': 0.001, 'DROPOUT': 0.1, 'WEIGHT_DECAY': 0.1, 'BATCH_SIZE': 64, 'ACCUMULATION_STEPS': 4}`

