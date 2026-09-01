# Deep-dive sweep report -- round 2  (`round2_smoke_20260901_154411`)

## VERDICT

**Branch `L`: Causality probe FAILED -- do not interpret any metric until this is fixed**

- No arm produced a rollout score at all; check the per-arm logs.

No Round 2 arms: fix the blocker above first, then re-run Round 1.

## Run settings

```
run_id             = round2_smoke_20260901_154411
round              = 2
max_parallel       = 5
max_steps          = 60
max_hours          = 1.0
seed               = 1337
subset_ratio       = 0.1
rollout_seqs       = 8
gpus               = 0
started            = 2026-09-01 15:44:11
finished           = 2026-09-01 15:44:46
trainer_git_head   = 27ffbb2
```

## Diagnostics (run once, before any training)

_Diagnostics did not complete -- see `diagnostics.log`._

## Arm results

```
arm            status      steps   mins  params     IMPROV%     frame1%       last%    roll MSE    pers MSE     val tf      train  >const?  >anch?
--------------------------------------------------------------------------------------------------------------------------------------------------
e6_sched_noise exit 1          -    0.0    --          --          --          --          --          --         --         --          -       -
a5b_wd_heavy   exit 1          -    0.0    --          --          --          --          --          --         --         --          -       -
```

`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the full 28-frame horizon, with model and baseline scored on the SAME validation rows. Positive means better than doing nothing.

`>const?` is the sanity floor: did this arm's training loss beat the best CONSTANT predictor? `NO` means the arm learned nothing at all and its `IMPROV%` is not worth interpreting. `>anch?` is the same question against copying the previous time frame -- that anchor IS the persistence baseline expressed in the training objective, so an arm that cannot beat it in-sample cannot beat persistence in rollout.

Note: `train` and `val tf` are not comparable between token-level and frame-level arms -- their loss is a norm over 47 and 1222 dimensions respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all arms, which is why the ranking uses them.

## Improvement by rollout horizon

One row per arm, 0 predicted time frames left to right. This is the shape that tells you whether the model predicts well and then drifts, or never predicted well.

```
scale: '.' = +0.0%   '@' = +1.0%

```


## Training curves (subsampled)

## What each arm was

### `e6_sched_noise`

- **what**: Scheduled sampling + noise on the fed-back prediction itself (not ground truth): simulates 'the thing you're about to condition on is already wrong', not just generic input noise. sched's 2-forward cost (no sequential chain) makes this the only AR-family arm that runs on MPS/CPU unmodified -- see resolve_train_regime().
- **status**: exit 1 -- see `e6_sched_noise.log`

### `a5b_wd_heavy`

- **what**: Weight decay 0.1 + dropout 0.1: damp the amplifying modes.
- **status**: exit 1 -- see `a5b_wd_heavy.log`

## Failures

- `e6_sched_noise`: exit 1 (tail of `e6_sched_noise.log`)

```
$ CUDA_VISIBLE_DEVICES=0 /Users/kkreth/PycharmProjects/cgan_last_venv_ever/bin/python -u /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py --arm e6_sched_noise --round 2 --out-dir /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/sweep_logs/round2_smoke_20260901_154411 --max-steps 60 --max-hours 1.0 --val-every 30 --seed 1337 --rollout-seqs 8 --subset-ratio 0.1 --no-wandb --fresh --set VAL_CONTEXT_STEPS=32

[DEVICE DETECTED] MPS (Apple Silicon)
🌈 MICRO-BATCH MODE (micro_batch=1)
WHY: on MPS/CPU the caching allocator cannot reuse blocks across the changing shapes of an autoregressive rollout, so peak memory scales with the batch inside ONE rollout call and blows past the device ceiling on the very first batch. micro_batch=1 keeps each forward within budget (same rationale documented in persistence_formal_documentation.py).
WHAT CHANGES ON CUDA: micro_batch bumps to 32 (64 on H200), gradient accumulation collapses to 1, AMP bf16 turns on, torch.compile is attempted, cudnn.benchmark is enabled.
  [data] train=5,928/59,280 sequences  val=25,410
  [data] 'train': (5928, 800, 52) (0.99 GB float32) -> mps
  [data]   resident on mps after 0.1s
  [data] 'val': (25410, 800, 52) (4.23 GB float32) -> mps
  [data]   resident on mps after 0.4s
  [model] variant=base tokenization=token E=256 L=6 H=8 params=4.79M attn_impl=sdpa delta=False rope=False
  [model] feature stats installed: mean|max|=39.500 std range [0.009831, 41.74]
[warm-start] checkpoint not found: /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/saved_models/r1_a3b_delta_ar_rollout_best.pt. Pass --warm-start PATH or --no-warm-start.
```

- `a5b_wd_heavy`: exit 1 (tail of `a5b_wd_heavy.log`)

```
$ CUDA_VISIBLE_DEVICES=0 /Users/kkreth/PycharmProjects/cgan_last_venv_ever/bin/python -u /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py --arm a5b_wd_heavy --round 2 --out-dir /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/sweep_logs/round2_smoke_20260901_154411 --max-steps 60 --max-hours 1.0 --val-every 30 --seed 1337 --rollout-seqs 8 --subset-ratio 0.1 --no-wandb --fresh --set VAL_CONTEXT_STEPS=32

[DEVICE DETECTED] MPS (Apple Silicon)
🌈 MICRO-BATCH MODE (micro_batch=1)
WHY: on MPS/CPU the caching allocator cannot reuse blocks across the changing shapes of an autoregressive rollout, so peak memory scales with the batch inside ONE rollout call and blows past the device ceiling on the very first batch. micro_batch=1 keeps each forward within budget (same rationale documented in persistence_formal_documentation.py).
WHAT CHANGES ON CUDA: micro_batch bumps to 32 (64 on H200), gradient accumulation collapses to 1, AMP bf16 turns on, torch.compile is attempted, cudnn.benchmark is enabled.
  [data] train=5,928/59,280 sequences  val=25,410
  [data] 'train': (5928, 800, 52) (0.99 GB float32) -> mps
  [data]   resident on mps after 0.1s
  [data] 'val': (25410, 800, 52) (4.23 GB float32) -> mps
  [data]   resident on mps after 0.4s
  [model] variant=base tokenization=token E=256 L=6 H=8 params=4.79M attn_impl=sdpa delta=False rope=False
  [model] feature stats installed: mean|max|=39.500 std range [0.009831, 41.74]
[warm-start] checkpoint not found: /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/saved_models/r1_a3b_delta_ar_rollout_best.pt. Pass --warm-start PATH or --no-warm-start.
```

