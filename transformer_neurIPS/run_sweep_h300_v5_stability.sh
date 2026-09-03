#!/usr/bin/env bash
# v5.0: is our best hope (h9_ar_freq1) actually stable? -- see OVERVIEW.md
# v5.0 and ROUND2_ARMS["V"] in train_production_transformer_deep_dive.py.
#
# WHY THESE FOUR
# ================
# h9_ar_freq1 (h1_ar_freq2's config with AR-loss on EVERY step) is the best
# result in this whole investigation (+43.78% at only 400 steps). Before
# scaling it further, each arm here changes exactly ONE stability-relevant
# knob against h9's exact config -- four independent axes, not four random
# guesses:
#
#   v1_h9_wd         weight decay 0.01 -> 0.05 (regularisation)
#   v2_h9_clip       gradient clip 1.0 -> 0.5  (backprop magnitude control)
#   v3_h9_moreseqs   AR_SEQS 2 -> 4            (AR-gradient noise reduction)
#   v4_h9_lrlow      LR 1e-3 -> 5e-4           (optimisation aggressiveness)
#
# A failure pattern across these identifies WHICH axis (if any) h9's win is
# fragile against, rather than just re-confirming the headline number.
#
# TIME BUDGET: wall-clock is the SAFETY NET, --max-steps sets the real pace
# ============================================================================
# v5.0's first attempt at this launcher set `--max-steps 100000` (a number
# meant to never bind) and relied entirely on `--max-hours` (a REAL enforced
# stop condition inside the trainer's own loop, not just an external
# supervisory kill-switch) to end each arm. That broke something else:
# `Config.WARMUP_FRAC` (0.03 by default) sizes the LR WARMUP as a fraction
# of `Config.MAX_STEPS` -- `h9_ar_freq1`'s own successful run used
# `MAX_STEPS=400`, giving a 12-step warmup that reached peak LR by step
# ~25. With `MAX_STEPS=100000`, the warmup became 3000 steps -- but a
# 5-minute wall-clock budget only completes ~300-350 steps at this arm
# family's measured throughput, so every v5 arm spent its ENTIRE budget at
# roughly 1/10th peak LR, still deep in warmup. The resulting catastrophic-
# looking numbers were an LR-schedule artifact, not a real stability
# signal, and every arm from that run should be discarded.
#
# Fixed: `--max-steps` is now set to a REALISTIC estimate of how many
# steps this arm family actually completes in the chosen time budget
# (measured from `h9_ar_freq1`'s own run: 0.825 s/step), so
# `WARMUP_FRAC` computes a sane warmup length matching a run that's
# actually expected to finish close to its nominal step budget.
# `--max-hours` is kept as what it was always meant to be -- a safety net
# in case a specific arm's throughput differs (e.g. AR_SEQS=4 in
# `v3_h9_moreseqs` costs more per step than the other three), not the
# primary pacing mechanism.
#
#   4+ GPUs detected  -> --max-parallel 4, --max-steps ~1090, --max-hours 0.25 (15 min) --
#                        all four run genuinely simultaneously, ~15 min wall-clock total.
#   <4 GPUs (typical)  -> --max-parallel 1, --max-steps ~360, --max-hours 0.0833 (5 min) --
#                        four arms run back-to-back, ~20 min wall-clock total.
#
# WANDB: grouped, colour-coded comparison across all four runs
# ================================================================
# `Config.WANDB_GROUP` (new in v5.0) is set to "v5_stability" here so
# wandb's native multi-run comparison view groups these four runs
# together -- each gets its own auto-assigned colour in every shared
# chart. The trainer's per-eval wandb payload was also extended (v5.0) to
# include sampled per-frame improvement values
# (`frame_improvement/f00`, `f08`, `f16`, ...) and `rollout_rmse_mps`,
# not just the 3 fixed points (frame1/half/last) it logged before -- so
# each run's rollout SHAPE, not just its final number, is visible as it
# trains, across all four colour-coded runs at once.
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# run_sweep_h300_branch_h.sh's list:
#   transformer_neurIPS/train_production_transformer_deep_dive.py
#   transformer_neurIPS/model_variants.py
#   transformer_neurIPS/sweep_deep_dive.py
#   transformer_neurIPS/data/train_80.h5   (or a make_sweep_sample_data.py
#   transformer_neurIPS/data/val_80.h5      sample renamed to these exact
#                                            names -- see OVERVIEW.md 20.6)
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt
#
# If these are already on the box from the prior h9/h10 run, only
# train_production_transformer_deep_dive.py (new arms + wandb changes),
# sweep_deep_dive.py (--max-hours default + --set fix, from OVERVIEW.md
# v4.7 -- re-sync if not already done), and this launcher need uploading.
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_h300_v5_stability.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN      interpreter to use                       (default: python3)
#   MAX_STEPS       optimizer steps per arm                    (default: 1090 if 4+ GPUs, else 360 -- sized from h9's own measured throughput; override if your box's throughput differs)
#   ROLLOUT_SEQS    sequences scored per rollout eval          (default: 64)
#   SUBSET_RATIO    fraction of train data loaded              (default: 1.0)
#   VAL_EVERY       eval cadence in steps                      (default: 100 -- tighter than the usual 200, so a short wall-clock-bounded run still gets several eval points to show a trend, not just one)
#   NO_WANDB        set to 1 to skip wandb tracking             (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(v1_h9_wd v2_h9_clip v3_h9_moreseqs v4_h9_lrlow)
ROLLOUT_SEQS="${ROLLOUT_SEQS:-64}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
VAL_EVERY="${VAL_EVERY:-100}"
NO_WANDB="${NO_WANDB:-0}"

REQUIRED_FILES=(
  "$HERE/train_production_transformer_deep_dive.py"
  "$HERE/model_variants.py"
  "$HERE/sweep_deep_dive.py"
  "$HERE/data/train_80.h5"
  "$HERE/data/val_80.h5"
  "$REPO_ROOT/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
)

echo "=================================================================="
echo " v5.0 stability check -- arms: ${ARMS[*]}"
echo "=================================================================="
echo ""
echo "Checking required files..."
missing=0
for f in "${REQUIRED_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    size="$(du -h "$f" 2>/dev/null | cut -f1)"
    printf "  [OK]      %-95s (%s)\n" "$f" "$size"
  else
    printf "  [MISSING] %s\n" "$f"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo ""
  echo "One or more required files are missing -- see OVERVIEW.md 20.5/20.6"
  echo "for how to produce them."
  exit 1
fi
echo ""
echo "Checking interpreter has the required packages (torch+cuda, h5py, numpy)..."
set +e
VERSION_CHECK="$("$PYTHON_BIN" -c "
import torch, h5py, numpy
print('torch', torch.__version__, 'cuda_available=' + str(torch.cuda.is_available()))
print('numpy', numpy.__version__)
print('h5py', h5py.__version__)
try:
    import wandb
    print('wandb', wandb.__version__)
except ImportError:
    print('wandb NOT INSTALLED (fine if you pass --no-wandb / set NO_WANDB=1)')
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
" 2>&1)"
VERSION_CHECK_RC=$?
set -e
if [[ "$VERSION_CHECK_RC" -ne 0 ]]; then
  echo "  [MISSING] $PYTHON_BIN failed the import/CUDA check:"
  echo "$VERSION_CHECK" | sed 's/^/    /'
  exit 1
fi
echo "$VERSION_CHECK" | sed 's/^/  [OK]     /'
echo ""

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -eq 0 ]]; then
  echo "WARNING: nvidia-smi found no GPUs. All four arms use AR_MODE='frame_ar',"
  echo "which resolve_train_regime() silently disables without CUDA -- running"
  echo "this script here would produce a misleading result, not an error."
  echo "This script is meant for a CUDA box. Aborting."
  exit 1
fi

# Step counts sized from h9_ar_freq1's own measured throughput (0.825
# s/step) for the chosen wall-clock window, so Config.WARMUP_FRAC (0.03)
# produces a warmup length matched to a run actually expected to finish
# near this budget -- NOT an oversized sentinel relying on --max-hours
# alone (see the header comment: that broke the LR schedule in v5.0's
# first attempt). --max-hours stays on as a safety net in case a specific
# arm (e.g. v3_h9_moreseqs's AR_SEQS=4) costs more per step than measured.
if [[ "$GPU_COUNT" -ge 4 ]]; then
  MAX_PARALLEL=4
  MAX_STEPS="${MAX_STEPS:-1090}"
  MAX_HOURS="0.25"
  MODE_DESC="4+ GPUs detected -- all four arms run SIMULTANEOUSLY, 15 min each, ~15 min wall-clock total"
else
  MAX_PARALLEL=1
  MAX_STEPS="${MAX_STEPS:-360}"
  MAX_HOURS="0.0833333"
  MODE_DESC="$GPU_COUNT GPU(s) detected -- arms run SEQUENTIALLY, 5 min each, ~20 min wall-clock total"
fi

if [[ "$NO_WANDB" == "1" ]]; then
  WANDB_STATUS="disabled (--no-wandb)"
else
  WANDB_PROJECT_LIVE="$("$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$HERE')
from train_production_transformer_deep_dive import Config
print(Config.WANDB_PROJECT)
" 2>/dev/null || echo "<could not read Config.WANDB_PROJECT>")"
  WANDB_STATUS="enabled -- project $WANDB_PROJECT_LIVE, group v5_stability"
fi

echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "GPUs        : $GPU_COUNT detected  (nvidia-smi)"
echo "Mode        : $MODE_DESC"
echo "Budget      : max_steps=$MAX_STEPS (sized from h9's measured throughput so"
echo "              WARMUP_FRAC produces a sane warmup -- see header comment)."
echo "              --max-hours $MAX_HOURS is a SAFETY NET, checked inside the"
echo "              training loop every step, in case actual throughput differs."
echo "wandb       : $WANDB_STATUS"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- every arm cold-starts."
echo ""

EXTRA_FLAGS=()
if [[ "$NO_WANDB" == "1" ]]; then
  EXTRA_FLAGS+=(--no-wandb)
fi

cd "$HERE"
exec "$PYTHON_BIN" sweep_deep_dive.py \
  --arms "${ARMS[@]}" \
  --round 2 \
  --max-parallel "$MAX_PARALLEL" \
  --max-steps "$MAX_STEPS" \
  --max-hours "$MAX_HOURS" \
  --val-every "$VAL_EVERY" \
  --rollout-seqs "$ROLLOUT_SEQS" \
  --subset-ratio "$SUBSET_RATIO" \
  --skip-diagnostics \
  --fresh \
  --no-warm-start \
  --set "WANDB_GROUP=v5_stability" \
  "${EXTRA_FLAGS[@]}"
