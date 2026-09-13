#!/usr/bin/env bash
# The v5.0 stability round's production follow-up -- see OVERVIEW.md v5.5
# and ROUND2_ARMS["S"]["s8_h9_moreseqs_scaled"] in
# train_production_transformer_deep_dive.py.
#
# WHY THIS ARM
# ==============
# Four independent perturbations were tested against h9_ar_freq1 (weight
# decay, gradient clip, AR_SEQS, LR) -- all four held within a ~5-point
# band of h9's own +43.78%, confirming the win is stable, not fragile
# (OVERVIEW.md v5.0 section 28.7). One of them, v3_h9_moreseqs
# (AR_SEQS 2->4, less gradient noise per AR-loss application), didn't
# just hold -- it edged out h9 itself at +46.49%. This launcher scales
# THAT config, not h9's own unmodified one, to a real production budget --
# same "scale the winner, not the original" pattern as
# e3_ar_long -> s6_e3_scaled and h1_ar_freq2 -> h9_ar_freq1.
#
# THIS IS A PRODUCTION RUN, NOT A SWEEP-ARM SCREEN -- SEVERAL v4.7/v5.0
# SWEEP DEFAULTS ARE DELIBERATELY OVERRIDDEN BACK ON
# ========================================================================
# `arm_command()` (sweep_deep_dive.py) defaults every arm to
# `SAVE_SCRIPTED_MODELS=False` and `CHECKPOINT_EVERY_STEPS=<max_steps>`
# (OVERVIEW.md v4.7) -- sensible for a 5-30 minute throwaway screen, wrong
# for an 8-hour run meant to produce a real, deployable, resumable
# artifact:
#   - SAVE_SCRIPTED_MODELS=True   -- a deployable TorchScript export is
#                                    the actual point of a production run.
#   - CHECKPOINT_EVERY_STEPS=2000 -- resumable "latest" snapshots every
#                                    ~17 minutes of wall-clock, not just
#                                    once at the very end. A crash 7 hours
#                                    in without this loses everything.
# Both are passed as explicit --set overrides below, listed AFTER
# arm_command()'s own defaults -- the trainer's --set loop is
# last-value-wins, so these two intentionally take precedence.
#
# Diagnostics are NOT skipped this time either (unlike the v5.0 stability
# launcher) -- an 8-hour commitment is worth a fresh one-time ridge-
# baseline reconfirmation first (a minute or two, negligible against 8
# hours), not worth skipping to save time the way a 5-minute arm needed to.
#
# STEP BUDGET: measured from v3_h9_moreseqs's own steady-state throughput
# ==========================================================================
# From v3_h9_moreseqs.log (the v5.0 stability run): step 125 at 3.4m,
# step 300 at 4.9m -- a clean stretch with no slow first-eval or compile
# warmup included -- gives 0.514 s/step steady-state. At an assumed ~15
# minutes of combined startup/eval/checkpoint overhead across an 8-hour
# budget, that works out to roughly 54,000 steps at the measured rate;
# MAX_STEPS is set to 45,000 here, deliberately below that raw estimate,
# for margin against measurement noise and the slightly higher per-step
# cost of re-enabled scripted-checkpoint compiles. Unlike the v5.0
# stability launcher's failure mode (OVERVIEW.md v5.0 section 28.6, where
# an oversized MAX_STEPS broke the LR warmup on a 5-MINUTE run), being
# on the generous side here is low-risk: WARMUP_FRAC=0.03 x 45,000 = 1350
# steps, ~12 minutes at this rate -- a small fraction of an 8-hour budget
# either way. If actual throughput undershoots and --max-hours triggers
# first, the run simply stops with a somewhat-less-annealed LR rather
# than stuck deep in warmup -- a graceful degradation, not the same class
# of bug.
#
# --max-hours 8.0 is kept as the real, enforced backstop (checked inside
# the training loop every step, not just an external supervisory kill --
# see OVERVIEW.md v5.0 section 28.2) in case actual throughput differs
# from the estimate above.
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# every other run_sweep_h300_*.sh launcher's list:
#   transformer_neurIPS/train_production_transformer_deep_dive.py
#   transformer_neurIPS/model_variants.py
#   transformer_neurIPS/sweep_deep_dive.py
#   transformer_neurIPS/data/train_80.h5
#   transformer_neurIPS/data/val_80.h5
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt
#
# Only train_production_transformer_deep_dive.py (new s8 arm) needs
# re-syncing if the rest are already on the box from the v5.0 run --
# this launcher itself is also new.
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_h300_production_8h.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN         interpreter to use                    (default: python3)
#   MAX_STEPS           optimizer steps                        (default: 45000 -- see step-budget math above)
#   MAX_HOURS           wall-clock safety net                  (default: 8.0)
#   VAL_EVERY           eval cadence in steps                  (default: 2000 -- ~20 eval points over the run, not the 100-step cadence a 5-min screen needed)
#   ROLLOUT_SEQS        sequences scored per rollout eval       (default: 128 -- doubled from the sweep-arm default of 64; an 8-hour run affords a more statistically solid final read)
#   CHECKPOINT_EVERY    "latest" checkpoint cadence in steps    (default: 2000, matching VAL_EVERY -- real resumability, not v4.7's throwaway-sweep "once at the end")
#   SUBSET_RATIO        fraction of train data loaded           (default: 1.0)
#   NO_WANDB            set to 1 to skip wandb tracking          (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(s8_h9_moreseqs_scaled)
MAX_STEPS="${MAX_STEPS:-45000}"
MAX_HOURS="${MAX_HOURS:-8.0}"
VAL_EVERY="${VAL_EVERY:-2000}"
ROLLOUT_SEQS="${ROLLOUT_SEQS:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
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
echo " v5.5 PRODUCTION run (8h target) -- arm: ${ARMS[*]}"
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
  echo "WARNING: nvidia-smi found no GPUs. s8_h9_moreseqs_scaled uses"
  echo "AR_MODE='frame_ar', which resolve_train_regime() silently disables"
  echo "without CUDA -- running this script here would produce a"
  echo "misleading result, not an error. This script needs a CUDA box."
  echo "Aborting."
  exit 1
fi

if [[ "$NO_WANDB" == "1" ]]; then
  WANDB_STATUS="disabled (--no-wandb)"
else
  WANDB_PROJECT_LIVE="$("$PYTHON_BIN" -c "
import sys; sys.path.insert(0, '$HERE')
from train_production_transformer_deep_dive import Config
print(Config.WANDB_PROJECT)
" 2>/dev/null || echo "<could not read Config.WANDB_PROJECT>")"
  WANDB_STATUS="enabled -- project $WANDB_PROJECT_LIVE"
fi

echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "GPUs        : $GPU_COUNT detected  (nvidia-smi)"
echo "Budget      : max_steps=$MAX_STEPS (see header comment for the"
echo "              measured-throughput math) -- --max-hours $MAX_HOURS is"
echo "              the real enforced backstop."
echo "Eval        : every $VAL_EVERY steps, rollout_seqs=$ROLLOUT_SEQS"
echo "Checkpoints : SAVE_SCRIPTED_MODELS=True, CHECKPOINT_EVERY_STEPS=$CHECKPOINT_EVERY"
echo "              (production overrides of the v4.7 sweep-arm defaults --"
echo "              see header comment)"
echo "Diagnostics : WILL RUN (not skipped) -- worth a fresh reconfirmation"
echo "              before an 8-hour commitment."
echo "wandb       : $WANDB_STATUS"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- cold start."
echo ""

EXTRA_FLAGS=()
if [[ "$NO_WANDB" == "1" ]]; then
  EXTRA_FLAGS+=(--no-wandb)
fi

cd "$HERE"
exec "$PYTHON_BIN" sweep_deep_dive.py \
  --arms "${ARMS[@]}" \
  --round 2 \
  --max-parallel 1 \
  --max-steps "$MAX_STEPS" \
  --max-hours "$MAX_HOURS" \
  --val-every "$VAL_EVERY" \
  --rollout-seqs "$ROLLOUT_SEQS" \
  --subset-ratio "$SUBSET_RATIO" \
  --fresh \
  --no-warm-start \
  --set "SAVE_SCRIPTED_MODELS=True" "CHECKPOINT_EVERY_STEPS=$CHECKPOINT_EVERY" \
  "${EXTRA_FLAGS[@]}"
