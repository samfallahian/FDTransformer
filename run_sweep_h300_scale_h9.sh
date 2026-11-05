#!/usr/bin/env bash
# Single-arm follow-up to h9_ar_freq1's win -- see OVERVIEW.md v4.6 and
# ROUND2_ARMS["S"]["s7_h9_scaled"] in train_production_transformer_deep_dive.py.
#
# WHY THIS EXISTS, AND WHY NO STEP COUNT IS BAKED INTO THE ARM
# ===============================================================
# h9_ar_freq1 (h1_ar_freq2's config with AR-loss on EVERY step) posted
# +43.78% at only 400 steps -- the best result in this entire investigation,
# beating e3_ar_long, s6_e3_scaled, and h1_ar_freq2 itself. h10_ridge_residual
# (the architectural change tested alongside it) failed catastrophically for
# a real, verified-non-buggy reason (an expansive linear anchor compounding
# under AR feedback, unlike persistence's non-expansive copy) and is
# abandoned, not pursued further here.
#
# Unlike s6_e3_scaled (which baked MAX_STEPS=12000 into the arm itself),
# this arm bakes NO step count: AR_EVERY_N_STEPS=1 is the most expensive AR
# frequency tried in this whole investigation (measured at 0.825s/step,
# 400 steps -> 5.5 min on this exact box), so the right budget depends on
# actual available wall-clock, not a fixed convention. This launcher's
# MAX_STEPS default is sized for a specific ~30-minute time budget --
# override it via the environment variable below for a different budget.
#
# TIME BUDGET MATH (edit MAX_STEPS if your budget differs)
# ============================================================
#   measured rate           : 0.825 s/step (h9_ar_freq1, 400 steps, 5.5 min)
#   diagnostics             : SKIPPED by default (see below) -- this arm
#                              doesn't need the ridge map (that was only for
#                              the now-abandoned h10_ridge_residual), and
#                              diagnostics has its own real cost even after
#                              the max_val_seqs/val_chunk speedups.
#   budget                  : ~30 minutes = 1800s, minus ~60s launch/eval
#                              overhead = ~1740s of training
#   MAX_STEPS default       : 2000  (2000 * 0.825s = ~1650s = 27.5 min,
#                              leaving a margin for checkpoint saves and
#                              the final rollout eval)
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# run_sweep_h200_scale_e3.sh's list:
#   transformer_neurIPS/train_production_transformer_deep_dive.py
#   transformer_neurIPS/model_variants.py
#   transformer_neurIPS/sweep_deep_dive.py
#   transformer_neurIPS/data/train_80.h5   (or a make_sweep_sample_data.py
#   transformer_neurIPS/data/val_80.h5      sample renamed to these exact
#                                            names -- see OVERVIEW.md 20.6)
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt
#
# If train_80.h5/val_80.h5/the scripted decoder are ALREADY on the box from
# the prior run_sweep_h300_branch_h_followup.sh launch, nothing new needs
# uploading -- only the three edited Python files below need to be re-synced
# before running this (this launcher script itself is new too):
#   transformer_neurIPS/train_production_transformer_deep_dive.py  (new arm)
#   transformer_neurIPS/run_sweep_h300_scale_h9.sh                 (new file)
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_h300_scale_h9.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN         interpreter to use                      (default: python3)
#   MAX_STEPS          optimizer steps                          (default: 2000, sized for a ~30 min budget -- see math above)
#   VAL_EVERY          eval cadence in steps                    (default: 200)
#   ROLLOUT_SEQS       sequences scored per rollout eval         (default: 64)
#   SUBSET_RATIO       fraction of train data loaded             (default: 1.0)
#   SKIP_DIAGNOSTICS   set to 0 to re-run diagnostics             (default: 1, i.e. skipped -- this arm doesn't need the ridge map)
#   NO_WANDB           set to 1 to skip wandb tracking            (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(s7_h9_scaled)
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-200}"
ROLLOUT_SEQS="${ROLLOUT_SEQS:-64}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
SKIP_DIAGNOSTICS="${SKIP_DIAGNOSTICS:-1}"
NO_WANDB="${NO_WANDB:-0}"
# sweep_deep_dive.py's own --max-hours is a stuck-job kill-switch, separate
# from --max-steps. Its own default is now 0.25h (15 min, see OVERVIEW.md
# v4.7) -- set explicitly here too so this launcher's printed value always
# matches what actually gets passed, rather than relying on an implicit
# default that could drift independently of this script.
MAX_HOURS="${MAX_HOURS:-0.25}"

REQUIRED_FILES=(
  "$HERE/train_production_transformer_deep_dive.py"
  "$HERE/model_variants.py"
  "$HERE/sweep_deep_dive.py"
  "$HERE/data/train_80.h5"
  "$HERE/data/val_80.h5"
  "$REPO_ROOT/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
)

echo "=================================================================="
echo " H300/CUDA production-scale follow-up -- arm: ${ARMS[*]}"
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
  echo "WARNING: nvidia-smi found no GPUs. s7_h9_scaled uses AR_MODE='frame_ar',"
  echo "which resolve_train_regime() silently disables without CUDA -- running"
  echo "this script here would produce a misleading result, not an error."
  echo "This script is meant for a CUDA box. Aborting."
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
echo "Budget      : max_steps=$MAX_STEPS val_every=$VAL_EVERY rollout_seqs=$ROLLOUT_SEQS"
echo "              subset_ratio=$SUBSET_RATIO"
echo "              -- sized for ~30 min wall-clock at this arm's measured"
echo "              0.825 s/step; override MAX_STEPS if your budget differs."
echo "Kill switch : --max-hours $MAX_HOURS (a SEPARATE stuck-job safety net,"
echo "              not the expected duration -- sweep_deep_dive.py's own"
echo "              startup log below will print this as its 'wall safety"
echo "              net', distinct from the max_steps budget above)"
if [[ "$SKIP_DIAGNOSTICS" == "1" ]]; then
  echo "Diagnostics : SKIPPED (default) -- this arm doesn't need the ridge map,"
  echo "              only h10_ridge_residual (abandoned) did."
fi
echo "wandb       : $WANDB_STATUS"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- cold start, consistent with"
echo "              h9_ar_freq1's own run."
echo ""

EXTRA_FLAGS=()
if [[ "$NO_WANDB" == "1" ]]; then
  EXTRA_FLAGS+=(--no-wandb)
fi
if [[ "$SKIP_DIAGNOSTICS" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-diagnostics)
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
  "${EXTRA_FLAGS[@]}"
