#!/usr/bin/env bash
# Two direct follow-ups to the Branch-H grid's winner (h1_ar_freq2, +41.90%
# at 400 steps) -- see OVERVIEW.md v4.5 and ROUND2_ARMS["H"] in
# train_production_transformer_deep_dive.py.
#
# WHY THESE TWO, AND WHY SEPARATE FROM run_sweep_h300_branch_h.sh
# ==================================================================
# The 8-arm grid already ran and produced a clean, monotonic result:
# AR-loss frequency dominated every other knob tested (feedback noise, more
# AR sequences, weight decay, lower LR), with h1_ar_freq2 (every 2 steps)
# the outright winner. Re-running all 8 again would waste time re-deriving
# a conclusion already reached -- this launcher runs only the two NEW
# hypotheses built directly on that result:
#
#   h9_ar_freq1        h1's exact config (AR horizon 8, AR_SEQS 2) with the
#                      AR loss applied on EVERY step instead of every 2 --
#                      the direct extrapolation of the freq8->freq4->freq2
#                      trend, all of which favoured more frequent
#                      application. Cheap, low-risk: same mechanism, one
#                      more turn of the same knob.
#   h10_ridge_residual h1's exact config PLUS an architectural change:
#                      PREDICT_DELTA=True with DELTA_ANCHOR='ridge'. Instead
#                      of the network predicting the residual on top of raw
#                      persistence (the existing PREDICT_DELTA behaviour),
#                      it predicts the residual on top of the FITTED RIDGE
#                      MAP's prediction -- the same ridge regression that
#                      beats persistence by +69% in the same decoded-
#                      velocity units the model is scored in (OVERVIEW.md
#                      v4.5 section 5). The theory: less of the network's
#                      capacity needs to be spent re-deriving a mapping a
#                      closed-form linear regression already gets mostly
#                      right. Higher-risk than h9 -- new code path
#                      (model_variants.py's `_ridge_anchor`), smoke-tested
#                      locally on CPU across frame-aligned and mid-frame
#                      sequence lengths but never run against the real
#                      trainer/data before this launch.
#
# h10 REQUIRES diagnostics to have run in THIS sweep invocation --
# linear_frame_baseline() is what fits and saves the ridge map to
# Config.RIDGE_MAP_PATH (default: saved_models/ridge_frame_map.pt). Do NOT
# pass SKIP_DIAGNOSTICS=1 with this launcher, unless that file already
# exists on this box from a prior diagnostics run against the SAME data.
#
# Both use AR_MODE='frame_ar', so this launcher is CUDA-only, same as
# run_sweep_h300_branch_h.sh.
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
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_h300_branch_h_followup.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                          (default: python3)
#   MAX_STEPS      optimizer steps per arm                      (default: 2000)
#   VAL_EVERY      eval cadence in steps                        (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval             (default: 64)
#   SUBSET_RATIO   fraction of train data loaded                 (default: 1.0)
#   MAX_PARALLEL   concurrent arms (round-robins across GPUs)    (default: min(GPU count, 2))
#   NO_WANDB       set to 1 to skip wandb tracking                (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(h9_ar_freq1 h10_ridge_residual)
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-200}"
ROLLOUT_SEQS="${ROLLOUT_SEQS:-64}"
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
echo " H300/CUDA Branch-H follow-up -- arms: ${ARMS[*]}"
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
  echo "One or more required files are missing."
  echo "  - data/train_80.h5 / data/val_80.h5: run 'python transformer_neurIPS/prepare_data.py'"
  echo "    to regenerate on THIS box, or copy the FULL .h5 files over."
  echo "  - the scripted decoder: copy"
  echo "    encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
  echo "    from the box that trained it -- see OVERVIEW.md section 2 for provenance"
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
  echo "  Set PYTHON_BIN to the CUDA-enabled venv on this box, e.g.:"
  echo "    PYTHON_BIN=/path/to/venv/bin/python bash $0"
  exit 1
fi
echo "$VERSION_CHECK" | sed 's/^/  [OK]     /'
echo ""

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -eq 0 ]]; then
  echo "WARNING: nvidia-smi found no GPUs. Both arms use AR_MODE='frame_ar',"
  echo "which resolve_train_regime() silently disables without CUDA -- running"
  echo "this script here would produce a misleading result, not an error."
  echo "This script is meant for a CUDA box (e.g. H300/H200/B200). Aborting."
  exit 1
fi
DEFAULT_MAX_PARALLEL=$(( GPU_COUNT < ${#ARMS[@]} ? GPU_COUNT : ${#ARMS[@]} ))
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MAX_PARALLEL}"

echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "GPUs        : $GPU_COUNT detected  (nvidia-smi)"
echo "Budget      : max_steps=$MAX_STEPS val_every=$VAL_EVERY rollout_seqs=$ROLLOUT_SEQS"
echo "              subset_ratio=$SUBSET_RATIO"
echo "Concurrency : --max-parallel $MAX_PARALLEL"
echo "Diagnostics : WILL RUN (not skipped) -- h10_ridge_residual needs the ridge"
echo "              map this step fits and saves to Config.RIDGE_MAP_PATH."
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
echo "wandb       : $WANDB_STATUS"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- both arms cold-start."
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
  --val-every "$VAL_EVERY" \
  --rollout-seqs "$ROLLOUT_SEQS" \
  --subset-ratio "$SUBSET_RATIO" \
  --fresh \
  --no-warm-start \
  "${EXTRA_FLAGS[@]}"
