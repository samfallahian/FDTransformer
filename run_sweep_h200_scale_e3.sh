#!/usr/bin/env bash
# Single-arm, production-scale follow-up to run_sweep_h200.sh's shallow
# sweep -- see OVERVIEW.md v4.3.
#
# WHY THIS EXISTS, SEPARATE FROM run_sweep_h200.sh
# ==================================================
# The shallow sweep (2000 steps, run_sweep_h200.sh) found a real winner:
# e3_ar_long posted +30.77% average improvement vs. persistence over the
# full 68-frame rollout -- the first arm in this entire investigation to
# post a POSITIVE number there. sweep_deep_dive.py's own auto-classifier
# recommended a generic "Branch S: scale and refine" follow-up
# (s1_capacity_xl, s2_steps_3x, etc.), but those apply their scaling knobs
# to the CONTROL config's settings, not to e3_ar_long's AR mechanism --
# running them as suggested would scale the wrong starting point and lose
# the thing that actually won.
#
# This script runs `s6_e3_scaled` instead: a new arm (ROUND2_ARMS["S"])
# that is e3_ar_long's EXACT config (AR_MODE=frame_ar, AR_FRAMES=8,
# AR_SEQS=2, AR_EVERY_N_STEPS=8, AR_LOSS_WEIGHT=1.0), at a production-scale
# step budget (12000 by default -- comparable to the original
# a3b_delta_ar run that first exhibited the catastrophic rollout
# divergence this whole investigation started from) instead of the
# shallow sweep's 2000. The open question this answers: does the +30.77%
# / stable-plateau-from-frame~17-onward shape (see OVERVIEW.md v4.3) hold,
# grow, or erode over a genuinely long run -- not just a shallow proof of
# concept.
#
# One arm, so no GPU round-robin/max-parallel decision to make the way the
# 3-arm shallow sweep needed one.
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# run_sweep_h200.sh's list; nothing new is needed for this script:
#   transformer_neurIPS/train_production_transformer_deep_dive.py
#   transformer_neurIPS/model_variants.py
#   transformer_neurIPS/sweep_deep_dive.py
#   transformer_neurIPS/data/train_80.h5   (or a make_sweep_sample_data.py
#   transformer_neurIPS/data/val_80.h5      sample renamed to these exact
#                                            names -- see OVERVIEW.md 20.6.
#                                            For THIS run, prefer the FULL
#                                            files if bandwidth allows --
#                                            this is the production-scale
#                                            check, not a throwaway signal
#                                            test, so subset_ratio=1.0 on
#                                            the real data is what actually
#                                            answers the open question.)
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_h200_scale_e3.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                          (default: python3)
#   MAX_STEPS      optimizer steps                              (default: 12000)
#   VAL_EVERY      eval cadence in steps                        (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval             (default: 64)
#   SUBSET_RATIO   fraction of train data loaded                 (default: 1.0)
#   NO_WANDB       set to 1 to skip wandb tracking                (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(s6_e3_scaled)
MAX_STEPS="${MAX_STEPS:-12000}"
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
echo " H200/CUDA production-scale follow-up -- arm: ${ARMS[*]}"
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
  echo "    to regenerate on THIS box, or copy the FULL .h5 files over. Prefer"
  echo "    the full files for this particular run -- see the header comment"
  echo "    above on why subset_ratio=1.0 on real data is the point here."
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
  echo "  See transformer_neurIPS/requirements_sweep.txt for the exact"
  echo "  install command (torch nightly builds need a non-default index-url)."
  exit 1
fi
echo "$VERSION_CHECK" | sed 's/^/  [OK]     /'
echo ""

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -eq 0 ]]; then
  echo "WARNING: nvidia-smi found no GPUs. s6_e3_scaled uses AR_MODE='frame_ar',"
  echo "which resolve_train_regime() silently disables without CUDA -- running"
  echo "this script here would produce a misleading result, not an error."
  echo "This script is meant for a CUDA box (e.g. H200). Aborting."
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
echo "wandb       : $WANDB_STATUS"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- cold start, consistent with"
echo "              the shallow sweep this follows up on."
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
  --val-every "$VAL_EVERY" \
  --rollout-seqs "$ROLLOUT_SEQS" \
  --subset-ratio "$SUBSET_RATIO" \
  --fresh \
  --no-warm-start \
  "${EXTRA_FLAGS[@]}"
