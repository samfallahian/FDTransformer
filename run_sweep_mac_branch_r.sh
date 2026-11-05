#!/usr/bin/env bash
# Shallow screen of Branch R ("a linear map beats the transformer -- the
# model or the framing is broken") on the Mac -- see OVERVIEW.md v4.4.
#
# WHY THIS RUNS ON THE MAC, NOT H200
# ====================================
# The production-scale s6_e3_scaled run (OVERVIEW.md v4.3/v4.4) confirmed
# the AR-horizon win holds at 12000 steps (+27.41% vs persistence) -- but
# also triggered sweep_deep_dive.py's Branch-R verdict: a closed-form
# ridge regression baseline beats it by more than 2x (+60.64%). Unlike
# every AR arm this investigation has run, NONE of the five Branch-R arms
# use AR_MODE='frame_ar' -- they test tokenization/architecture/objective
# variations in isolation:
#
#   r1_frame           frame tokenisation (matches the linear baseline's
#                       own per-frame factorisation)
#   r2_frame_delta_mse  frame + delta + MSE -- as close to the linear
#                       baseline as a net gets
#   r3_tiny             deliberately tiny (E128/L2) -- if small beats
#                       large, this is an optimisation failure, not a
#                       capacity one
#   r4_lr_sweep          peak LR 3e-3, longer warmup
#   r5_mse_nonorm        MSE objective, NORMALIZE_FEATURES off -- isolates
#                       the normalisation change
#
# None of them need CUDA at all -- this is a genuinely free screen,
# mirroring the SAME shallow-screen-then-scale-the-winner workflow that
# found e3_ar_long here on the Mac before ever spending H200 time on it.
# An H200 "scale the Branch-R winner" script (analogous to
# run_sweep_h200_scale_e3.sh) is deliberately NOT created yet -- that
# comes after this screen identifies which arm (if any) is worth scaling.
#
# REQUIRED FILES (checked below before anything runs)
# =====================================================
#   transformer_neurIPS/train_production_transformer_deep_dive.py   the trainer
#   transformer_neurIPS/model_variants.py                           model definitions (imported by the trainer)
#   transformer_neurIPS/sweep_deep_dive.py                          this script's launcher/aggregator
#   transformer_neurIPS/data/train_80.h5                            training sequences (or a make_sweep_sample_data.py sample -- see OVERVIEW.md 20.6)
#   transformer_neurIPS/data/val_80.h5                              validation sequences (same caveat)
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt           frozen decoder -- centroid_velocity_loss
#                                                                    decodes through this every training step
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_mac_branch_r.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                    (default: this project's known venv, else python3)
#   MAX_STEPS      optimizer steps per arm                (default: 2000, matching run_sweep_mac.sh's shallow-screen convention)
#   VAL_EVERY      eval cadence in steps                  (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval       (default: 16)
#   SUBSET_RATIO   fraction of train data loaded           (default: 0.3)
#   ACCUM          gradient-accumulation steps             (default: 4, cut from this hardware's normal 32 -- see run_sweep_mac.sh's header for the tradeoff)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
_KNOWN_VENV="$(dirname "$REPO_ROOT")/cgan_last_venv_ever/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # explicit override, use as-is
elif [[ -x "$_KNOWN_VENV" ]]; then
  PYTHON_BIN="$_KNOWN_VENV"
else
  PYTHON_BIN="python3"
fi

ARMS=(r1_frame r2_frame_delta_mse r3_tiny r4_lr_sweep r5_mse_nonorm)
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-200}"
ROLLOUT_SEQS="${ROLLOUT_SEQS:-16}"
SUBSET_RATIO="${SUBSET_RATIO:-0.3}"
ACCUM="${ACCUM:-4}"

REQUIRED_FILES=(
  "$HERE/train_production_transformer_deep_dive.py"
  "$HERE/model_variants.py"
  "$HERE/sweep_deep_dive.py"
  "$HERE/data/train_80.h5"
  "$HERE/data/val_80.h5"
  "$REPO_ROOT/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
)

echo "=================================================================="
echo " Mac/MPS Branch-R screen -- arms: ${ARMS[*]}"
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
  echo "for how to produce them (prepare_data.py, or make_sweep_sample_data.py"
  echo "for a smaller sample)."
  exit 1
fi
echo ""
echo "Checking interpreter has the required packages (torch, h5py, numpy)..."
if ! "$PYTHON_BIN" -c "import torch, h5py, numpy" >/dev/null 2>&1; then
  echo "  [MISSING] $PYTHON_BIN cannot import torch/h5py/numpy."
  echo "  Set PYTHON_BIN explicitly, e.g.:"
  echo "    PYTHON_BIN=/path/to/venv/bin/python bash $0"
  exit 1
fi
echo "  [OK]"
echo ""
echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "Device      : MPS if available, else CPU -- none of these 5 arms use"
echo "              AR_MODE='frame_ar', so unlike the earlier AR arms, this"
echo "              is a genuinely full (not degraded) test on this hardware."
echo "Budget      : max_steps=$MAX_STEPS val_every=$VAL_EVERY rollout_seqs=$ROLLOUT_SEQS"
echo "              subset_ratio=$SUBSET_RATIO accum=$ACCUM"
echo "Concurrency : --max-parallel 1 (serialize on this Mac's single MPS device)"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- every arm cold-starts."
echo ""

cd "$HERE"
exec "$PYTHON_BIN" sweep_deep_dive.py \
  --arms "${ARMS[@]}" \
  --round 2 \
  --max-parallel 1 \
  --max-steps "$MAX_STEPS" \
  --val-every "$VAL_EVERY" \
  --rollout-seqs "$ROLLOUT_SEQS" \
  --subset-ratio "$SUBSET_RATIO" \
  --accum "$ACCUM" \
  --no-wandb \
  --fresh \
  --no-warm-start
