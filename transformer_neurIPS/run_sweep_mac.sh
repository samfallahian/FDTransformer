#!/usr/bin/env bash
# Shallow Round-2 sweep for the arms that can meaningfully run on Apple
# Silicon (MPS) / CPU -- see OVERVIEW.md v3.8 (Documentation section
# "v3.8 -- rollout-divergence diagnosis and Round-2 sweep infrastructure").
#
# WHY ONLY THESE TWO ARMS
# ========================
# ROUND2_ARMS["A"] and ["E"] in train_production_transformer_deep_dive.py
# define several "stabilise the rollout" arms, but most use
# AR_MODE='frame_ar', which resolve_train_regime() forces to 'none' on
# MPS/CPU (the sequential AR loop retains AR_FRAMES*NUM_X activation graphs,
# which blows past the MPS memory ceiling regardless of AR_SEQS). Running
# those arms here would silently degrade to "AR loss off" and hand back a
# misleading number.
#
#   e6_sched_noise   AR_MODE='sched' -- verified empirically (not assumed)
#                     to cost about the same as one normal training step on
#                     this hardware (~34 MB vs ~32 MB measured), because it's
#                     exactly two forwards with no retained sequential chain.
#                     This is the ONLY AR-family arm that exercises its real
#                     mechanism on MPS/CPU today.
#   a5b_wd_heavy      No AR component at all (weight_decay=0.1, dropout=0.1)
#                     -- tests a different hypothesis: that the model's own
#                     weights amplify error under repeated self-application,
#                     independent of exposure/horizon.
#
# The CUDA-only counterparts (a4b_ar_very_long, e3_ar_long,
# a6b_ar_feedback_noise) live in run_sweep_h200.sh. Run both -- they test
# non-overlapping hypotheses, not the same one on different hardware.
#
# Stage 0 diagnostic verdict this sweep is following up on
# (diagnose_rollout_noise_sensitivity.py, run against
# saved_models/old/r1_a3b_delta_ar_latest.pt): the divergence is
# BIAS-dominated (|bias|/RMSE = 0.822, zero sensitivity to injected noise up
# to 100x). That makes a5b_wd_heavy's hypothesis the weaker of the two priors
# here -- e6_sched_noise is included because scheduled sampling exposes the
# model to (and lets gradient correct) its own one-step error during
# training, which is a plausible bias-correction mechanism even without the
# long AR horizon this hardware can't run.
#
# REQUIRED FILES (checked below before anything runs)
# =====================================================
#   transformer_neurIPS/train_production_transformer_deep_dive.py   the trainer
#   transformer_neurIPS/model_variants.py                           model definitions (imported by the trainer)
#   transformer_neurIPS/sweep_deep_dive.py                          this script's launcher/aggregator
#   transformer_neurIPS/data/train_80.h5                            training sequences (prepare_data.py output)
#   transformer_neurIPS/data/val_80.h5                              validation sequences (prepare_data.py output)
#   encoder/autoencoderGEN3/saved_models_production/
#     Model_GEN3_05_AttentionSE_absolute_best_scripted.pt           frozen decoder -- centroid_velocity_loss
#                                                                    decodes through this every training step
#
# USAGE
# =====
#   bash transformer_neurIPS/run_sweep_mac.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                    (default: python3)
#   MAX_STEPS      optimizer steps per arm                (default: 400)
#   VAL_EVERY      eval cadence in steps                  (default: 100)
#   ROLLOUT_SEQS   sequences scored per rollout eval       (default: 16)
#   SUBSET_RATIO   fraction of train data loaded           (default: 0.3)
#   ACCUM          gradient-accumulation steps             (default: 4)
#
# ACCUM is deliberately cut from this hardware's normal default (32, to
# reach an effective batch of 32 at micro_batch=1) down to 4 -- an
# 8x-fewer-micro-forwards speedup, trading gradient-estimate noise for
# turnaround time. This is a SHALLOW, throwaway signal check ("did this arm
# show ANY sign of promoting a real _rollout_best.pt"), not a final-quality
# run -- a promising arm here should be re-run at full settings before
# trusting the number, not just scaled up on the same hardware.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
# Resolution order: explicit PYTHON_BIN env override > this project's known
# sibling venv (has h5py/torch/wandb installed; a bare `python3` from PATH
# is very likely the system/homebrew interpreter, which does NOT) > python3
# from PATH as a last resort.
_KNOWN_VENV="$(dirname "$REPO_ROOT")/cgan_last_venv_ever/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # explicit override, use as-is
elif [[ -x "$_KNOWN_VENV" ]]; then
  PYTHON_BIN="$_KNOWN_VENV"
else
  PYTHON_BIN="python3"
fi

ARMS=(e6_sched_noise a5b_wd_heavy)
MAX_STEPS="${MAX_STEPS:-400}"
VAL_EVERY="${VAL_EVERY:-100}"
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
echo " Mac/MPS shallow sweep -- arms: ${ARMS[*]}"
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
  echo "  - the scripted decoder: see OVERVIEW.md section 2 for provenance"
  exit 1
fi
echo ""
echo "Checking interpreter has the required packages (torch, h5py, numpy)..."
if ! "$PYTHON_BIN" -c "import torch, h5py, numpy" >/dev/null 2>&1; then
  echo "  [MISSING] $PYTHON_BIN cannot import torch/h5py/numpy."
  echo "  A bare 'python3' from PATH is very likely the system/homebrew"
  echo "  interpreter, not this project's venv. Set PYTHON_BIN explicitly, e.g.:"
  echo "    PYTHON_BIN=/path/to/venv/bin/python bash $0"
  exit 1
fi
echo "  [OK]"
echo ""
echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "Device      : MPS if available, else CPU (this box has no CUDA -- see"
echo "              pick_device() in train_production_transformer_deep_dive.py)"
echo "Budget      : max_steps=$MAX_STEPS val_every=$VAL_EVERY rollout_seqs=$ROLLOUT_SEQS"
echo "              subset_ratio=$SUBSET_RATIO accum=$ACCUM (cut from the hardware"
echo "              default of 32 for turnaround -- see header comment)"
echo "Concurrency : --max-parallel 1 (serialize on this Mac's single MPS device;"
echo "              sweep_deep_dive.py's CUDA_VISIBLE_DEVICES round-robin is a"
echo "              no-op without CUDA, so parallel launches would just contend"
echo "              for the same device)"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- every arm cold-starts, so"
echo "              this is a controlled comparison, not biased by whichever"
echo "              external checkpoint happens to be on disk."
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
