#!/usr/bin/env bash
# Shallow Round-2 sweep for the CUDA-only "stabilise the rollout" arms --
# see OVERVIEW.md v3.8 (Documentation section "v3.8 -- rollout-divergence
# diagnosis and Round-2 sweep infrastructure").
#
# WHY THESE ARMS NEED CUDA
# =========================
# All three arms below use AR_MODE='frame_ar': the sequential autoregressive
# loop that retains AR_FRAMES*NUM_X activation graphs for backward. On
# MPS/CPU that's forced to 'none' by resolve_train_regime() (retained-graph
# memory blowup); on CUDA it runs at the arm-specified AR_SEQS untouched.
# These are the ONLY arms in the current menu that can test the AR-horizon
# hypothesis at all -- run_sweep_mac.sh covers the (weaker, per the Stage 0
# diagnostic below) noise/regularization hypothesis on cheaper hardware.
#
#   a4b_ar_very_long        AR horizon 14 frames (140 tokens) -- ~21% of the
#                            68-frame eval rollout, the biggest single jump
#                            available in the current arm menu.
#   e3_ar_long               AR horizon 8 frames (80 tokens) -- cheaper
#                            middle ground between the current a3b (4
#                            frames) and a4b.
#   a6b_ar_feedback_noise    a4b's 14-frame horizon PLUS AR_FEEDBACK_NOISE_STD
#                            on the fed-back prediction -- tests whether the
#                            horizon extension and the feedback-noise
#                            mechanism combine better than either alone.
#
# Stage 0 diagnostic verdict these arms are following up on
# (diagnose_rollout_noise_sensitivity.py, run against
# saved_models/old/r1_a3b_delta_ar_latest.pt): the divergence is
# BIAS-dominated -- |bias|/RMSE = 0.822, and last-frame RMSE was IDENTICAL
# (16.76) whether injected noise was 0 or 100x larger. That is direct
# evidence AGAINST the noise/chaos hypothesis and FOR the horizon-extension
# hypothesis these three arms test: the model needs to see far enough into
# its own accumulated error during training to learn to correct it, not
# just be made more robust to small perturbations.
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
#   bash transformer_neurIPS/run_sweep_h200.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                          (default: python3)
#   MAX_STEPS      optimizer steps per arm                      (default: 2000)
#   VAL_EVERY      eval cadence in steps                        (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval             (default: 64)
#   SUBSET_RATIO   fraction of train data loaded                 (default: 1.0)
#   MAX_PARALLEL   concurrent arms (round-robins across GPUs)    (default: min(GPU count, 3))
#   NO_WANDB       set to 1 to skip wandb tracking                (default: 0, i.e. tracked)
#
# Unlike run_sweep_mac.sh, ACCUM is left at each arm's own default here --
# H200's throughput (bf16, torch.compile, micro_batch bumped well past 1)
# makes the full-fidelity settings cheap enough that there's no reason to
# trade gradient-estimate quality for turnaround the way the Mac script does.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(a4b_ar_very_long e3_ar_long a6b_ar_feedback_noise)
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
echo " H200/CUDA shallow sweep -- arms: ${ARMS[*]}"
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
  echo "    (regenerate on THIS box, or copy the .h5 files over -- they are large"
  echo "    and are not something this script will fetch for you)"
  echo "  - the scripted decoder: copy"
  echo "    encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
  echo "    from the box that trained it -- see OVERVIEW.md section 2 for provenance"
  exit 1
fi
echo ""
echo "Checking interpreter has the required packages (torch+cuda, h5py, numpy)..."
if ! "$PYTHON_BIN" -c "import torch, h5py, numpy; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "  [MISSING] $PYTHON_BIN cannot import torch/h5py/numpy, or torch.cuda.is_available()"
  echo "  is False. Set PYTHON_BIN to the CUDA-enabled venv on this box, e.g.:"
  echo "    PYTHON_BIN=/path/to/venv/bin/python bash $0"
  exit 1
fi
echo "  [OK]"
echo ""

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -eq 0 ]]; then
  echo "WARNING: nvidia-smi found no GPUs. These arms use AR_MODE='frame_ar',"
  echo "which resolve_train_regime() silently disables without CUDA -- running"
  echo "this script here would produce a misleading result, not an error."
  echo "This script is meant for a CUDA box (e.g. H200). Aborting."
  exit 1
fi
DEFAULT_MAX_PARALLEL=$(( GPU_COUNT < ${#ARMS[@]} ? GPU_COUNT : ${#ARMS[@]} ))
MAX_PARALLEL="${MAX_PARALLEL:-$DEFAULT_MAX_PARALLEL}"

echo "Interpreter : $("$PYTHON_BIN" --version 2>&1) at $(command -v "$PYTHON_BIN")"
echo "GPUs        : $GPU_COUNT detected  (nvidia-smi)"
echo "Budget      : max_steps=$MAX_STEPS val_every=$VAL_EVERY rollout_seqs=$ROLLOUT_SEQS"
echo "              subset_ratio=$SUBSET_RATIO"
echo "Concurrency : --max-parallel $MAX_PARALLEL (round-robins arms across"
echo "              CUDA_VISIBLE_DEVICES; if GPU_COUNT < #arms, some GPUs run"
echo "              more than one arm sequentially)"
echo "wandb       : $([[ "$NO_WANDB" == "1" ]] && echo "disabled (--no-wandb)" || echo "enabled -- project NI_Review")"
echo "Report      : transformer_neurIPS/sweep_logs/<run_id>/UPLOAD_ME.md"
echo "Warm-start  : disabled (--no-warm-start) -- every arm cold-starts, so"
echo "              this is a controlled comparison, not biased by whichever"
echo "              external checkpoint happens to be on disk."
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
