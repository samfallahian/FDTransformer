#!/usr/bin/env bash
# Wide, parallel search around the ONE mechanism that has ever produced a
# real rollout win in this investigation -- see OVERVIEW.md v4.4 and
# ROUND2_ARMS["H"] in train_production_transformer_deep_dive.py.
#
# WHY THIS EXISTS
# =================
# The just-completed Branch-R screen (5 arms: frame tokenization, tiny
# model, high LR, MSE+nonorm) reconfirmed -- again -- that NO amount of
# architecture/objective tweaking fixes rollout divergence without AR-mode
# training: 4/5 arms beat the training-objective anchor yet catastrophically
# diverge on the real 68-frame rollout (one, r2_frame_delta_mse, literally
# exploded to 284 MSE against a 0.0028 persistence floor). The only arm
# across this entire investigation to ever post a positive rollout
# improvement is `e3_ar_long` (AR_MODE='frame_ar', AR_FRAMES=8, AR_SEQS=2,
# AR_EVERY_N_STEPS=8) -- and OVERVIEW.md 23.2 showed it beat the
# longer-horizon a4b_ar_very_long arm PURELY because it got AR-loss applied
# twice as often at the shallow 2000-step budget, not because 8 frames is
# fundamentally better than 14.
#
# Given rented fast hardware and limited time, the highest-value use of it
# is NOT another single serial arm -- it's a wide, one-shot, parallel search
# around e3's config: frequency, horizon, batch, regularisation, and LR, all
# rooted in the mechanism that's actually worked, instead of another
# architecture guess that's already been ruled out 9 times over
# (a1/a2/r1-r5/a3b/etc.).
#
#   h1_ar_freq2         e3's horizon at 4x its AR-loss frequency (every 2 steps)
#   h2_ar_freq4         e3's horizon at 2x its AR-loss frequency (every 4 steps)
#   h3_ar_short_freq4   shorter horizon (4 frames), high frequency -- brackets e3 from below
#   h4_ar_long_freq4    a4b's 14-frame horizon at e3's proven-better frequency band --
#                       tests whether a4b only lost because of under-application
#   h5_ar_moreseqs      e3's config, AR_SEQS 2->8 (less gradient noise per AR step)
#   h6_ar_fbnoise       e3's config + noise on the fed-back prediction during the AR loop
#   h7_ar_wd            e3's config + heavier weight decay/dropout
#   h8_ar_lrlow         e3's config at half the peak LR (Branch R showed non-AR
#                       arms are LR-sensitive to the point of catastrophic
#                       divergence; checks whether AR training is too)
#
# All eight use AR_MODE='frame_ar', so this launcher is CUDA-only, same as
# run_sweep_h200.sh -- run_sweep_mac_branch_r.sh remains the free/non-CUDA
# counterpart for architecture-only questions.
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# run_sweep_h200.sh's list:
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
#   bash transformer_neurIPS/run_sweep_h300_branch_h.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                          (default: python3)
#   MAX_STEPS      optimizer steps per arm                      (default: 2000, shallow-screen budget matching e3_ar_long's own discovery run)
#   VAL_EVERY      eval cadence in steps                        (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval             (default: 64)
#   SUBSET_RATIO   fraction of train data loaded                 (default: 1.0)
#   MAX_PARALLEL   concurrent arms (round-robins across GPUs)    (default: min(GPU count, 8) -- all 8 arms at once if the box has 8+ GPUs)
#   NO_WANDB       set to 1 to skip wandb tracking                (default: 0, i.e. tracked)
#
# This is a SHALLOW screen (2000 steps), same convention as the original
# e3_ar_long discovery run -- whichever arm(s) win here should be re-run at
# production scale (12000+ steps) the same way s6_e3_scaled followed
# e3_ar_long, not trusted at this budget alone.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(h1_ar_freq2 h2_ar_freq4 h3_ar_short_freq4 h4_ar_long_freq4 h5_ar_moreseqs h6_ar_fbnoise h7_ar_wd h8_ar_lrlow)
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-200}"
ROLLOUT_SEQS="${ROLLOUT_SEQS:-64}"
SUBSET_RATIO="${SUBSET_RATIO:-1.0}"
NO_WANDB="${NO_WANDB:-0}"
SKIP_DIAGNOSTICS="${SKIP_DIAGNOSTICS:-0}"

REQUIRED_FILES=(
  "$HERE/train_production_transformer_deep_dive.py"
  "$HERE/model_variants.py"
  "$HERE/sweep_deep_dive.py"
  "$HERE/data/train_80.h5"
  "$HERE/data/val_80.h5"
  "$REPO_ROOT/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
)

echo "=================================================================="
echo " H300/CUDA wide AR-mode search -- arms: ${ARMS[*]}"
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
  echo "    to regenerate on THIS box, or copy the FULL .h5 files over. On a"
  echo "    throttled link, generate smaller samples locally instead with"
  echo "    'python transformer_neurIPS/make_sweep_sample_data.py' and upload"
  echo "    those, renamed to exactly train_80.h5 / val_80.h5 (this script"
  echo "    and the trainer are filename-based, not size-based -- see"
  echo "    OVERVIEW.md section 20.6)."
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
  echo "WARNING: nvidia-smi found no GPUs. All 8 arms use AR_MODE='frame_ar',"
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
echo "Concurrency : --max-parallel $MAX_PARALLEL (round-robins arms across"
echo "              CUDA_VISIBLE_DEVICES; if GPU_COUNT < 8, some GPUs run"
echo "              more than one arm sequentially -- if this box is a"
echo "              single GPU, all 8 run one after another unattended,"
echo "              which is still the point: queue everything, walk away)"
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
echo "Warm-start  : disabled (--no-warm-start) -- every arm cold-starts, so"
echo "              this is a controlled comparison, not biased by whichever"
echo "              external checkpoint happens to be on disk."
echo ""

EXTRA_FLAGS=()
if [[ "$NO_WANDB" == "1" ]]; then
  EXTRA_FLAGS+=(--no-wandb)
fi
if [[ "$SKIP_DIAGNOSTICS" == "1" ]]; then
  EXTRA_FLAGS+=(--skip-diagnostics)
  echo "Diagnostics : SKIPPED (SKIP_DIAGNOSTICS=1) -- reusing a prior run's linear-baseline"
  echo "              number instead of recomputing it. It does not depend on which arms"
  echo "              run or their step budget, only on the fixed train/val data, so this"
  echo "              is safe as long as the data hasn't changed since it was last computed."
  echo ""
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
