#!/usr/bin/env bash
# Branch-R screen (r1-r5) on CUDA -- see OVERVIEW.md v4.4 sections 24.3/24.4.
#
# WHY THIS EXISTS, SEPARATE FROM run_sweep_mac_branch_r.sh
# ==========================================================
# s6_e3_scaled's production-scale confirmation (OVERVIEW.md 24.2, +27.41%
# vs persistence) also surfaced a bigger finding: a closed-form ridge
# regression frame-to-frame map beats it by more than 2x (+60.64%,
# section 24.3) -- a real "Branch R" verdict, trustworthy because
# s6_e3_scaled had actually converged when it fired (unlike the false
# positive at section 20.2's shallow, unconverged Mac run).
#
# The five ROUND2_ARMS["R"] arms below don't use AR_MODE='frame_ar' at
# all, so run_sweep_mac_branch_r.sh was built to screen them for free on
# the Mac -- but with CUDA hardware sitting idle and 10x+ faster per
# section 24's Mac ETA estimate (~20-30h serialized on the Mac vs.
# minutes-to-an-hour here), there's no reason to wait on the Mac run.
# This script runs the exact same 5 arms on CUDA instead, at full
# fidelity (no ACCUM cut, no SUBSET_RATIO cut) since the hardware can
# afford it directly -- mirroring run_sweep_h200.sh's "leave ACCUM at each
# arm's own default, H200 throughput makes the full-fidelity settings
# cheap" rationale.
#
#   r1_frame             Frame tokenisation -- matches the linear
#                         baseline's own per-frame factorisation.
#   r2_frame_delta_mse   Frame + delta + MSE -- as close to the linear
#                         baseline as a net gets.
#   r3_tiny              Deliberately tiny (E128/L2) -- if small beats
#                         large, this is an optimisation failure, not a
#                         capacity one.
#   r4_lr_sweep          Peak LR 3e-3, longer warmup.
#   r5_mse_nonorm        MSE objective, NORMALIZE_FEATURES off -- isolates
#                         the normalisation change.
#
# None of these five use AR_MODE='frame_ar' either, unlike run_sweep_h200.sh's
# three arms -- so unlike that script, running here is a speed choice, not
# a correctness requirement (run_sweep_mac_branch_r.sh would give the same
# answer, just much slower). Because none retain a sequential AR graph,
# they're also cheap enough to round-robin across multiple GPUs if this
# box has more than one, same as run_sweep_h200.sh does.
#
# REQUIRED FILES (checked below before anything runs) -- identical to
# run_sweep_h200.sh's list; nothing new is needed for this script:
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
#   bash transformer_neurIPS/run_sweep_h200_branch_r.sh
#
# Overridable via environment variables (all optional):
#   PYTHON_BIN     interpreter to use                          (default: python3)
#   MAX_STEPS      optimizer steps per arm                      (default: 2000, matching the shallow-screen budget these arms were defined at -- see LATEST_UPLOAD_ME.md's auto-suggested command; bump to 12000 here for a scaled follow-up the same way s6_e3_scaled followed e3_ar_long)
#   VAL_EVERY      eval cadence in steps                        (default: 200)
#   ROLLOUT_SEQS   sequences scored per rollout eval             (default: 64)
#   SUBSET_RATIO   fraction of train data loaded                 (default: 1.0)
#   MAX_PARALLEL   concurrent arms (round-robins across GPUs)    (default: min(GPU count, 5))
#   NO_WANDB       set to 1 to skip wandb tracking                (default: 0, i.e. tracked)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARMS=(r1_frame r2_frame_delta_mse r3_tiny r4_lr_sweep r5_mse_nonorm)
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
echo " H200/CUDA Branch-R screen -- arms: ${ARMS[*]}"
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
  echo "WARNING: nvidia-smi found no GPUs. None of these arms strictly"
  echo "require CUDA (they're all runnable on run_sweep_mac_branch_r.sh),"
  echo "so this isn't a correctness abort the way run_sweep_h200.sh's is --"
  echo "but this script is meant for a CUDA box for speed. Aborting;"
  echo "use run_sweep_mac_branch_r.sh on non-CUDA hardware instead."
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
