#!/usr/bin/env bash
# Phase 1 of 2: copies everything needed to build the venv and start
# training EXCEPT the large train/val data files -- code, requirements,
# the bootstrap script, the recovered checkpoint, the ridge map, the AE
# decoder. All small; this finishes in seconds, not minutes.
#
# WHY SPLIT THIS OUT
# ===================
# The old approach built one big tar.gz (code + ~11.3GB of data) and
# scp'd it as a single blob -- correct, but means NOTHING can start on
# the pod until the entire 11.3GB has landed, even though bootstrapping
# the venv (creating it, installing numpy/h5py/wandb/torch) needs none
# of the data at all. Splitting the transfer means:
#   - This phase (small files) finishes almost immediately.
#   - You can open a SECOND ssh session right after this phase and run
#     bootstrap_remote.sh (venv + deps) while phase 2
#     (scp_data_files.sh, the big transfer) is still running in your
#     first session/terminal -- overlapping two things that don't
#     depend on each other instead of doing them strictly in sequence.
#
# USAGE
# =====
#   POD_HOST=1.2.3.4 POD_PORT=22 bash singleshot/scp_env_files.sh
#
# Overridable via environment variables:
#   POD_HOST      required -- the pod's SSH host/IP
#   POD_PORT      default: 22
#   POD_USER      default: root
#   SSH_KEY       default: ~/.ssh/id_ed25519
#   REMOTE_ROOT   default: /workspace  -- files land at
#                 $REMOTE_ROOT/cgan/...

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"

if [[ -z "$POD_HOST" ]]; then
  echo "POD_HOST is required, e.g.: POD_HOST=1.2.3.4 POD_PORT=22222 bash $0" >&2
  exit 1
fi

# (local relative path, remote directory under $REMOTE_ROOT/cgan/) pairs.
FILES=(
  "transformer_neurIPS/train_production_transformer_deep_dive.py:transformer_neurIPS"
  "transformer_neurIPS/model_variants.py:transformer_neurIPS"
  "transformer_neurIPS/sweep_deep_dive.py:transformer_neurIPS"
  "transformer_neurIPS/run_sweep_h300_production_8h.sh:transformer_neurIPS"
  "transformer_neurIPS/saved_models/r2_s7_h9_scaled_rollout_best.pt:transformer_neurIPS/saved_models"
  "transformer_neurIPS/saved_models/r2_s7_h9_scaled_rollout_best_scripted.pt:transformer_neurIPS/saved_models"
  # MPS baseline for the two-experiment h11_ridge_distill comparison
  # (OVERVIEW.md v6.3 follow-up): Experiment A resumes from this file
  # (--round 2, matching its run_name exactly) instead of cold-starting.
  # Only the plain state-dict is needed -- load_resume_checkpoint() never
  # reads the _scripted twin, so it's deliberately not listed here.
  "transformer_neurIPS/saved_models/r2_h11_ridge_distill_latest.pt:transformer_neurIPS/saved_models"
  # s8_h9_moreseqs_scaled, the standing production candidate (OVERVIEW.md
  # v3.x/v6.7 -- v3_h9_moreseqs's exact config at production scale): this
  # is its last known checkpoint, step 4000/45000 (+22-27%, noisy, never
  # completed). Resuming with --round 2 --max-steps 45000 continues from
  # here rather than restarting from scratch. Plain state-dict only, same
  # reason as h11 above.
  "transformer_neurIPS/saved_models/r2_s8_h9_moreseqs_scaled_latest.pt:transformer_neurIPS/saved_models"
  # s7_h9_scaled, round 5: h9_ar_freq1's exact config, a clean/unconfounded
  # cold-start production-scale run (started this session -- the earlier
  # round-2 confounded number came from an accidental cold-restart glued
  # onto a resumed run). Currently the strongest live lead (+40.2% at just
  # step 2425/20000). Resume target for continuing it further.
  "transformer_neurIPS/saved_models/r5_s7_h9_scaled_latest.pt:transformer_neurIPS/saved_models"
  # ROUND 6 for both s8 and h11 -- NOT their `_latest.pt` progress, their
  # `_best.pt` (the checkpoint actually keyed to peak rollout performance,
  # never re-derived once training moved past that point). Both r2 runs
  # showed the SAME pattern: train_loss kept improving while the actual
  # rollout-vs-persistence metric got steadily WORSE the longer they ran
  # (s8: 28.5% best -> 16.4% latest at step 6500; h11: 37.3% best,
  # unchanged since step 2500 -> 23.3% latest at step 5000) -- textbook
  # overfitting/rollout-drift, and every resume so far has been
  # continuing from the increasingly-degraded `_latest.pt`, not the true
  # peak. Round 6 resumes from each arm's actual best instead, to see
  # whether continuing from the peak (rather than past it) can push
  # further without the same drift.
  "transformer_neurIPS/saved_models/r6_s8_h9_moreseqs_scaled_latest.pt:transformer_neurIPS/saved_models"
  "transformer_neurIPS/saved_models/r6_h11_ridge_distill_latest.pt:transformer_neurIPS/saved_models"
  # Rounds 7/8/9: s7_h9_scaled x3, FRESH cold starts, identical config
  # except WD/dropout 0 / 0.05 / 0.10 -- a controlled A/B/C testing
  # whether regularization prevents the overfitting/rollout-drift
  # pattern common to every arm tried so far (OVERVIEW.md v6.9). Only
  # reached step ~900-1000 of 6000 in the first 0.5h window (3-way
  # same-arm GPU contention costs ~2.4x the throughput of 3 different
  # arms, v6.9 section 39.1) -- these resume targets continue that same
  # comparison, NOT restart it; --max-steps must stay 6000 on every
  # future resume too (v6.7's LR-schedule-on-resume bug).
  "transformer_neurIPS/saved_models/r7_s7_h9_scaled_latest.pt:transformer_neurIPS/saved_models"
  "transformer_neurIPS/saved_models/r8_s7_h9_scaled_latest.pt:transformer_neurIPS/saved_models"
  "transformer_neurIPS/saved_models/r9_s7_h9_scaled_latest.pt:transformer_neurIPS/saved_models"
  "transformer_neurIPS/saved_models/ridge_frame_map.pt:transformer_neurIPS/saved_models"
  "encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt:encoder/autoencoderGEN3/saved_models_production"
  "transformer_neurIPS/singleshot/requirements.txt:transformer_neurIPS/singleshot"
  "transformer_neurIPS/singleshot/bootstrap_remote.sh:transformer_neurIPS/singleshot"
)

echo "=================================================================="
echo " Phase 1/2: env files -> ${POD_USER}@${POD_HOST}:${POD_PORT}${REMOTE_ROOT}/cgan"
echo "=================================================================="
echo ""
echo "Checking required local files..."
missing=0
for pair in "${FILES[@]}"; do
  rel="${pair%%:*}"
  f="$REPO_ROOT/$rel"
  if [[ -f "$f" ]]; then
    size="$(du -h "$f" 2>/dev/null | cut -f1)"
    printf "  [OK]      %-70s (%s)\n" "$rel" "$size"
  else
    printf "  [MISSING] %s\n" "$rel"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo ""
  echo "One or more required files are missing -- fix the path above before" >&2
  echo "sending anything." >&2
  exit 1
fi
echo ""

SSH_OPTS=(-p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)

# Pre-create every remote directory once, up front -- avoids N separate
# "mkdir -p" round-trips interleaved with N scp calls.
REMOTE_DIRS=$(printf '%s\n' "${FILES[@]}" | cut -d: -f2 | sort -u)
MKDIR_CMD="mkdir -p"
while IFS= read -r d; do
  MKDIR_CMD+=" '$REMOTE_ROOT/cgan/$d'"
done <<< "$REMOTE_DIRS"
ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "$MKDIR_CMD"

echo "Sending files..."
for pair in "${FILES[@]}"; do
  rel="${pair%%:*}"
  remote_dir="${pair##*:}"
  local_size="$(stat -f%z "$REPO_ROOT/$rel" 2>/dev/null || stat -c%s "$REPO_ROOT/$rel" 2>/dev/null)"
  scp "${SCP_OPTS[@]}" "$REPO_ROOT/$rel" \
    "$POD_USER@$POD_HOST:$REMOTE_ROOT/cgan/$remote_dir/"
  remote_size="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
    "stat -c%s '$REMOTE_ROOT/cgan/$remote_dir/$(basename "$rel")' 2>/dev/null || stat -f%z '$REMOTE_ROOT/cgan/$remote_dir/$(basename "$rel")'")"
  if [[ "$local_size" != "$remote_size" ]]; then
    echo "  SIZE MISMATCH on $rel: local=$local_size remote=$remote_size" >&2
    exit 1
  fi
done

ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
  "chmod +x '$REMOTE_ROOT/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh' '$REMOTE_ROOT/cgan/transformer_neurIPS/run_sweep_h300_production_8h.sh'"

echo ""
echo "Done -- env files landed at ${REMOTE_ROOT}/cgan/. (data files NOT sent"
echo "yet -- that's scp_data_files.sh, phase 2)."
echo ""
echo "You do NOT need a second ssh session for this anymore -- SSH in ONCE"
echo "and run bootstrap in a detached 'screen' session, which keeps going"
echo "even if this connection drops. See singleshot/README.md's screen-"
echo "session table for the exact commands (bootstrap/exp-a/exp-b):"
echo ""
echo "  ssh -p $POD_PORT -i $SSH_KEY $POD_USER@$POD_HOST"
echo "  screen -dmS bootstrap bash -c '"
echo "    bash ${REMOTE_ROOT}/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh"
echo "    tail -F ${REMOTE_ROOT}/cgan/transformer_neurIPS/exp_a.log'"
echo ""
echo "Then, whenever you're ready for the data (can run before, during, or"
echo "after the above -- independent of it):"
echo "  POD_HOST=$POD_HOST POD_PORT=$POD_PORT bash $HERE/scp_data_files.sh"
