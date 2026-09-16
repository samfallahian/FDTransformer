#!/usr/bin/env bash
# Phase 2 of 2: copies the large train/val data files -- the slow part.
# Run this any time after (or, in a separate terminal, AT THE SAME TIME
# as) scp_env_files.sh -- it's independent of the venv/bootstrap step.
#
# train_80.h5 (~9.9G) and val_80.h5 (~4.2G) -- stored uncompressed as of
# decompress_h5.py (see OVERVIEW.md: gzip only shrank this data ~13%
# while still costing full single-threaded zlib decompression on every
# load, a bad trade).
#
# Each file is split into LANES pieces (default 10) sent via LANES
# CONCURRENT scp processes, reassembled and verified (size AND sha256)
# on the far end -- see lib_scp.sh and bench_scp_variants.sh for the
# full story. CONFIRMED in this session: 10 independent concurrent
# scp/ssh connections hit 268.3 MB/s raw (iperf3) against this exact
# pod infrastructure, close to the ~312 MB/s link ceiling -- far above
# the flat ~85-93 MB/s a croc self-hosted relay topped out at. croc has
# been dropped from this pipeline entirely: it pays for a PAKE
# handshake and its own encryption layer ON TOP of the already-
# encrypted SSH connection for no security benefit here, and its own
# split-lanes benchmark measured far below plain scp at the same lane
# count (see bench_croc_variants.sh/bench_scp_variants.sh headers for
# the comparison, and singleshot/CROC_JOURNEY.md for the original
# croc investigation this superseded).
#
# The two files' SENDS are still SEQUENTIAL (train_80.h5 fully split-
# sent, THEN val_80.h5 starts sending), each getting the full LANES
# upload-bandwidth budget to itself -- NOT two files x LANES lanes each
# concurrently, which would just reintroduce the same connection-
# contention problem this design exists to avoid.
#
# Reassembly/verification is DIFFERENT: it's remote CPU/disk work (a
# `cat` + `stat` + `shasum` over ssh) that uses none of the local
# upload bandwidth the NEXT file's send needs, so there's no reason to
# block on it. Each file's reassembly/verify runs in the BACKGROUND as
# soon as its own send finishes, overlapping with the next file's send
# -- e.g. train_80.h5's ~9.9GB reassembly+sha256 (mostly remote CPU)
# overlaps with val_80.h5's send (mostly local upload bandwidth)
# instead of strictly happening before it. Both are awaited (and their
# pass/fail checked) before this script reports done.
#
# USAGE
# =====
#   POD_HOST=1.2.3.4 POD_PORT=22 bash singleshot/scp_data_files.sh
#
# Overridable via environment variables:
#   POD_HOST      required -- the pod's SSH host/IP
#   POD_PORT      default: 22
#   POD_USER      default: root
#   SSH_KEY       default: ~/.ssh/id_ed25519
#   REMOTE_ROOT   default: /workspace  -- files land at
#                 $REMOTE_ROOT/cgan/transformer_neurIPS/data/
#   LANES         concurrent scp streams PER FILE (default 10 --
#                 confirmed via iperf3 to approach this pod
#                 infrastructure's real link ceiling; see lib_scp.sh).
#                 Files still send one after another, each getting this
#                 many lanes to itself -- see above for why.
#   SCP_LANE_TIMEOUT  per-lane hard cutoff in seconds (default 600 --
#                 generous margin: even train_80.h5's ~9.9GB split 10
#                 ways is ~1GB/lane, which clears this at any
#                 throughput above ~1.7 MB/s per lane)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"

# lib_scp.sh's functions print progress via these -- this script's own
# convention is plain, uncolored echo (unlike bench_scp_variants.sh),
# so match that instead of introducing color codes here.
ok()   { echo "  [OK] $1"; }
warn() { echo "  [WARN] $1"; }
err()  { echo "  [FAIL] $1" >&2; }
info() { echo "  [..] $1"; }

# shellcheck source=./lib_common.sh
source "$HERE/lib_common.sh"
# shellcheck source=./lib_scp.sh
source "$HERE/lib_scp.sh"

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
LANES="${LANES:-10}"
SCP_LANE_TIMEOUT="${SCP_LANE_TIMEOUT:-600}"

if [[ -z "$POD_HOST" ]]; then
  echo "POD_HOST is required, e.g.: POD_HOST=1.2.3.4 POD_PORT=22222 bash $0" >&2
  exit 1
fi

FILES=("train_80.h5" "val_80.h5")

echo "=================================================================="
echo " Phase 2/2: data files -> ${POD_USER}@${POD_HOST}:${POD_PORT}${REMOTE_ROOT}/cgan"
echo "=================================================================="
echo ""
missing=0
for f in "${FILES[@]}"; do
  path="$REPO_ROOT/transformer_neurIPS/data/$f"
  if [[ -f "$path" ]]; then
    printf "  [OK]      %-20s (%s)\n" "$f" "$(du -h "$path" | cut -f1)"
  else
    printf "  [MISSING] %s\n" "$path"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "" >&2
  echo "One or more data files are missing locally -- see OVERVIEW.md for" >&2
  echo "how to regenerate them." >&2
  exit 1
fi
echo ""

SSH_OPTS=(-p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
SCP_OPTS=(-P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
REMOTE_DATA_DIR="$REMOTE_ROOT/cgan/transformer_neurIPS/data"
ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "mkdir -p '$REMOTE_DATA_DIR'"

echo "Sending train_80.h5, then val_80.h5, each via $LANES concurrent scp"
echo "streams (sends are sequential per file; each file's reassembly/"
echo "verify overlaps with the NEXT file's send instead of blocking it"
echo "-- see this script's header for why that's safe)..."
echo ""

# reassemble_verify_async REMOTE_FILENAME SCRATCH BYTES SHA PIECES_DIR OUTVAR_PID
# Backgrounds scp_reassemble_verify, tagging its output so it stays
# distinguishable from whatever the next file's send is printing at the
# same time. `set -o pipefail` INSIDE the subshell (not just this
# script's own top-level `set -euo pipefail`) makes the subshell's own
# exit status reflect scp_reassemble_verify's real result rather than
# `awk`'s (which basically never fails) -- same pattern, and the same
# reason, as provision_and_run.sh's launch_training.
reassemble_verify_async() {
  local remote_filename="$1" scratch="$2" bytes="$3" sha="$4" pieces_dir="$5" outvar="$6"
  local tag="[reassemble $remote_filename]"
  (
    set -o pipefail
    scp_reassemble_verify "$REMOTE_DATA_DIR" "$remote_filename" "$scratch" "$bytes" "$sha" "$pieces_dir" \
      2>&1 | awk -v tag="$tag" '{print tag, $0; fflush()}'
  ) &
  eval "$outvar=\$!"
}

TRAIN_SCRATCH=""; TRAIN_PIECES=""; TRAIN_BYTES=""; TRAIN_SHA=""; TRAIN_VERIFY_PID=""
VAL_SCRATCH=""; VAL_PIECES=""; VAL_BYTES=""; VAL_SHA=""; VAL_VERIFY_PID=""

echo "  [scp] sending train_80.h5 via $LANES concurrent streams..."
if ! scp_send_pieces "$REPO_ROOT/transformer_neurIPS/data/train_80.h5" "$REMOTE_DATA_DIR" \
    "$LANES" "$SCP_LANE_TIMEOUT" TRAIN_SCRATCH TRAIN_PIECES TRAIN_BYTES TRAIN_SHA; then
  echo "" >&2
  echo "train_80.h5 failed to send -- re-run this script (scp overwrites" >&2
  echo "partial files on retry, it does not resume them)." >&2
  exit 1
fi
reassemble_verify_async "train_80.h5" "$TRAIN_SCRATCH" "$TRAIN_BYTES" "$TRAIN_SHA" "$TRAIN_PIECES" TRAIN_VERIFY_PID

echo "  [scp] sending val_80.h5 via $LANES concurrent streams (train_80.h5's"
echo "        reassembly/verify above is running concurrently with this)..."
if ! scp_send_pieces "$REPO_ROOT/transformer_neurIPS/data/val_80.h5" "$REMOTE_DATA_DIR" \
    "$LANES" "$SCP_LANE_TIMEOUT" VAL_SCRATCH VAL_PIECES VAL_BYTES VAL_SHA; then
  echo "" >&2
  echo "val_80.h5 failed to send -- re-run this script (scp overwrites" >&2
  echo "partial files on retry, it does not resume them)." >&2
  wait "$TRAIN_VERIFY_PID" || true
  exit 1
fi
reassemble_verify_async "val_80.h5" "$VAL_SCRATCH" "$VAL_BYTES" "$VAL_SHA" "$VAL_PIECES" VAL_VERIFY_PID

echo ""
echo "Waiting for both files' reassembly/verification to finish..."
TRAIN_OK=1; wait "$TRAIN_VERIFY_PID" || TRAIN_OK=0
VAL_OK=1; wait "$VAL_VERIFY_PID" || VAL_OK=0

echo ""
if [[ "$TRAIN_OK" != "1" || "$VAL_OK" != "1" ]]; then
  echo "AT LEAST ONE FILE FAILED reassembly/verification -- see errors above." >&2
  echo "Re-run this script (scp overwrites partial files on retry, it does" >&2
  echo "not resume them)." >&2
  exit 1
fi
echo "Done. Data files are at ${REMOTE_DATA_DIR}/."
