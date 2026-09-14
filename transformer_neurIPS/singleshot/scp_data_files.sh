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
# Default (TRANSFER_METHOD=scp): two CONCURRENT scp processes rather
# than one after another -- if the bottleneck is per-connection
# overhead/latency rather than raw link bandwidth, this gets both done
# in roughly the time of the larger file instead of the sum of both.
#
# TRANSFER_METHOD=croc: CONFIRMED faster -- 2.4x the public relay at
# real 1GB-file scale (singleshot/CROC_JOURNEY.md) -- but sequential
# per file, not concurrent, since both would otherwise share one
# relay/tunnel and starve each other (the journey doc's bug #6).
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
#   TRANSFER_METHOD  "scp" (default) or "croc". croc's DEFAULT public
#                 relay is NOT what "croc" here means -- see below --
#                 it self-hosts a relay ON THE POD, reached through an
#                 SSH tunnel (never a direct connection to the pod's
#                 public IP -- see lib_croc.sh). CONFIRMED at 1GB real-
#                 file scale: 2.4x the public relay (93.1 vs 39.4
#                 MB/s), using croc's own default settings -- see
#                 singleshot/CROC_JOURNEY.md for the full story,
#                 including why NOT to override --transfers (going
#                 from croc's default of 4 real connections to its
#                 hard-capped max of 8 measured SLOWER, not faster).
#                 DELIBERATELY NO SCP FALLBACK: if TRANSFER_METHOD=croc
#                 is requested and any step of it fails, this script
#                 exits 1 with a diagnosis instead of silently sending
#                 the same bytes over scp anyway -- a silent fallback
#                 would hide exactly the signal needed to tell whether
#                 the self-hosted-relay path is actually working.
#                 The two files send SEQUENTIALLY under this method
#                 (unlike the scp path below, which sends them
#                 concurrently) -- they'd share the one relay/tunnel,
#                 and CROC_JOURNEY.md's bug #6 confirmed concurrent
#                 croc transfers sharing one tunnel starve each other
#                 badly, worse than just doing them one at a time.
#   CROC_RELAY_PORT  base port for the self-hosted relay (default 9019
#                 -- deliberately not croc's own default 9009, to avoid
#                 colliding with anything already using that).
#   RELAY_TRANSFERS  how many extra ports the relay opens beyond the
#                 base port (default 8, i.e. 9 ports total -- matches
#                 croc's own confirmed hard cap of 8 real parallel
#                 connections per transfer, see lib_croc.sh; opening
#                 more would be pure waste, per CROC_JOURNEY.md).
#                 Tunneled entirely through SSH -- no pod port
#                 exposure needed beyond SSH itself.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"

# lib_croc.sh's functions print progress via these -- this script's own
# convention is plain, uncolored echo (unlike bench_croc_variants.sh),
# so match that instead of introducing color codes here.
ok()   { echo "  [OK] $1"; }
warn() { echo "  [WARN] $1"; }
err()  { echo "  [FAIL] $1" >&2; }
info() { echo "  [..] $1"; }

# shellcheck source=./lib_croc.sh
source "$HERE/lib_croc.sh"

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
TRANSFER_METHOD="${TRANSFER_METHOD:-scp}"
CROC_RELAY_PORT="${CROC_RELAY_PORT:-9019}"
RELAY_TRANSFERS="${RELAY_TRANSFERS:-8}"

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

# Safety net for an interrupted run (Ctrl-C mid-transfer, etc.) -- both
# functions are no-ops if TRANSFER_METHOD=scp never set RELAY_PID/
# TUNNEL_PID in the first place.
trap 'croc_tunnel_stop; croc_relay_stop' EXIT

croc_fail() {
  echo "" >&2
  echo "==================================================================" >&2
  echo " TRANSFER_METHOD=croc FAILED -- $1" >&2
  echo "==================================================================" >&2
  echo "$2" >&2
  echo "" >&2
  echo "NOT falling back to scp (by design -- see this script's header" >&2
  echo "comment). Re-run with TRANSFER_METHOD=scp if you want the data" >&2
  echo "moved right now regardless; otherwise fix the issue above first." >&2
  croc_tunnel_stop
  croc_relay_stop
  exit 1
}

if [[ "$TRANSFER_METHOD" == "croc" ]]; then
  echo "TRANSFER_METHOD=croc -- self-hosted relay via SSH tunnel, sequentially"
  echo "per file (no scp fallback). See CROC_JOURNEY.md for why: confirmed"
  echo "2.4x the public relay at 1GB scale (93.1 vs 39.4 MB/s)."
  echo ""
  croc_require_local_binary
  croc_ensure_installed_remote
  croc_ensure_classic_mode_remote
  croc_relay_start "$CROC_RELAY_PORT" "$RELAY_TRANSFERS"
  croc_tunnel_start "$CROC_RELAY_PORT" "$RELAY_TRANSFERS"

  for f in "${FILES[@]}"; do
    src_path="$REPO_ROOT/transformer_neurIPS/data/$f"
    expected_bytes="$(stat -f%z "$src_path" 2>/dev/null || stat -c%s "$src_path")"
    echo "  [croc] sending $f ($expected_bytes bytes)..."
    slog="$(mktemp)"
    # No --transfers override on either side -- croc's own default (4
    # real connections) is the CONFIRMED-fastest setting at real file
    # size; going to its hard-capped max of 8 measured SLOWER, not
    # faster (CROC_JOURNEY.md's "second wall" section).
    croc_send "--relay localhost:$CROC_RELAY_PORT" "" "--no-local" "$src_path" "$slog"
    send_pid=$!
    code="$(croc_parse_code_from_log "$slog" 30)"
    if [[ -z "$code" ]]; then
      send_output="$(cat "$slog")"
      kill "$send_pid" 2>/dev/null || true
      rm -f "$slog"
      croc_fail "$f: croc send never printed a code within 30s" "$send_output"
    fi
    rlog="$(mktemp)"
    croc_receive 300 "$REMOTE_DATA_DIR" "$f" "--relay localhost:$CROC_RELAY_PORT" "" \
      "$code" "$rlog" "$expected_bytes"
    if grep -q "RC=0 " "$rlog" && grep -q "SIZE_OK" "$rlog"; then
      elapsed="$(grep -oE 'ELAPSED=[0-9]+' "$rlog" | head -1 | cut -d= -f2)"
      [[ -z "$elapsed" || "$elapsed" == "0" ]] && elapsed=1
      mb_per_s=$(awk -v b="$expected_bytes" -v s="$elapsed" 'BEGIN{printf "%.1f", (b/1048576)/s}')
      echo "  [croc] $f: OK -- ${elapsed}s, ${mb_per_s} MB/s, size verified"
    else
      recv_output="$(cat "$rlog")"
      kill "$send_pid" 2>/dev/null || true
      rm -f "$slog" "$rlog"
      croc_fail "$f: receive failed, timed out, or size mismatch" "$recv_output"
    fi
    kill "$send_pid" 2>/dev/null || true
    rm -f "$slog" "$rlog"
  done
  croc_tunnel_stop
  croc_relay_stop
  echo ""
else
  echo "Sending train_80.h5 and val_80.h5 via scp, concurrently..."
  echo ""

  pids=()
  for f in "${FILES[@]}"; do
    (
      scp "${SCP_OPTS[@]}" "$REPO_ROOT/transformer_neurIPS/data/$f" \
        "$POD_USER@$POD_HOST:$REMOTE_DATA_DIR/"
    ) &
    pids+=("$!")
  done

  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "" >&2
    echo "At least one transfer failed -- re-run this script (scp overwrites" >&2
    echo "partial files on retry, it does not resume them)." >&2
    exit 1
  fi
fi

echo ""
echo "Verifying sizes on both ends..."
for f in "${FILES[@]}"; do
  local_size="$(stat -f%z "$REPO_ROOT/transformer_neurIPS/data/$f" 2>/dev/null || stat -c%s "$REPO_ROOT/transformer_neurIPS/data/$f" 2>/dev/null)"
  remote_size="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
    "stat -c%s '$REMOTE_DATA_DIR/$f' 2>/dev/null || stat -f%z '$REMOTE_DATA_DIR/$f'")"
  if [[ "$local_size" != "$remote_size" ]]; then
    echo "  SIZE MISMATCH on $f: local=$local_size remote=$remote_size" >&2
    exit 1
  fi
  echo "  OK -- $f ($remote_size bytes)"
done

echo ""
echo "Done. Data files are at ${REMOTE_DATA_DIR}/."
echo "If bootstrap_remote.sh (phase 1's follow-up) has already finished in"
echo "your other session, you're ready to smoke-test / launch training."
