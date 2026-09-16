#!/usr/bin/env bash
# lib_scp.sh -- shared parallel-scp split/send/reassemble/verify logic,
# used by both bench_scp_variants.sh (local smoke test / real-pod
# benchmark) and scp_data_files.sh (the actual production transfer of
# train_80.h5/val_80.h5). Extracted here for the exact reason
# lib_croc.sh already documents: duplicating this ad hoc in every
# script that needs it is how bugs get introduced twice.
#
# WHY PLAIN SCP, NOT CROC, FOR THIS -- see bench_scp_variants.sh's
# header for the full story: croc's own split-lanes phase (real file,
# 10 independent relay+tunnel pairs) measured far below raw iperf3's
# number at the same lane count, because croc pays for a PAKE
# handshake, its own encryption layer ON TOP of the already-encrypted
# SSH tunnel, and optional compression per lane -- none of which iperf3
# exercises. Plain scp already gets one independent ssh connection per
# invocation with a single encryption layer and no relay, which is
# CONFIRMED (bench_scp_variants.sh, this session) to reach 268.3 MB/s
# at 10 concurrent lanes against this same infrastructure.
#
# WHY SEND AND REASSEMBLE ARE SEPARATE FUNCTIONS
# ================================================
# Reassembly/verification (remote `cat` + `stat` + `shasum`) is remote
# CPU/disk work -- it uses none of the LOCAL machine's upload bandwidth.
# The next file's send is upload-bandwidth-bound and uses essentially no
# remote CPU. These don't compete for the same resource, so there's no
# reason to block the next file's send on the previous file's
# reassembly finishing -- scp_data_files.sh runs them concurrently:
# send file N (blocking) -> launch file N's reassembly/verify in the
# BACKGROUND -> send file N+1 (blocking, overlapping with N's
# reassembly) -> ... -> wait for every backgrounded reassembly at the
# end. scp_send_split below is a synchronous convenience wrapper around
# the same two primitives, for callers (bench_scp_variants.sh) that
# don't need the overlap.
#
# REQUIRED GLOBALS (set these before calling anything below):
#   SSH_OPTS   bash array, e.g. (-p "$POD_PORT" -i "$SSH_KEY"
#              -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
#   SCP_OPTS   same idea, scp's flag spelling (-P not -p)
#   POD_HOST   the pod's SSH host/IP
#   POD_USER   the pod's SSH login user
#
# Also requires run_with_timeout (lib_common.sh) and ok/warn/err/info
# already defined by the caller (same convention lib_croc.sh set).

# ---------------------------------------------------------------- #
# scp_send_pieces LOCAL_FILE REMOTE_DIR LANES TIMEOUT_PER_LANE \
#                 OUTVAR_SCRATCH OUTVAR_PIECES_DIR OUTVAR_BYTES OUTVAR_SHA
#
# Splits LOCAL_FILE into LANES pieces and sends them via LANES
# concurrent `scp` processes (each already its own independent ssh
# connection -- no manual tunnel/port-forward setup needed, unlike
# croc's self-hosted relay) into a scratch directory under REMOTE_DIR
# UNIQUE to this file (keyed by this process's pid AND the file's own
# basename -- two calls from the SAME script for two DIFFERENT files
# share one $$, so the basename is required to keep their scratch dirs
# from colliding once callers run them concurrently).
#
# On success (0), writes the scratch dir, local pieces dir, expected
# byte count, and expected sha256 into the four OUTVARs -- the caller
# needs all four later to call scp_reassemble_verify, possibly from a
# BACKGROUNDED context, which is why they're returned explicitly rather
# than reassembled here. Does NOT reassemble or clean up on success --
# that's scp_reassemble_verify's job. On failure (1), cleans up its own
# local pieces dir and remote scratch dir itself (nothing to reassemble
# anyway) and does not touch the OUTVARs.
# ---------------------------------------------------------------- #

# scp_lane_with_retry TIMEOUT_PER_ATTEMPT SRC DEST [MAX_ATTEMPTS]
#
# Retries one lane's scp up to MAX_ATTEMPTS (default 3) with a short
# backoff -- confirmed necessary the hard way: `kex_exchange_
# identification: read: Connection reset by peer` on one lane out of
# 10 while another ssh connection (phase 1's env-files transfer) was
# ALSO opening at the same moment. That's sshd's own `MaxStartups`
# anti-flood throttle randomly dropping a NEW, not-yet-authenticated
# connection once too many arrive at once -- a real, expected
# consequence of running several concurrent SSH-based transfers against
# one pod, not a sign the pod or the network is actually broken.
# Without this retry, ONE such transient drop failed the entire lane,
# which failed the entire file (scp_send_pieces below has no partial-
# success path), which aborted the whole pipeline -- for a failure mode
# that a simple retry a few seconds later almost always clears.
scp_lane_with_retry() {
  local timeout_secs="$1" src="$2" dest="$3" max_attempts="${4:-3}"
  local attempt=1
  while (( attempt <= max_attempts )); do
    if run_with_timeout "$timeout_secs" scp "${SCP_OPTS[@]}" "$src" "$dest"; then
      return 0
    fi
    if (( attempt < max_attempts )); then
      sleep $((attempt * 3))
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

# ssh_cmd_with_retry REMOTE_CMD [MAX_ATTEMPTS]
# Same MaxStartups-flood reasoning as scp_lane_with_retry, for a single
# plain ssh command (e.g. the initial `mkdir -p` below, which can also
# get hit by the same connection-burst throttle before any lane even
# starts sending).
ssh_cmd_with_retry() {
  local remote_cmd="$1" max_attempts="${2:-3}"
  local attempt=1
  while (( attempt <= max_attempts )); do
    if ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "$remote_cmd"; then
      return 0
    fi
    if (( attempt < max_attempts )); then
      sleep $((attempt * 3))
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

scp_send_pieces() {
  local local_file="$1" remote_dir="$2" lanes="$3" timeout_secs="$4"
  local outvar_scratch="$5" outvar_pieces_dir="$6" outvar_bytes="$7" outvar_sha="$8"
  local expected_bytes expected_sha piece_bytes pieces_dir remote_scratch
  local -a send_pids=() piece_names=()
  local i piece_name fail=0 t0 elapsed mbps fname

  fname="$(basename "$local_file")"
  expected_bytes="$(stat -f%z "$local_file" 2>/dev/null || stat -c%s "$local_file")"
  expected_sha="$(shasum -a 256 "$local_file" | awk '{print $1}')"
  pieces_dir="$(mktemp -d)"
  remote_scratch="${remote_dir}/.scp_split_${$}_${fname}"

  piece_bytes=$(( (expected_bytes + lanes - 1) / lanes ))
  info "  splitting $fname into $lanes pieces (~$((piece_bytes / 1048576)) MB each)..."
  split -b "$piece_bytes" -d -a 2 "$local_file" "$pieces_dir/piece_"

  if ! ssh_cmd_with_retry "mkdir -p '$remote_scratch'"; then
    err "  could not create remote scratch dir for $fname after retries"
    rm -rf "$pieces_dir"
    return 1
  fi

  info "  sending $fname via $lanes concurrent scp processes..."
  t0=$(date +%s)
  for i in $(seq 0 $((lanes - 1))); do
    piece_name="piece_$(printf '%02d' "$i")"
    piece_names[$i]="$piece_name"
    (
      scp_lane_with_retry "$timeout_secs" "$pieces_dir/$piece_name" \
        "$POD_USER@$POD_HOST:$remote_scratch/$piece_name"
    ) &
    send_pids[$i]=$!
  done
  for i in $(seq 0 $((lanes - 1))); do
    if ! wait "${send_pids[$i]}"; then
      err "  lane $i (${piece_names[$i]}) failed or timed out sending $fname"
      fail=1
    fi
  done
  if [[ "$fail" != "0" ]]; then
    err "  one or more lanes failed for $fname -- cleaning up, not attempting reassembly."
    rm -rf "$pieces_dir"
    ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "rm -rf '$remote_scratch'" 2>/dev/null || true
    return 1
  fi
  elapsed=$(( $(date +%s) - t0 ))
  [[ "$elapsed" -lt 1 ]] && elapsed=1
  mbps=$(awk -v b="$expected_bytes" -v s="$elapsed" 'BEGIN{printf "%.1f", (b/1048576)/s}')
  ok "  all $lanes lanes finished sending $fname in ${elapsed}s (${mbps} MB/s aggregate)"

  eval "$outvar_scratch=\"\$remote_scratch\""
  eval "$outvar_pieces_dir=\"\$pieces_dir\""
  eval "$outvar_bytes=\"\$expected_bytes\""
  eval "$outvar_sha=\"\$expected_sha\""
  return 0
}

# ---------------------------------------------------------------- #
# scp_reassemble_verify REMOTE_DIR REMOTE_FILENAME REMOTE_SCRATCH \
#                        EXPECTED_BYTES EXPECTED_SHA PIECES_DIR
#
# Reassembles the pieces scp_send_pieces already sent to REMOTE_SCRATCH
# into REMOTE_DIR/REMOTE_FILENAME, then verifies BOTH size AND sha256
# against EXPECTED_BYTES/EXPECTED_SHA (stronger than the croc
# benchmark's size-only check -- a passing byte count alone doesn't
# prove the bytes are actually right, and this is cheap enough to just
# always do). Always cleans up PIECES_DIR (local) and REMOTE_SCRATCH
# (remote), on both success and failure.
#
# Safe to run in a BACKGROUNDED subshell (see this file's header) --
# touches no shared state beyond the paths passed in as arguments.
# Returns 0 on verified success, 1 otherwise.
# ---------------------------------------------------------------- #
scp_reassemble_verify() {
  local remote_dir="$1" remote_filename="$2" remote_scratch="$3"
  local expected_bytes="$4" expected_sha="$5" pieces_dir="$6"
  local remote_bytes remote_sha rc=0

  info "  reassembling and verifying $remote_filename on $POD_HOST..."
  ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
    "mkdir -p '$remote_dir' && cd '$remote_scratch' && cat \$(ls piece_* | sort) > '$remote_dir/$remote_filename' && rm -f piece_*"
  remote_bytes="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "stat -c%s '$remote_dir/$remote_filename' 2>/dev/null || stat -f%z '$remote_dir/$remote_filename'")"
  remote_sha="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "shasum -a 256 '$remote_dir/$remote_filename' 2>/dev/null | awk '{print \$1}'")"

  if [[ "$remote_bytes" == "$expected_bytes" && "$remote_sha" == "$expected_sha" ]]; then
    ok "  size AND sha256 verified for $remote_filename"
  else
    err "  VERIFICATION FAILED for $remote_filename"
    err "    size:   got=${remote_bytes:-<none>} want=$expected_bytes"
    err "    sha256: got=${remote_sha:-<none>} want=$expected_sha"
    rc=1
  fi
  rm -rf "$pieces_dir"
  ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "rm -rf '$remote_scratch'" 2>/dev/null || true
  return "$rc"
}

# ---------------------------------------------------------------- #
# scp_send_split LOCAL_FILE REMOTE_DIR REMOTE_FILENAME LANES TIMEOUT_PER_LANE
#
# Synchronous convenience wrapper: scp_send_pieces followed immediately
# by scp_reassemble_verify, for callers (bench_scp_variants.sh) that
# only ever send one file and have nothing to overlap reassembly with.
# scp_data_files.sh calls the two primitives directly instead, so file
# N+1's send can overlap file N's reassembly -- see this file's header.
#
# Returns 0 on full success (every lane transferred + reassembly +
# verification all passed), 1 otherwise.
# ---------------------------------------------------------------- #
scp_send_split() {
  local local_file="$1" remote_dir="$2" remote_filename="$3" lanes="$4" timeout_secs="$5"
  local scratch pieces_dir bytes sha
  scp_send_pieces "$local_file" "$remote_dir" "$lanes" "$timeout_secs" \
    scratch pieces_dir bytes sha || return 1
  scp_reassemble_verify "$remote_dir" "$remote_filename" "$scratch" "$bytes" "$sha" "$pieces_dir"
}
