#!/usr/bin/env bash
# Phase 2 of 2: copies the large train/val data files -- the slow part.
# Run this any time after (or, in a separate terminal, AT THE SAME TIME
# as) scp_env_files.sh -- it's independent of the venv/bootstrap step.
#
# train_80.h5 (~9.9G) and val_80.h5 (~4.2G) -- stored uncompressed as of
# decompress_h5.py (see OVERVIEW.md: gzip only shrank this data ~13%
# while still costing full single-threaded zlib decompression on every
# load, a bad trade) -- are sent as two CONCURRENT scp processes rather
# than one after another -- if the bottleneck is per-connection
# overhead/latency rather than raw link bandwidth, this gets both done
# in roughly the time of the larger file instead of the sum of both. If
# the link genuinely is bandwidth-saturated already, this costs nothing
# (both processes just share the same pipe). On a fast symmetric link,
# `croc` (singleshot/README.md's "Other transfer options") may beat
# this further since it parallelizes within each file too, not just
# across the two files.
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
#                 it self-hosts a relay ON THE POD instead, since the
#                 public relay measured ~18-21 MB/s in a live test
#                 (tests/test_croc_transfer_speed.py), the same ceiling
#                 seen in real uploads regardless of local link speed.
#                 DELIBERATELY NO SCP FALLBACK: if TRANSFER_METHOD=croc
#                 is requested and any step of it fails, this script
#                 exits 1 with a diagnosis instead of silently sending
#                 the same bytes over scp anyway -- a silent fallback
#                 would hide exactly the signal needed to tell whether
#                 the self-hosted-relay fix (still unverified end to
#                 end as of this writing -- see README's "Other
#                 transfer options") is actually working.
#   CROC_RELAY_PORT  base port for the self-hosted relay (default 9019
#                 -- deliberately not croc's own default 9009, to avoid
#                 colliding with anything already using that). croc's
#                 relay actually needs a RANGE of 5 ports starting here
#                 (base + 4 "transfers" ports) -- provision_and_run.sh's
#                 `pod create` call must expose all 5 as TCP ports for
#                 this to have any chance of working.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
TRANSFER_METHOD="${TRANSFER_METHOD:-scp}"
CROC_RELAY_PORT="${CROC_RELAY_PORT:-9019}"

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

RELAY_PID=""

croc_cleanup_relay() {
  # Kill by the EXACT pid captured at launch -- NOT `pkill -f "croc
  # relay"`. That was tried first in tests/test_croc_transfer_speed.py
  # and, when combined with any command after it in the same ssh call,
  # self-destructed the whole SSH session: `pkill -f <pattern>` scans
  # every process's FULL command line, which trivially includes the
  # invoking shell's own command line (since that literally contains
  # the pkill invocation's text) -- pkill's self-exclusion only covers
  # the pkill process itself, not its parent shell. A numeric `kill
  # <pid>` has no such hazard.
  if [[ -n "$RELAY_PID" ]]; then
    ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
      "kill $RELAY_PID 2>/dev/null; true" || true
    RELAY_PID=""
  fi
}

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
  croc_cleanup_relay
  exit 1
}

if [[ "$TRANSFER_METHOD" == "croc" ]]; then
  echo "TRANSFER_METHOD=croc -- self-hosted-relay transfer (no scp fallback)."
  echo ""
  if ! command -v croc >/dev/null 2>&1; then
    croc_fail "croc not installed locally" \
      "Install it (e.g. 'brew install croc' on macOS, or the official\ninstaller: curl https://getcroc.schollz.com | bash) and re-run."
  fi
  if ! ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "command -v croc" >/dev/null 2>&1; then
    croc_fail "croc not installed on the pod" \
      "bootstrap_remote.sh installs this -- confirm it actually ran and\nfinished on this pod (check its console output for the 'Installing\ncroc' step)."
  fi

  # classic mode: croc 11.x refuses a receive code as a bare CLI arg by
  # default. This is a PERSISTED toggle (~/.config/croc), flipped by
  # *running* `croc --classic` -- so check the flag file directly
  # rather than invoking that (which would just flip it again).
  flag="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
    "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
  if [[ "$flag" != "enabled" ]]; then
    ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "printf 'y\\n' | croc --classic" >/dev/null 2>&1 || true
    flag="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
      "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
    if [[ "$flag" != "enabled" ]]; then
      croc_fail "could not enable croc's classic mode on the pod" \
        "Needed so a receive code can be passed as a bare CLI argument\n(croc 11.x otherwise demands interactive stdin or \$CROC_SECRET).\nTry manually: ssh in and run 'printf \"y\\\\n\" | croc --classic'."
    fi
  fi

  RELAY_PID="$(ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
    "rm -f /tmp/croc_relay.log; bash -c 'setsid croc relay --port $CROC_RELAY_PORT </dev/null >/tmp/croc_relay.log 2>&1 & echo \$!'" \
    2>/dev/null || true)"
  sleep 2
  if [[ -z "$RELAY_PID" ]]; then
    croc_fail "could not start/confirm a relay process on the pod" \
      "The launch command itself returned no pid. Check the pod has\nenough resources free and croc's binary isn't broken (try running\n'croc relay --port $CROC_RELAY_PORT' manually over SSH to see its\nown error output directly)."
  fi
  echo "  [croc] relay running on pod, pid=$RELAY_PID, ports ${CROC_RELAY_PORT}-$((CROC_RELAY_PORT + 4))"

  # Fail fast and with a clear diagnosis on an unreachable port, rather
  # than discovering it 3 minutes into a hung transfer (what happened
  # the first time this was tried against a real pod, before this
  # check existed) -- /dev/tcp is a bash builtin, no nc/netcat needed.
  if ! timeout 5 bash -c "echo >/dev/tcp/$POD_HOST/$CROC_RELAY_PORT" 2>/dev/null; then
    croc_fail "cannot reach $POD_HOST:$CROC_RELAY_PORT from this machine" \
      "The relay is running on the pod but its port isn't reachable from\nhere. Most likely cause: the pod wasn't created with this port\nexposed -- provision_and_run.sh's 'pod create' call requests\nCROC_RELAY_PORT through CROC_RELAY_PORT+4 as TCP ports, but a pod\ncreated before that fix landed (or reused via POD_ID, or created by\nsome other means) won't have it. Recreate the pod, or add the port\nrange to it manually if your provider supports that after creation."
  fi
  echo "  [croc] port $CROC_RELAY_PORT reachable from this machine -- good sign."

  for f in "${FILES[@]}"; do
    echo "  [croc] sending $f ..."
    SEND_LOG="$(mktemp)"
    croc --yes --relay "$POD_HOST:$CROC_RELAY_PORT" send \
      "$REPO_ROOT/transformer_neurIPS/data/$f" >"$SEND_LOG" 2>&1 &
    SEND_PID=$!
    CODE=""
    for _ in $(seq 1 30); do
      if grep -q "run:" "$SEND_LOG" 2>/dev/null; then
        CODE="$(grep -A1 "run:" "$SEND_LOG" | tail -1 | sed -E 's/^\s*croc\s+//; s/\s*\(.*$//' | awk '{print $NF}')"
        [[ -n "$CODE" ]] && break
      fi
      sleep 1
    done
    if [[ -z "$CODE" ]]; then
      SEND_OUTPUT="$(cat "$SEND_LOG")"
      kill "$SEND_PID" 2>/dev/null || true
      rm -f "$SEND_LOG"
      croc_fail "$f: croc send never printed a code within 30s" "$SEND_OUTPUT"
    fi
    t0=$(date +%s)
    if timeout 300 ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" \
      "cd '$REMOTE_DATA_DIR' && rm -f '$f' && croc --yes --relay localhost:$CROC_RELAY_PORT '$CODE'"; then
      elapsed=$(( $(date +%s) - t0 ))
      size_bytes="$(stat -f%z "$REPO_ROOT/transformer_neurIPS/data/$f" 2>/dev/null || stat -c%s "$REPO_ROOT/transformer_neurIPS/data/$f")"
      mb_per_s=$(awk -v b="$size_bytes" -v s="$elapsed" 'BEGIN{ if (s<1) s=1; printf "%.1f", (b/1048576)/s }')
      echo "  [croc] $f: OK -- ${elapsed}s, ${mb_per_s} MB/s"
    else
      kill "$SEND_PID" 2>/dev/null || true
      rm -f "$SEND_LOG"
      croc_fail "$f: receive side failed or timed out after 300s" \
        "See ssh output above. If it hung on 'waiting for sender...' the\nsender (this machine) likely couldn't complete the handshake even\nthough the initial port probe above succeeded -- could be a\nmid-transfer firewall/NAT issue rather than a fully-closed port."
    fi
    kill "$SEND_PID" 2>/dev/null || true
    rm -f "$SEND_LOG"
  done
  croc_cleanup_relay
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
