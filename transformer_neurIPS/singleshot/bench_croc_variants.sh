#!/usr/bin/env bash
# Spins up ONE cheap RunPod pod, tunnels a self-hosted croc relay's
# ports through the pod's ALREADY-working SSH connection (see
# "Establishing SSH tunnel" below for why -- 5 real pods in a row
# proved RunPod's Secure Cloud pods don't expose arbitrary custom TCP
# ports the way earlier versions of this script assumed), then
# benchmarks several croc transfer settings CONCURRENTLY against it --
# to answer, with a real number, whether a self-hosted relay actually
# fixes the ~18-21 MB/s ceiling measured against croc's DEFAULT public
# relay (tests/test_croc_transfer_speed.py), which is the exact
# bottleneck seen on real train_80.h5/val_80.h5 uploads.
#
# Deliberately does NOT need a GPU-class pod -- this tests the
# network/IO stack only, so it defaults to the cheapest reliably-
# stocked GPU tier available (still a "GPU pod" because that's the
# only pod type RunPod's API here creates, but the actual GPU is
# irrelevant to this test).
#
# WHY CONCURRENT, NOT SEQUENTIAL
# ===============================
# Every variant below shares the SAME self-hosted relay (croc's relay
# multiplexes independent transfers into separate "rooms" keyed by
# their one-time codephrase -- this is exactly what it's designed for,
# see `croc relay --help`'s --max-rooms-open) and runs at the same
# time, each logging to its own file. A background monitor loop prints
# a consolidated, live-refreshing status table (every ~5s) so real
# throughput is visible within the first 15-20 seconds, not just at
# the end. The public-relay baseline variant never touches the
# self-hosted relay at all, so it's safe to run alongside the others.
#
# USAGE
# =====
#   bash singleshot/bench_croc_variants.sh
#
# This script creates and DELETES a real billed pod. Read the printed
# cost-per-hour before confirming, and expect the whole run (pod boot +
# ~20-40s of transfers + teardown) to cost well under $1 at the
# defaults below.
#
# ENV VARS (all optional)
#   GPU_ID            exact `gpuId` field from `runpodctl gpu list -o
#                      json` -- NOT the `displayName` (learned the hard
#                      way: "A40" fails with a graphql "Unknown GPU
#                      type" error; the real id is "NVIDIA A40").
#                      Default "NVIDIA A40": cheap, $0.49/hr
#                      secure-cloud, "High" stock as of this writing.
#                      Override if unavailable -- check BOTH fields
#                      with:
#                        runpodctl gpu list -o json | jq -r '.[] | "\(.gpuId)\t\(.securePricePerHr)\t\(.stockStatus)"' | sort -n -k2
#   CONTAINER_DISK_GB  default 20 -- only a ~100MB test file per
#                      variant is ever created, no real training data
#                      touches this pod
#   WAIT_TIMEOUT       seconds to wait for SSH to come up before giving
#                      up on pod creation entirely (default 60 -- 20s,
#                      then 35s, both proved too tight: real pods that
#                      boot fine can still take a while just to get
#                      their SSH port allocated, seen up to ~33s on
#                      one CA-MTL-1 attempt and a full timeout at 35s
#                      on another)
#   TRANSFER_TIMEOUT   per-variant hard cutoff in seconds (default 30
#                      -- if a 100MB transfer isn't done in 30s at
#                      ANY of these settings, on a same-datacenter
#                      link, it isn't going to be)
#   FILE_SIZE_MB       size of the random test file (default 100)
#   CROC_RELAY_PORT    base port for the self-hosted relay, as seen by
#                      croc AND by this machine (default 9019). Only
#                      ever bound to localhost on both ends now -- see
#                      "Establishing SSH tunnel" below for why. 5 real
#                      pods across earlier attempts confirmed RunPod's
#                      Secure Cloud pods don't expose arbitrary custom
#                      TCP ports the way this script originally
#                      assumed (see docs.runpod.io/pods/configuration/
#                      expose-ports's "70000 offset" symmetric-mapping
#                      trick -- tried, confirmed via a 30s retry loop
#                      that the resulting env var never appeared), so
#                      this no longer touches the pod's exposed-ports
#                      list at all for anything beyond SSH.
#   RELAY_TRANSFERS    how many extra ports the self-hosted relay
#                      opens beyond the base port (default 16, so all
#                      concurrent variants below have separate ports
#                      to multiplex over -- see `croc relay --help`).
#                      Each one gets its own `-L` forward in the SSH
#                      tunnel below.
#   SSH_KEY            default: ~/.runpod/ssh/runpodctl-ssh-key
#   KEEP_POD           set to 1 to skip the final `pod delete`
#   POD_ID             reuse an already-running pod instead of
#                      creating one

set -uo pipefail  # NOT -e: this script's whole point is comparing
                   # variants where some are EXPECTED to fail/timeout;
                   # a failing variant must not abort the others or
                   # skip the summary table.

# ---------------------------------------------------------------- #
# Colors (respects NO_COLOR / non-tty, matching this repo's existing
# convention in train_production_transformer_deep_dive.py / PFD_NO_COLOR)
# ---------------------------------------------------------------- #
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ -z "${PFD_NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""
  C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
fi
hdr()  { printf '%s\n' "${C_BOLD}${C_CYAN}=================================================================="; printf '%s\n' " $1${C_RESET}"; printf '%s\n' "${C_BOLD}${C_CYAN}==================================================================${C_RESET}"; }
ok()   { printf '%s\n' "${C_GREEN}[OK]${C_RESET} $1"; }
warn() { printf '%s\n' "${C_YELLOW}[WARN]${C_RESET} $1"; }
err()  { printf '%s\n' "${C_RED}[FAIL]${C_RESET} $1" >&2; }
info() { printf '%s\n' "${C_BLUE}[..]${C_RESET} $1"; }

RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
if ! command -v "$RUNPODCTL" >/dev/null 2>&1; then
  err "runpodctl not found at $RUNPODCTL -- see singleshot/README.md's 'RunPod agent setup'."
  exit 1
fi
if ! command -v croc >/dev/null 2>&1; then
  err "croc not installed locally. brew install croc (macOS) or: curl https://getcroc.schollz.com | bash"
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  err "jq not installed locally (needed to parse runpodctl's JSON output)."
  exit 1
fi

GPU_ID="${GPU_ID:-NVIDIA A40}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
TRANSFER_TIMEOUT="${TRANSFER_TIMEOUT:-30}"
FILE_SIZE_MB="${FILE_SIZE_MB:-100}"
CROC_RELAY_PORT="${CROC_RELAY_PORT:-9019}"
RELAY_TRANSFERS="${RELAY_TRANSFERS:-16}"
SSH_KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
KEEP_POD="${KEEP_POD:-0}"

# ---------------------------------------------------------------- #
# Portable per-command timeout -- macOS has no `timeout(1)` by
# default. Background watchdog: sleep N, then kill the target pid if
# it's still alive. `wait` for the target's real exit code either way.
# ---------------------------------------------------------------- #
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local target=$!
  ( sleep "$secs"; kill -9 "$target" 2>/dev/null ) &
  local watchdog=$!
  local rc=0
  wait "$target" 2>/dev/null || rc=$?
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null
  return "$rc"
}

hdr "Preflight: RunPod credentials"
AUTH_CHECK="$("$RUNPODCTL" user -o json 2>&1 || true)"
if echo "$AUTH_CHECK" | jq -e '.error' >/dev/null 2>&1; then
  err "No RunPod credentials configured. Run 'runpodctl doctor' or export RUNPOD_API_KEY first."
  exit 1
fi
ok "credentials present"
echo ""

# ---------------------------------------------------------------- #
# Pod lifecycle
# ---------------------------------------------------------------- #
POD_ID="${POD_ID:-}"
CREATED_POD=0
cleanup() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [[ -n "${RELAY_PID:-}" ]]; then
    # Kill by exact pid, NEVER `pkill -f "croc relay"` -- see
    # scp_data_files.sh's croc_cleanup_relay() comment for why that
    # self-destructed an entire SSH session when tried.
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "kill $RELAY_PID 2>/dev/null; true" 2>/dev/null || true
  fi
  if [[ "$CREATED_POD" == "1" && -n "$POD_ID" ]]; then
    if [[ "$KEEP_POD" == "1" ]]; then
      warn "KEEP_POD=1 -- leaving pod $POD_ID running. Terminate yourself: runpodctl pod delete $POD_ID"
    else
      info "Terminating pod $POD_ID..."
      "$RUNPODCTL" pod delete "$POD_ID" || err "delete failed -- terminate manually: runpodctl pod delete $POD_ID"
    fi
  fi
}
trap cleanup EXIT

# Back to plain SSH-only exposure. 5 real pods in a row confirmed
# RunPod's Secure Cloud pods don't expose arbitrary custom TCP ports
# the way this script originally assumed -- neither literally (remapped
# to some undiscoverable random external port, like SSH's 22->22158)
# nor via the documented "add 70000" symmetric-mapping trick (the
# resulting $RUNPOD_TCP_PORT_* env var never appeared, confirmed with a
# 30s retry loop, so not just a propagation delay either). Rather than
# keep guessing at RunPod's port-exposure internals, this now tunnels
# every relay port through the ALREADY-proven-reliable SSH connection
# instead (see "Establishing SSH tunnel" below) -- no extra ports
# needed on the pod at all beyond the one that's always worked.
PORT_LIST="22/tcp"

if [[ -z "$POD_ID" ]]; then
  hdr "Preflight: validating GPU_ID against runpodctl gpu list"
  # Learned the hard way: RunPod's API wants the `gpuId` field (e.g.
  # "NVIDIA A40"), not the shorter `displayName` shown elsewhere (e.g.
  # "A40") -- the latter fails with a graphql "Unknown GPU type" error
  # AFTER already printing "Creating pod...". Catch it here instead,
  # before wasting any wait time on a request that can't succeed.
  GPU_LIST_JSON="$("$RUNPODCTL" gpu list -o json 2>&1)"
  if ! echo "$GPU_LIST_JSON" | jq -e --arg g "$GPU_ID" \
      '(. as $items | ($items[]? // $items.gpus[]?)) | select(.gpuId == $g)' \
      >/dev/null 2>&1; then
    err "GPU_ID=\"$GPU_ID\" does not match any 'gpuId' field from"
    err "'runpodctl gpu list'. Closest matches by displayName:"
    echo "$GPU_LIST_JSON" | jq -r --arg g "$GPU_ID" \
      '(. // .gpus) | .[]? | select(.displayName != null and (.displayName | ascii_downcase | contains($g | ascii_downcase)) or (.gpuId | ascii_downcase | contains($g | ascii_downcase))) | "  gpuId=\"\(.gpuId)\"  displayName=\"\(.displayName)\"  price=\(.securePricePerHr)  stock=\(.stockStatus)"' \
      2>/dev/null | head -5
    err "Full list: runpodctl gpu list -o json | jq -r '.[] | \"\\(.gpuId)\\t\\(.securePricePerHr)\\t\\(.stockStatus)\"'"
    exit 1
  fi
  ok "GPU_ID=\"$GPU_ID\" is valid"
  echo ""

  hdr "Creating pod (gpu=$GPU_ID, disk=${CONTAINER_DISK_GB}GB, ports=22/tcp only -- relay ports go through an SSH tunnel, not the pod's exposed-ports list)"
  info "Waiting up to ${WAIT_TIMEOUT}s for SSH -- if this times out, the pod"
  info "may still be booting; check 'runpodctl pod list' before retrying."
  POD_TEMPLATE_IMAGE="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
  POD_RAW="$("$RUNPODCTL" pod create \
    --gpu-id "$GPU_ID" \
    --template-id runpod-torch-v280 \
    --container-disk-in-gb "$CONTAINER_DISK_GB" \
    --ports "$PORT_LIST" \
    --public-ip \
    --wait --wait-timeout "${WAIT_TIMEOUT}s" \
    -o json 2>&1)"
  echo "$POD_RAW"
  # runpodctl interleaves human-readable progress ("note: ...", "waiting
  # for ssh...", "ssh on pod ... ready...") with the actual JSON on the
  # SAME stream (confirmed: both go to stderr even on success, so
  # separating stdout/stderr doesn't help) -- extract just the `{...}`
  # block by line-anchored range instead of trusting the whole capture
  # to be valid JSON on its own. Found the hard way: a pod was created
  # successfully and billing, but the raw combined text failed to
  # parse as JSON, so the id was never extracted and cleanup() never
  # ran -- the pod sat there running until deleted by hand.
  POD_JSON="$(echo "$POD_RAW" | awk '/^\{/,/^\}/')"
  POD_ID="$(echo "$POD_JSON" | jq -r '.id // .podId // empty' 2>/dev/null)"
  if [[ -z "$POD_ID" ]]; then
    err "Could not parse a pod id from the output above."
    warn "Safety net: looking for exactly one untracked pod matching this"
    warn "run's image ($POD_TEMPLATE_IMAGE) to avoid a stray billing pod..."
    CANDIDATE="$("$RUNPODCTL" pod list -o json 2>/dev/null | jq -r --arg img "$POD_TEMPLATE_IMAGE" \
      '[.[] | select(.imageName == $img)] | if length == 1 then .[0].id else empty end' 2>/dev/null)"
    if [[ -n "$CANDIDATE" ]]; then
      warn "Found exactly one match ($CANDIDATE) -- deleting it automatically."
      "$RUNPODCTL" pod delete "$CANDIDATE" || err "delete failed -- terminate manually: runpodctl pod delete $CANDIDATE"
    else
      err "Could not safely auto-identify a single candidate (zero or"
      err "multiple matches) -- check 'runpodctl pod list' and delete"
      err "manually: runpodctl pod delete <id>"
    fi
    exit 1
  fi
  CREATED_POD=1
  ok "pod id: $POD_ID"
  POD_HOST="$(echo "$POD_JSON" | jq -r '.ssh.ip // .ip // empty' 2>/dev/null)"
  POD_PORT="$(echo "$POD_JSON" | jq -r '.ssh.port // .port // empty' 2>/dev/null)"
  if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
    INFO_RAW="$("$RUNPODCTL" ssh info "$POD_ID" -o json 2>&1)"
    echo "$INFO_RAW"
    INFO_JSON="$(echo "$INFO_RAW" | awk '/^\{/,/^\}/')"
    POD_HOST="$(echo "$INFO_JSON" | jq -r '.host // .ip // .publicIp // empty' 2>/dev/null)"
    POD_PORT="$(echo "$INFO_JSON" | jq -r '.port // .sshPort // empty' 2>/dev/null)"
  fi
else
  info "Reusing existing pod $POD_ID (POD_ID was set)."
  INFO_RAW="$("$RUNPODCTL" ssh info "$POD_ID" -o json 2>&1)"
  echo "$INFO_RAW"
  INFO_JSON="$(echo "$INFO_RAW" | awk '/^\{/,/^\}/')"
  POD_HOST="$(echo "$INFO_JSON" | jq -r '.host // .ip // .publicIp // empty' 2>/dev/null)"
  POD_PORT="$(echo "$INFO_JSON" | jq -r '.port // .sshPort // empty' 2>/dev/null)"
fi

if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
  err "Could not determine SSH host/port for pod $POD_ID -- see raw JSON above."
  exit 1
fi
ok "SSH: root@$POD_HOST:$POD_PORT"
echo ""

SSH_OPTS=(-p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

hdr "Waiting for sshd to actually accept connections"
SSHD_READY=0
for i in $(seq 1 "$WAIT_TIMEOUT"); do
  if ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "echo ready" >/dev/null 2>&1; then
    SSHD_READY=1
    ok "sshd answered after ${i}s"
    break
  fi
  sleep 1
done
if [[ "$SSHD_READY" != "1" ]]; then
  err "sshd never answered within ${WAIT_TIMEOUT}s of pod creation reporting SSH ready."
  err "This is exactly the failure mode worth knowing about fast -- not a bug in"
  err "this script, a real signal that this pod/region is having a bad day."
  exit 1
fi
echo ""


hdr "Preparing the pod (croc present, classic mode, test file)"
# Bad assumption fixed: an earlier pod happened to have croc baked into
# its image, but that isn't guaranteed -- mirror bootstrap_remote.sh's
# already-proven install step (official installer, not apt) instead of
# just hoping it's there.
if ! ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "command -v croc" >/dev/null 2>&1; then
  info "croc not found on the pod -- installing via the official installer"
  info "(same command as singleshot/bootstrap_remote.sh)..."
  if ! ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "curl -s https://getcroc.schollz.com | bash" >/dev/null 2>&1; then
    err "croc install failed on the pod."
    exit 1
  fi
  if ! ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "command -v croc" >/dev/null 2>&1; then
    err "croc still not found on the pod after running the installer."
    exit 1
  fi
  ok "croc installed"
fi
ok "croc present on pod: $(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "croc --version" 2>/dev/null)"

FLAG="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
if [[ "$FLAG" != "enabled" ]]; then
  ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "printf 'y\\n' | croc --classic" >/dev/null 2>&1 || true
  FLAG="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
  if [[ "$FLAG" != "enabled" ]]; then
    err "could not enable croc's classic mode on the pod (needed for a scripted receive)."
    exit 1
  fi
fi
ok "croc classic mode enabled on pod"

REMOTE_DIR="/tmp/croc_bench"
ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "mkdir -p $REMOTE_DIR && rm -f $REMOTE_DIR/*"

LOCAL_DIR="$(mktemp -d)"
LOCAL_FILE="$LOCAL_DIR/bench_${FILE_SIZE_MB}mb.bin"
info "generating a random ${FILE_SIZE_MB}MB local test file (random, not zeros --"
info "avoids flattering any compression) ..."
dd if=/dev/urandom "of=$LOCAL_FILE" bs=1m count="$FILE_SIZE_MB" 2>/dev/null
ok "test file ready: $LOCAL_FILE"
echo ""

hdr "Starting the self-hosted relay on the pod"
RELAY_PID="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
  "rm -f /tmp/croc_relay.log; bash -c 'setsid croc relay --port $CROC_RELAY_PORT --transfers $RELAY_TRANSFERS </dev/null >/tmp/croc_relay.log 2>&1 & echo \$!'" \
  2>/dev/null || true)"
sleep 2
if [[ -z "$RELAY_PID" ]]; then
  err "relay did not start -- see /tmp/croc_relay.log on the pod."
  exit 1
fi
ok "relay running, pid=$RELAY_PID, ports $CROC_RELAY_PORT-$((CROC_RELAY_PORT + RELAY_TRANSFERS))"
echo ""

hdr "Establishing SSH tunnel for the relay's port range"
# Forwards EVERY relay port from this machine's localhost, through the
# pod's SSH connection, to localhost on the pod -- i.e. exactly where
# the relay above is listening. Needs NOTHING from RunPod beyond the
# SSH port that's worked all along; sidesteps the whole custom-TCP-
# port-exposure question entirely instead of relying on RunPod ever
# actually mapping $CROC_RELAY_PORT the way earlier attempts assumed.
TUNNEL_FLAGS=()
for i in $(seq 0 "$RELAY_TRANSFERS"); do
  p=$((CROC_RELAY_PORT + i))
  TUNNEL_FLAGS+=(-L "$p:localhost:$p")
done
ssh -N -o ExitOnForwardFailure=yes "${SSH_OPTS[@]}" "${TUNNEL_FLAGS[@]}" "root@$POD_HOST" &
TUNNEL_PID=$!
sleep 2
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
  err "SSH tunnel process exited immediately -- ExitOnForwardFailure means"
  err "at least one of the $((RELAY_TRANSFERS + 1)) local port forwards"
  err "couldn't bind (maybe already in use on THIS machine -- check for"
  err "another process on $CROC_RELAY_PORT-$((CROC_RELAY_PORT + RELAY_TRANSFERS))"
  err "locally, e.g. 'lsof -i :$CROC_RELAY_PORT')."
  exit 1
fi
ok "tunnel established, pid=$TUNNEL_PID, forwarding localhost:$CROC_RELAY_PORT-$((CROC_RELAY_PORT + RELAY_TRANSFERS)) -> pod"

if ! run_with_timeout 5 bash -c "echo >/dev/tcp/localhost/$CROC_RELAY_PORT" 2>/dev/null; then
  err "cannot reach localhost:$CROC_RELAY_PORT even through the SSH tunnel --"
  err "the tunnel process is alive but the actual forward isn't answering."
  err "Aborting here rather than spending more time/money on self-hosted"
  err "variants that are now confirmed doomed."
  exit 1
fi
ok "port $CROC_RELAY_PORT reachable via the tunnel from this machine"
echo ""

# ---------------------------------------------------------------- #
# Variants. `croc`'s usage is `[GLOBAL OPTIONS] [COMMAND] [COMMAND
# OPTIONS] [file]`. Verified directly against the installed binary's
# own --help (NOT assumed, after getting burned once already):
#   global (before "send", and the ONLY thing that applies to receive
#   too, since receive has no distinct subcommand keyword):
#     --relay, --yes, --no-compress
#   send-subcommand-only (must follow "send", never valid on receive
#   or as a bare global flag):
#     --no-local, --transfers
# Mixing these up is exactly what broke two variants the first time
# this ran: "flag provided but not defined: -transfers" (transfers
# passed as global), and this comment's OWN first draft made the
# identical mistake with --no-local before being checked against
# `croc --help` directly.
#
# --no-local on every SEND (not receive -- no such flag exists there),
# always: without it, `croc send` ALSO tries to open a local-network-
# discovery relay on ports 9009-9013 -- harmless with ONE send
# running, but with all 4 variants launching within the same second,
# they collide on those local binds ("address already in use",
# eventually a `panic:` in one log). Irrelevant here anyway since
# every variant already talks to a real relay (public or self-hosted),
# never local-network discovery.
#
# Columns: (name, uses-self-hosted-relay [0/1], --no-compress [0/1],
# send-only extra flags). The relay ADDRESS itself is deliberately NOT
# baked in here as one shared string -- the send side (this machine)
# must reach it via the pod's PUBLIC IP, but the receive side (running
# ON the pod, over ssh) must use "localhost", not its own public IP
# back at itself. Collapsing those into one shared flag was a real bug
# the first time this ran: the receive side tried to dial
# $POD_HOST:$CROC_RELAY_PORT (its own public address) and got
# "i/o timeout" on EVERY self-hosted variant -- exactly the
# public-vs-localhost distinction scp_data_files.sh/
# tests/test_croc_transfer_speed.py already got right. Resolved to the
# correct address per-side, per-variant, in the loop below instead.
# ---------------------------------------------------------------- #
declare -a V_NAME=() V_USE_RELAY=() V_NO_COMPRESS=() V_SEND_ONLY_FLAGS=()

V_NAME+=("public-relay-baseline");   V_USE_RELAY+=("0"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local")
V_NAME+=("self-hosted-default");     V_USE_RELAY+=("1"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local")
V_NAME+=("self-hosted-transfers16"); V_USE_RELAY+=("1"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local --transfers 16")
V_NAME+=("self-hosted-no-compress"); V_USE_RELAY+=("1"); V_NO_COMPRESS+=("1"); V_SEND_ONLY_FLAGS+=("--no-local")

N="${#V_NAME[@]}"
LOGDIR="$(mktemp -d)"
declare -a SEND_PID=() WATCHER_PID=() RESULT_MBPS=() RESULT_STATUS=() START_TS=()

hdr "Launching $N variants CONCURRENTLY against $LOGDIR"
for i in $(seq 0 $((N - 1))); do
  name="${V_NAME[$i]}"
  # Both sides now say "localhost" -- send connects to the tunnel's
  # local end on THIS machine, receive connects to the pod's own
  # localhost where the relay is bound. No longer a POD_HOST/localhost
  # asymmetry to get wrong (that mismatch was a real bug earlier).
  relay_flag=""
  [[ "${V_USE_RELAY[$i]}" == "1" ]] && relay_flag="--relay localhost:$CROC_RELAY_PORT"
  send_relay="$relay_flag"; recv_relay="$relay_flag"
  nocompress=""
  [[ "${V_NO_COMPRESS[$i]}" == "1" ]] && nocompress="--no-compress"
  sonly="${V_SEND_ONLY_FLAGS[$i]}"
  slog="$LOGDIR/$name.send.log"
  rlog="$LOGDIR/$name.recv.log"
  : >"$slog"; : >"$rlog"
  info "[$name] sending..."
  # shellcheck disable=SC2086
  croc --yes $send_relay $nocompress send $sonly "$LOCAL_FILE" >"$slog" 2>&1 &
  SEND_PID[$i]=$!
  START_TS[$i]=$(date +%s)
  WATCHER_PID[$i]=0
  RESULT_STATUS[$i]="running"
  RESULT_MBPS[$i]="-"
  ( # background: wait for the code, then run the timed receive
    code=""
    for _ in $(seq 1 15); do
      if grep -q "run:" "$slog" 2>/dev/null; then
        code="$(grep -A1 "run:" "$slog" | tail -1 | sed -E 's/^\s*croc\s+//; s/\s*\(.*$//' | awk '{print $NF}')"
        [[ -n "$code" ]] && break
      fi
      sleep 1
    done
    if [[ -z "$code" ]]; then
      echo "NEVER_GOT_CODE" >>"$rlog"
      exit 1
    fi
    t0=$(date +%s)
    # shellcheck disable=SC2086
    run_with_timeout "$TRANSFER_TIMEOUT" \
      ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
      "cd $REMOTE_DIR && rm -f $(basename "$LOCAL_FILE") && croc --yes $recv_relay $nocompress '$code'" \
      >>"$rlog" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))
    echo "RC=$rc ELAPSED=$elapsed" >>"$rlog"
  ) &
  WATCHER_PID[$i]=$!  # reuse WATCHER_PID[] to hold the background-wrapper's own pid
done
echo ""

hdr "Live status (refreshing every 5s, ${TRANSFER_TIMEOUT}s hard cutoff per variant)"
DEADLINE=$(( $(date +%s) + TRANSFER_TIMEOUT + 20 ))
while true; do
  all_done=1
  printf '%s\n' "${C_DIM}$(date '+%H:%M:%S')${C_RESET}"
  for i in $(seq 0 $((N - 1))); do
    name="${V_NAME[$i]}"
    rlog="$LOGDIR/$name.recv.log"
    if kill -0 "${WATCHER_PID[$i]}" 2>/dev/null; then
      all_done=0
      last_progress="$(grep -oE '[0-9.]+ ?MB/s' "$rlog" 2>/dev/null | tail -1)"
      elapsed=$(( $(date +%s) - START_TS[$i] ))
      printf '  %-28s %s(running, %ss)%s  %s\n' "$name" "$C_YELLOW" "$elapsed" "$C_RESET" "${last_progress:-...}"
    else
      if [[ "${RESULT_STATUS[$i]}" == "running" ]]; then
        if grep -q "NEVER_GOT_CODE" "$rlog" 2>/dev/null; then
          RESULT_STATUS[$i]="FAILED (no send code)"
        elif grep -q "RC=0 " "$rlog" 2>/dev/null; then
          elapsed="$(grep -oE 'ELAPSED=[0-9]+' "$rlog" | head -1 | cut -d= -f2)"
          [[ -z "$elapsed" || "$elapsed" == "0" ]] && elapsed=1
          mbps=$(awk -v s="$FILE_SIZE_MB" -v t="$elapsed" 'BEGIN{printf "%.1f", s/t}')
          RESULT_STATUS[$i]="OK"
          RESULT_MBPS[$i]="$mbps"
        else
          RESULT_STATUS[$i]="FAILED (see $rlog)"
        fi
      fi
      color="$C_GREEN"; [[ "${RESULT_STATUS[$i]}" != "OK" ]] && color="$C_RED"
      printf '  %-28s %s%-10s%s  %s MB/s\n' "$name" "$color" "${RESULT_STATUS[$i]}" "$C_RESET" "${RESULT_MBPS[$i]}"
    fi
  done
  echo ""
  [[ "$all_done" == "1" ]] && break
  if [[ $(date +%s) -gt $DEADLINE ]]; then
    warn "global deadline hit -- killing anything still running."
    for i in $(seq 0 $((N - 1))); do
      kill -9 "${WATCHER_PID[$i]}" 2>/dev/null
      kill -9 "${SEND_PID[$i]}" 2>/dev/null
      # Without this, a variant killed by the deadline (rather than
      # finishing/failing on its own) would stay stuck at "running"
      # forever in the final table below -- caught by a local dry run
      # against fake background jobs before ever touching a real pod.
      [[ "${RESULT_STATUS[$i]}" == "running" ]] && RESULT_STATUS[$i]="FAILED (global timeout)"
    done
    break
  fi
  sleep 5
done

for i in $(seq 0 $((N - 1))); do
  kill "${SEND_PID[$i]}" 2>/dev/null
done

hdr "FINAL RESULTS"
printf '%-28s %-22s %10s\n' "variant" "status" "MB/s"
printf '%s\n' "${C_DIM}------------------------------------------------------------${C_RESET}"
BEST_MBPS="0"; BEST_NAME="none"
for i in $(seq 0 $((N - 1))); do
  name="${V_NAME[$i]}"
  color="$C_GREEN"; [[ "${RESULT_STATUS[$i]}" != "OK" ]] && color="$C_RED"
  printf '%-28s %s%-22s%s %10s\n' "$name" "$color" "${RESULT_STATUS[$i]}" "$C_RESET" "${RESULT_MBPS[$i]}"
  if [[ "${RESULT_STATUS[$i]}" == "OK" ]]; then
    if awk -v a="${RESULT_MBPS[$i]}" -v b="$BEST_MBPS" 'BEGIN{exit !(a>b)}'; then
      BEST_MBPS="${RESULT_MBPS[$i]}"; BEST_NAME="$name"
    fi
  fi
done
echo ""
if [[ "$BEST_NAME" != "none" ]]; then
  ok "fastest: $BEST_NAME at $BEST_MBPS MB/s"
  BASELINE_MBPS="-"
  for i in $(seq 0 $((N - 1))); do
    [[ "${V_NAME[$i]}" == "public-relay-baseline" && "${RESULT_STATUS[$i]}" == "OK" ]] && BASELINE_MBPS="${RESULT_MBPS[$i]}"
  done
  if [[ "$BASELINE_MBPS" != "-" ]]; then
    SPEEDUP=$(awk -v a="$BEST_MBPS" -v b="$BASELINE_MBPS" 'BEGIN{printf "%.1f", a/b}')
    info "that is ${SPEEDUP}x the public-relay baseline ($BASELINE_MBPS MB/s)"
  fi
else
  err "every variant failed -- see logs under $LOGDIR on this machine and"
  err "/tmp/croc_relay.log on the pod for diagnosis."
fi
echo ""
info "raw logs kept at: $LOGDIR"
info "(pod teardown happens next, via the EXIT trap, unless KEEP_POD=1)"
