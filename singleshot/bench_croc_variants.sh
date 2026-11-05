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
#   bash singleshot/bench_croc_variants.sh --split-only   # skip the
#     4-variant benchmark, validate ONLY the split-lanes approach
#     against a real file (see "--split-only" below)
#   bash singleshot/bench_croc_variants.sh --iperf3        # skip croc
#     entirely, raw-network diagnostic (see "--iperf3" below)
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
#   CONTAINER_DISK_GB  default 20 -- only a ~1GB test file per variant
#                      is ever created, no real training data touches
#                      this pod
#   WAIT_TIMEOUT       seconds to wait for SSH to come up before giving
#                      up on pod creation entirely (default 60 -- 20s,
#                      then 35s, both proved too tight: real pods that
#                      boot fine can still take a while just to get
#                      their SSH port allocated, seen up to ~33s on
#                      one CA-MTL-1 attempt and a full timeout at 35s
#                      on another)
#   TRANSFER_TIMEOUT   per-variant hard cutoff in seconds (default 60 --
#                      confirmed runs at 1GB have all finished in
#                      11-19s per self-hosted variant and ~26s for the
#                      public-relay baseline; 60s is real margin above
#                      that without being an excessive wait on a
#                      genuine hang)
#   FILE_SIZE_MB       size of the random test file (default 1024,
#                      i.e. 1GB -- bumped from the original 100MB once
#                      the self-hosted relay was confirmed actually
#                      working (2x the public relay in a real run:
#                      50.0 vs 25.0 MB/s), to see if that holds at a
#                      size closer to the real train_80.h5/val_80.h5
#                      payload instead of just a quick sanity check)
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
#                      opens beyond the base port (default 8, i.e. 9
#                      ports total). NOT bumped to 99/100 despite an
#                      earlier plan to do exactly that -- confirmed
#                      directly against croc v11.5.2's own source
#                      (github.com/schollz/croc, src/croc/croc.go,
#                      `activateRelayDataChannels`: `limit :=
#                      min(len(c.Options.RelayPorts), 8)`, and the
#                      matching `index >= 8` guard) that croc hard-caps
#                      REAL parallel data connections at 8 per
#                      transfer, no matter how many ports the relay
#                      offers or `--transfers` requests. Opening more
#                      than 9 ports (and tunneling them) would be pure
#                      waste -- confirmed the "100 ports = 100 workers"
#                      premise was simply wrong before spending a pod
#                      rental finding that out the hard way. Each port
#                      gets its own `-L` forward in the SSH tunnel
#                      below.
#   SSH_KEY            default: ~/.runpod/ssh/runpodctl-ssh-key
#   KEEP_POD           set to 1 to skip the final `pod delete`
#   POD_ID             reuse an already-running pod instead of
#                      creating one
#   SPLIT_LANES        (default 10, matching the --iperf3 mode's
#                      IPERF3_LANES default -- that mode confirmed 10
#                      genuinely independent concurrent SSH tunnels hit
#                      268.3 MB/s raw, close to the ~312 MB/s link
#                      ceiling). After the main 4-variant benchmark,
#                      runs ONE additional phase: split the test file
#                      into this many pieces, send each through its OWN
#                      independent relay+tunnel pair (own local ports,
#                      own SSH process), running all lanes truly
#                      concurrently, then reassemble and verify. This
#                      is the lever for approaching that same real link
#                      ceiling with actual croc file transfers, not just
#                      raw iperf3 -- more connections through ONE shared
#                      tunnel twice measured WORSE, not better (see
#                      CROC_JOURNEY.md), so this uses genuinely separate
#                      tunnels instead. Set to 0 to skip this phase
#                      entirely.
#   SPLIT_BASE_PORT     base port for lane 0's relay (default 9500,
#                       deliberately far from CROC_RELAY_PORT's range
#                       to avoid any overlap)
#   SPLIT_PORT_STRIDE   ports between each lane's base (default 20,
#                       comfortably more than one lane's 9-port range)
#   SPLIT_TRANSFER_TIMEOUT  per-lane hard cutoff in seconds for the
#                       split-lanes receive step (default 90 -- was
#                       hardcoded to 60 before, not configurable at all;
#                       raised slightly since a 10-lane run divides
#                       throughput 10 ways per lane, unlike the 4-lane
#                       case this was first tuned against)
#
# --iperf3 (a MODE SWITCH, not an env var -- pass it as an argument)
#   Skips croc entirely. The split-lanes phase above (genuinely
#   independent SSH tunnels, not just more multiplexed streams through
#   one) STILL measured worse than a single tunnel (53.9 vs 85.3 MB/s),
#   which raised the question of whether croc/SSH have a real
#   single-threaded bottleneck even with idle cores on both ends, or
#   whether the ~85 MB/s ceiling is actually the network path/provider
#   itself and croc has nothing to do with it. This mode answers that
#   by tunneling raw `iperf3` traffic through the SAME SSH mechanism
#   croc uses (RunPod's Secure Cloud pods don't expose arbitrary ports
#   directly -- confirmed above, so this is the only way to reach the
#   pod for anything other than SSH), installing iperf3 on both ends if
#   missing, then running exactly ONE test: IPERF3_LANES independent
#   SSH tunnels + iperf3 servers, all launched concurrently, each doing
#   a single stream -- mirrors the split-lanes phase (genuinely
#   separate ssh processes, not more multiplexed streams through one).
#   No single-tunnel or low-lane-count variant is run here anymore --
#   this mode exists specifically to see what IPERF3_LANES fully
#   concurrent independent tunnels can do against the real link ceiling
#   (confirmed 2.5 Gb symmetric = ~312 MB/s).
#   IPERF3_PORT     base port for the per-lane ports (default 9520)
#   IPERF3_LANES    number of independent concurrent tunnels (default 10)
#   IPERF3_DURATION seconds per iperf3 run (default 10)
#   IPERF3_LOCAL_INSTALL_TIMEOUT
#                   hard cutoff in seconds for a local `brew install
#                   iperf3` if it's missing on this machine (default
#                   180 -- brew installs are slower than apt, given
#                   more margin, but still must not hang forever)
#   IPERF3_INSTALL_TIMEOUT
#                   hard cutoff in seconds for the remote `apt-get
#                   install iperf3` step (default 60). Found the hard
#                   way: iperf3's Debian postinst script asks a
#                   debconf question (whether to auto-start the iperf3
#                   server on boot); without DEBIAN_FRONTEND=
#                   noninteractive forced, `apt-get install -y` does
#                   NOT reliably suppress this over a non-TTY SSH
#                   command, and it hung indefinitely waiting on a
#                   prompt nothing could ever answer. Now this always
#                   forces DEBIAN_FRONTEND=noninteractive AND wraps the
#                   whole install in run_with_timeout, so a repeat of
#                   this failure mode fails loud within this many
#                   seconds instead of hanging the script forever. Also
#                   now explicitly checks for an already-installed
#                   iperf3 FIRST and skips the install step entirely if
#                   found, rather than relying on apt's own no-op
#                   behavior for an already-satisfied package.
#
# --split-only (a MODE SWITCH, not an env var -- pass it as an argument)
#   Skips the main 4-variant benchmark entirely and goes straight from
#   pod prep to the split-lanes phase below (SPLIT_LANES independent
#   relay+tunnel pairs, real file, reassembled and size-verified). Use
#   this to validate the split-lanes approach on its own budget instead
#   of burning most of GLOBAL_TIMEOUT_SECS on the 4-variant benchmark
#   first -- confirmed the hard way: with defaults, that benchmark's own
#   deadline (TRANSFER_TIMEOUT * 4 + 20 = 260s) plus pod boot plus
#   SPLIT_LANES-many relay/tunnel startups left too little of a 300s
#   GLOBAL_TIMEOUT_SECS for the split-lanes phase to ever finish, and it
#   got SIGTERM'd mid-transfer instead of producing a real number.
#
# GLOBAL_TIMEOUT_SECS  Hard wall-clock cap on the ENTIRE script
#                   (default 600 = 10 minutes, raised from an original
#                   300 after that default proved too tight for even
#                   ONE full run with the split-lanes phase enabled --
#                   see "--split-only" just above), no matter which
#                   phase gets stuck -- iperf3 install, tunnel setup, a
#                   hung `ssh`, anything. Every individual step above
#                   already has its own timeout, but auditing every
#                   single command in this file for one is exactly
#                   the whack-a-mole that already bit twice (the
#                   iperf3-install debconf hang, then a bare `wait`
#                   blocking on long-lived tunnel processes) -- a
#                   real pod sat there billing until killed by hand
#                   both times. See "GLOBAL TIMEOUT WRAPPER" below for
#                   the mechanism and its two independent layers.

set -uo pipefail  # NOT -e: this script's whole point is comparing
                   # variants where some are EXPECTED to fail/timeout;
                   # a failing variant must not abort the others or
                   # skip the summary table.

# ---------------------------------------------------------------- #
# GLOBAL TIMEOUT WRAPPER -- two independent layers, because a stuck
# pod is real money and this has already needed a manual kill twice:
#
#   Layer 1: this process re-execs ITSELF as a backgrounded child,
#   then a watchdog subshell sleeps GLOBAL_TIMEOUT_SECS and, if the
#   child is still alive, sends SIGTERM (confirmed locally: bash DOES
#   still run its EXIT trap -- i.e. `cleanup()`, which deletes the pod
#   -- on SIGTERM, this is not a bypass), waits 10s of grace, then
#   SIGKILL as a last resort. This bounds the SCRIPT's own execution,
#   whatever it happens to be stuck on.
#
#   Layer 2: a fully DETACHED (`disown`ed) background job, started
#   independently once POD_ID is known (see below, near "SSH:
#   root@..."), that unconditionally calls `runpodctl pod delete` at a
#   fixed deadline slightly AFTER layer 1's. This is the actual belt-
#   and-suspenders part: if layer 1's SIGTERM->cleanup->pod-delete path
#   ever fails for any reason (a hung `ssh` inside cleanup() itself,
#   for instance), this is a completely independent process that does
#   not depend on this script's signal handling, job control, or
#   control flow at all -- it only needs runpodctl and the pod id.
#
# The re-exec only happens ONCE: _BENCH_INNER marks "this is the real
# work", so the child doesn't re-wrap itself again.
# ---------------------------------------------------------------- #
GLOBAL_TIMEOUT_SECS="${GLOBAL_TIMEOUT_SECS:-600}"
if [[ -z "${_BENCH_INNER:-}" ]]; then
  export _BENCH_INNER=1
  "$0" "$@" &
  _inner_pid=$!
  (
    sleep "$GLOBAL_TIMEOUT_SECS"
    if kill -0 "$_inner_pid" 2>/dev/null; then
      echo "" >&2
      echo "[FAIL] GLOBAL ${GLOBAL_TIMEOUT_SECS}s TIMEOUT HIT -- sending SIGTERM (10s grace for pod cleanup via the EXIT trap, then SIGKILL)." >&2
      kill -TERM "$_inner_pid" 2>/dev/null
      sleep 10
      kill -KILL "$_inner_pid" 2>/dev/null
    fi
  ) &
  _watchdog_pid=$!
  wait "$_inner_pid" 2>/dev/null
  _rc=$?
  kill "$_watchdog_pid" 2>/dev/null
  wait "$_watchdog_pid" 2>/dev/null
  exit "$_rc"
fi
SCRIPT_START_TS=$(date +%s)

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

MODE_IPERF3=0
MODE_SPLIT_ONLY=0
for _arg in "$@"; do
  case "$_arg" in
    --iperf3) MODE_IPERF3=1 ;;
    --split-only) MODE_SPLIT_ONLY=1 ;;
    *) echo "Unknown argument: $_arg (only --iperf3 or --split-only are supported)" >&2; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib_croc.sh
source "$HERE/lib_croc.sh"

RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
if ! command -v "$RUNPODCTL" >/dev/null 2>&1; then
  err "runpodctl not found at $RUNPODCTL -- see singleshot/README.md's 'RunPod agent setup'."
  exit 1
fi
if [[ "$MODE_IPERF3" == "1" ]]; then
  if ! command -v iperf3 >/dev/null 2>&1; then
    info "iperf3 not found locally -- installing via Homebrew (${IPERF3_LOCAL_INSTALL_TIMEOUT:-180}s timeout)..."
    if ! command -v brew >/dev/null 2>&1; then
      err "Homebrew not found locally -- install iperf3 manually (brew install iperf3) and retry."
      exit 1
    fi
    if ! run_with_timeout "${IPERF3_LOCAL_INSTALL_TIMEOUT:-180}" brew install iperf3; then
      err "brew install iperf3 failed or timed out -- install manually and retry."
      exit 1
    fi
  fi
  if ! command -v iperf3 >/dev/null 2>&1; then
    err "iperf3 still not found locally after the install step completed."
    exit 1
  fi
  ok "iperf3 present locally"
else
  croc_require_local_binary
fi
if ! command -v jq >/dev/null 2>&1; then
  err "jq not installed locally (needed to parse runpodctl's JSON output)."
  exit 1
fi

GPU_ID="${GPU_ID:-NVIDIA A40}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
TRANSFER_TIMEOUT="${TRANSFER_TIMEOUT:-60}"
FILE_SIZE_MB="${FILE_SIZE_MB:-1024}"
CROC_RELAY_PORT="${CROC_RELAY_PORT:-9019}"
RELAY_TRANSFERS="${RELAY_TRANSFERS:-8}"
SSH_KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
KEEP_POD="${KEEP_POD:-0}"
SPLIT_LANES="${SPLIT_LANES:-10}"
SPLIT_BASE_PORT="${SPLIT_BASE_PORT:-9500}"
SPLIT_PORT_STRIDE="${SPLIT_PORT_STRIDE:-20}"
SPLIT_TRANSFER_TIMEOUT="${SPLIT_TRANSFER_TIMEOUT:-90}"
IPERF3_PORT="${IPERF3_PORT:-9520}"
IPERF3_LANES="${IPERF3_LANES:-10}"
IPERF3_DURATION="${IPERF3_DURATION:-10}"
IPERF3_INSTALL_TIMEOUT="${IPERF3_INSTALL_TIMEOUT:-60}"

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
  croc_tunnel_stop
  croc_relay_stop
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

# Layer 2 of the global timeout (see "GLOBAL TIMEOUT WRAPPER" near the
# top of this file): a fully detached backup that deletes THIS pod at
# a fixed deadline no matter what happens to the rest of this script
# -- it does not depend on cleanup()'s trap firing, this shell's job
# control, or anything else in this process. Deadline is set slightly
# AFTER layer 1's own SIGTERM/SIGKILL deadline, so the normal cleanup
# path gets a real chance to run first; this only actually deletes
# anything if that path somehow failed.
#
# Only armed when this script actually OWNS the pod's lifecycle
# (CREATED_POD=1, KEEP_POD unset) -- matching cleanup()'s own guard
# exactly. A reused pod (POD_ID passed in by the caller) or one the
# caller explicitly asked to keep must never get auto-deleted out from
# under them just because this run happened to also touch it.
if [[ "$CREATED_POD" == "1" && "$KEEP_POD" != "1" ]]; then
  SAFETY_NET_SECS=$(( GLOBAL_TIMEOUT_SECS + 30 - ($(date +%s) - SCRIPT_START_TS) ))
  [[ "$SAFETY_NET_SECS" -lt 1 ]] && SAFETY_NET_SECS=1
  ( sleep "$SAFETY_NET_SECS"; "$RUNPODCTL" pod delete "$POD_ID" >/dev/null 2>&1 ) &
  disown 2>/dev/null || true
  info "backup pod-deletion safety net armed: $POD_ID will be deleted in ${SAFETY_NET_SECS}s regardless, unless cleanup already ran"
else
  info "backup pod-deletion safety net skipped (reused/kept pod -- not this script's responsibility to delete)"
fi

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

if [[ "$MODE_IPERF3" == "1" ]]; then
  hdr "iperf3 diagnostic mode (skips croc entirely)"
  info "Tunnels raw iperf3 traffic through the same SSH mechanism croc uses"
  info "(RunPod doesn't expose arbitrary ports directly -- confirmed earlier)."
  info "Single test: $IPERF3_LANES independent tunnels, launched fully concurrently"
  info "(mirrors the split-lanes phase: genuinely separate ssh processes)."
  echo ""

  info "checking whether iperf3 is already installed on the pod..."
  if ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "command -v iperf3" >/dev/null 2>&1; then
    ok "iperf3 already present on pod -- skipping install"
  else
    info "not present -- installing (DEBIAN_FRONTEND=noninteractive, ${IPERF3_INSTALL_TIMEOUT}s timeout)..."
    # DEBIAN_FRONTEND=noninteractive is REQUIRED here, not cosmetic:
    # iperf3's Debian postinst asks a debconf question (auto-start on
    # boot?) that apt-get -y alone does not suppress over a non-TTY
    # SSH command -- confirmed the hard way, it hung indefinitely.
    # run_with_timeout (lib_croc.sh) bounds the whole thing so a repeat
    # of that failure mode fails loud instead of hanging forever.
    if ! run_with_timeout "$IPERF3_INSTALL_TIMEOUT" ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
        "DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iperf3" \
        >/dev/null 2>&1; then
      err "iperf3 install failed or timed out after ${IPERF3_INSTALL_TIMEOUT}s on the pod."
      err "If this keeps happening: ssh in and run"
      err "  'DEBIAN_FRONTEND=noninteractive dpkg --configure -a'"
      err "to clear any half-configured package state left behind by a"
      err "killed/timed-out install before retrying."
      exit 1
    fi
    if ! ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "command -v iperf3" >/dev/null 2>&1; then
      err "iperf3 still not found on the pod after the install step completed."
      exit 1
    fi
    ok "iperf3 installed on pod"
  fi
  echo ""

  # Mirrors croc_relay_start_lane's setsid pattern (lib_croc.sh) -- a
  # plain `ssh host "cmd &"` gets SIGHUP'd the instant that ssh
  # command's own shell exits, so backgrounding needs setsid nested
  # inside `bash -c '... &'`. Kills by exact pid only, never `pkill -f`
  # (see croc_relay_stop's docstring for why that's dangerous).
  iperf3_server_start() {
    local port="$1" outvar="$2" pid
    pid="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
      "rm -f /tmp/iperf3_server_$port.log; bash -c 'setsid iperf3 -s -p $port -B 127.0.0.1 </dev/null >/tmp/iperf3_server_$port.log 2>&1 & echo \$!'" \
      2>/dev/null || true)"
    if [[ -z "$pid" ]]; then
      err "iperf3 server on port $port did not start"
      return 1
    fi
    sleep 1
    eval "$outvar=\"\$pid\""
  }
  iperf3_server_stop() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "kill $pid 2>/dev/null; true" 2>/dev/null || true
  }

  # IPERF3_CLIENT_TIMEOUT bounds every iperf3 client below (iperf3's own
  # -t is a target TEST duration, not a hard process-kill guarantee -- a
  # client stuck before the connection ever completes its handshake
  # would otherwise hang past -t indefinitely). Margin over IPERF3_DURATION
  # covers connection setup + JSON teardown, not just the raw test window.
  IPERF3_CLIENT_TIMEOUT=$((IPERF3_DURATION + 15))

  hdr "Test: $IPERF3_LANES independent tunnels, concurrent single streams"
  # NOTE: each lane's tunnel setup (croc_tunnel_start_lane) reachability
  # probe is a bare `echo >/dev/tcp/...` -- fine against croc's relay
  # (tolerates garbage connections), but iperf3's server does NOT: it
  # logs a one-time "unable to receive cookie" error for that probe
  # before going back to listening normally. Harmless, but expect to
  # see exactly one such line per lane in /tmp/iperf3_server_*.log on
  # the pod -- confirmed by hand, not a sign anything actually failed.
  declare -a T3_TUNNEL_PIDS=() T3_SERVER_PIDS=() T3_PORTS=() T3_LOGS=() T3_CLIENT_PIDS=()
  T3_OK=1
  for i in $(seq 0 $((IPERF3_LANES - 1))); do
    p=$((IPERF3_PORT + 100 + i * 10))
    T3_PORTS[$i]="$p"
    if ! iperf3_server_start "$p" "T3_SERVER_PIDS[$i]"; then T3_OK=0; break; fi
    if ! croc_tunnel_start_lane "$p" 0 "T3_TUNNEL_PIDS[$i]"; then T3_OK=0; break; fi
  done

  T3_MBPS="FAILED"
  if [[ "$T3_OK" == "1" ]]; then
    ok "all $IPERF3_LANES lanes' tunnels established"
    for i in $(seq 0 $((IPERF3_LANES - 1))); do
      p="${T3_PORTS[$i]}"
      log="$(mktemp)"
      T3_LOGS[$i]="$log"
      ( run_with_timeout "$IPERF3_CLIENT_TIMEOUT" iperf3 -c localhost -p "$p" -t "$IPERF3_DURATION" -J >"$log" 2>&1 ) &
      T3_CLIENT_PIDS[$i]=$!
    done
    # `wait` with NO arguments waits for every background job of this
    # shell, not just the lanes just launched -- and the SSH TUNNEL
    # processes above are ALSO background jobs of this same shell,
    # deliberately long-lived for the rest of this phase. A bare `wait`
    # here hung forever on those tunnels even after all iperf3
    # clients had already finished (confirmed live: pod-side logs
    # showed completed tests + errors, zero iperf3 client processes
    # left running locally, yet the script sat at "all lanes'
    # tunnels established" indefinitely). Wait on the exact client PIDs
    # only, never a bare `wait`, in a section that also has tunnels
    # backgrounded.
    for i in $(seq 0 $((IPERF3_LANES - 1))); do
      wait "${T3_CLIENT_PIDS[$i]}" 2>/dev/null || true
    done
    T3_TOTAL="0"
    for i in $(seq 0 $((IPERF3_LANES - 1))); do
      mb="$(jq -r '.end.sum_received.bits_per_second / 8 / 1048576' "${T3_LOGS[$i]}" 2>/dev/null)"
      [[ -z "$mb" || "$mb" == "null" ]] && mb="0"
      T3_TOTAL="$(awk -v a="$T3_TOTAL" -v b="$mb" 'BEGIN{printf "%.1f", a+b}')"
    done
    T3_MBPS="$T3_TOTAL"
  else
    err "one or more lane tunnels failed to establish -- see errors above."
  fi
  ok "Test ($IPERF3_LANES independent tunnels, concurrent): ${T3_MBPS} MB/s aggregate"

  for i in $(seq 0 $((IPERF3_LANES - 1))); do
    croc_stop_lane "${T3_TUNNEL_PIDS[$i]:-}"
    iperf3_server_stop "${T3_SERVER_PIDS[$i]:-}"
  done

  echo ""
  hdr "IPERF3 DIAGNOSTIC RESULTS"
  printf '%-60s %10s\n' "test" "MB/s"
  printf '%s\n' "${C_DIM}------------------------------------------------------------------${C_RESET}"
  printf '%-60s %10s\n' "$IPERF3_LANES independent tunnels, concurrent (aggregate)" "$T3_MBPS"
  echo ""
  info "compare against croc's own confirmed numbers (CROC_JOURNEY.md):"
  info "  public-relay-baseline ~40 MB/s, self-hosted-default ~85-93 MB/s,"
  info "  split-lanes ~54 MB/s aggregate. Real link ceiling: ~312 MB/s"
  info "  (2.5 Gb symmetric)."
  echo ""
  info "(pod teardown happens next, via the EXIT trap, unless KEEP_POD=1)"
  exit 0
fi

hdr "Preparing the pod (croc present, classic mode, test file)"
croc_ensure_installed_remote
croc_ensure_classic_mode_remote

REMOTE_DIR="/tmp/croc_bench"
ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "mkdir -p $REMOTE_DIR && rm -f $REMOTE_DIR/*"

LOCAL_DIR="$(mktemp -d)"
LOCAL_FILE="$LOCAL_DIR/bench_${FILE_SIZE_MB}mb.bin"
info "generating a random ${FILE_SIZE_MB}MB local test file (random, not zeros --"
info "avoids flattering any compression) ..."
dd if=/dev/urandom "of=$LOCAL_FILE" bs=1m count="$FILE_SIZE_MB" 2>/dev/null
EXPECTED_BYTES="$(stat -f%z "$LOCAL_FILE" 2>/dev/null || stat -c%s "$LOCAL_FILE")"
ok "test file ready: $LOCAL_FILE ($EXPECTED_BYTES bytes)"
echo ""

BEST_MBPS="0"; BEST_NAME="none"
if [[ "$MODE_SPLIT_ONLY" == "1" ]]; then
  info "--split-only: skipping the 4-variant benchmark entirely (and its"
  info "single shared relay/tunnel, only needed by that benchmark) --"
  info "going straight to the $SPLIT_LANES-lane split phase below so it"
  info "gets the full GLOBAL_TIMEOUT_SECS budget to itself."
  echo ""
else

hdr "Starting the self-hosted relay on the pod"
croc_relay_start "$CROC_RELAY_PORT" "$RELAY_TRANSFERS"
echo ""

hdr "Establishing SSH tunnel for the relay's port range"
# See lib_croc.sh's croc_tunnel_start docstring for the full "why" --
# in short, this sidesteps RunPod's custom-TCP-port-exposure question
# entirely (5 real pods proved it doesn't work the way earlier
# attempts assumed) by tunneling through the SSH port that's always
# worked instead.
croc_tunnel_start "$CROC_RELAY_PORT" "$RELAY_TRANSFERS"
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
# send-only extra flags, MUST_BE_SEQUENTIAL [0/1]). HISTORICAL NOTE,
# now moot but worth keeping: an earlier version had send and receive
# reach the relay via DIFFERENT addresses (send: the pod's public IP;
# receive: localhost), which was a real bug -- the receive side
# (already running ON the pod) tried to dial its own public IP back
# at itself and got "i/o timeout" on every self-hosted variant. Now
# that both sides tunnel through SSH (see croc_tunnel_start above),
# they use the IDENTICAL "localhost:$CROC_RELAY_PORT" flag -- no more
# asymmetry to get wrong.
#
# MUST_BE_SEQUENTIAL: all self-hosted variants share the SAME single
# SSH tunnel (one TCP connection to the pod). At the original 100MB
# test size this never mattered -- transfers finished in ~2s, no time
# for contention to bite. At the real 1GB size it did: 3 self-hosted
# variants launched concurrently moved a combined ~5.4 MB/s (each
# stuck around 1.7-1.8 MB/s, all killed by TRANSFER_TIMEOUT at ~31%),
# drastically worse than the ~50 MB/s a SINGLE self-hosted transfer
# got in an earlier, uncontended run. Production only ever sends 2
# real files concurrently (train_80.h5 + val_80.h5), never 3+
# redundant copies of the same file through different settings, so
# comparing these settings fairly means giving each the FULL tunnel to
# itself -- hence sequential. The public-relay baseline uses a
# completely separate path (never touches our tunnel), so it stays
# concurrent with whichever self-hosted variant is currently running.
# ---------------------------------------------------------------- #
declare -a V_NAME=() V_USE_RELAY=() V_NO_COMPRESS=() V_SEND_ONLY_FLAGS=() V_SEQUENTIAL=()

# "self-hosted-default" leaves --transfers at croc's own send-side
# default (4) -- which is what actually produced the earlier confirmed
# 2.0x result, NOT the relay's port count. "self-hosted-transfers8"
# explicitly requests the real, confirmed cap (8, see RELAY_TRANSFERS
# above) to test the one axis of real, not-yet-measured upside: going
# from croc's default 4 real connections to its actual maximum of 8.
V_NAME+=("public-relay-baseline");   V_USE_RELAY+=("0"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local");               V_SEQUENTIAL+=("0")
V_NAME+=("self-hosted-default");     V_USE_RELAY+=("1"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local");               V_SEQUENTIAL+=("1")
V_NAME+=("self-hosted-transfers8");  V_USE_RELAY+=("1"); V_NO_COMPRESS+=("0"); V_SEND_ONLY_FLAGS+=("--no-local --transfers 8"); V_SEQUENTIAL+=("1")
V_NAME+=("self-hosted-no-compress"); V_USE_RELAY+=("1"); V_NO_COMPRESS+=("1"); V_SEND_ONLY_FLAGS+=("--no-local");               V_SEQUENTIAL+=("1")

N="${#V_NAME[@]}"
LOGDIR="$(mktemp -d)"
declare -a SEND_PID=() WATCHER_PID=() RESULT_MBPS=() RESULT_STATUS=() START_TS=() LAUNCHED=()
for i in $(seq 0 $((N - 1))); do
  RESULT_STATUS[$i]="queued"
  RESULT_MBPS[$i]="-"
  LAUNCHED[$i]=0
  WATCHER_PID[$i]=0
done

launch_variant() {
  local i="$1" name relay_flag send_relay recv_relay nocompress sonly slog rlog
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
  croc_send "$send_relay" "$nocompress" "$sonly" "$LOCAL_FILE" "$slog"
  SEND_PID[$i]=$!
  START_TS[$i]=$(date +%s)
  RESULT_STATUS[$i]="running"
  ( # background: wait for the code, then run the timed receive
    code="$(croc_parse_code_from_log "$slog" 15)"
    if [[ -z "$code" ]]; then
      echo "NEVER_GOT_CODE" >>"$rlog"
      exit 1
    fi
    # A UNIQUE remote dir per variant, not one shared "$REMOTE_DIR" --
    # concurrent variants receiving into the same path were racing
    # each other's `rm -f` + write with zero isolation, and nothing
    # before EXPECTED_BYTES/SIZE_OK below would ever have caught it (a
    # passing exit code only means croc's protocol thought it
    # finished, not that the bytes are actually right).
    croc_receive "$TRANSFER_TIMEOUT" "$REMOTE_DIR/$name" "$(basename "$LOCAL_FILE")" \
      "$recv_relay" "$nocompress" "$code" "$rlog" "$EXPECTED_BYTES"
  ) &
  WATCHER_PID[$i]=$!  # reuse WATCHER_PID[] to hold the background-wrapper's own pid
  LAUNCHED[$i]=1
}

hdr "Launching variants ($((N - 1)) self-hosted ones run SEQUENTIALLY against $LOGDIR)"
SEQ_QUEUE=()
for i in $(seq 0 $((N - 1))); do
  if [[ "${V_SEQUENTIAL[$i]}" == "1" ]]; then
    SEQ_QUEUE+=("$i")
  else
    launch_variant "$i"
  fi
done
SEQ_POS=0
if [[ "${#SEQ_QUEUE[@]}" -gt 0 ]]; then
  launch_variant "${SEQ_QUEUE[0]}"
fi
echo ""

hdr "Live status (refreshing every 5s, ${TRANSFER_TIMEOUT}s hard cutoff per variant)"
# Budget: every sequential variant may need up to TRANSFER_TIMEOUT,
# one after another, plus the concurrent one, plus a flat buffer --
# NOT just TRANSFER_TIMEOUT+20 like a single-round benchmark would
# need (that was fine for 4 truly-concurrent variants; it is not
# enough once 3 of them run one after another).
DEADLINE=$(( $(date +%s) + TRANSFER_TIMEOUT * (${#SEQ_QUEUE[@]} + 1) + 20 ))
while true; do
  all_done=1
  printf '%s\n' "${C_DIM}$(date '+%H:%M:%S')${C_RESET}"
  for i in $(seq 0 $((N - 1))); do
    name="${V_NAME[$i]}"
    if [[ "${LAUNCHED[$i]}" != "1" ]]; then
      all_done=0
      printf '  %-28s %s(queued)%s\n' "$name" "$C_DIM" "$C_RESET"
      continue
    fi
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
        elif grep -q "SIZE_MISMATCH" "$rlog" 2>/dev/null; then
          # A passing croc exit code alone doesn't mean the bytes that
          # arrived actually match what was sent -- see croc_receive's
          # docstring. Never skip this check just because RC=0 looked fine.
          RESULT_STATUS[$i]="FAILED (size mismatch, see $rlog)"
        elif grep -q "RC=0 " "$rlog" 2>/dev/null && grep -q "SIZE_OK" "$rlog" 2>/dev/null; then
          elapsed="$(grep -oE 'ELAPSED=[0-9]+' "$rlog" | head -1 | cut -d= -f2)"
          [[ -z "$elapsed" || "$elapsed" == "0" ]] && elapsed=1
          mbps=$(awk -v s="$FILE_SIZE_MB" -v t="$elapsed" 'BEGIN{printf "%.1f", s/t}')
          RESULT_STATUS[$i]="OK"
          RESULT_MBPS[$i]="$mbps"
        else
          RESULT_STATUS[$i]="FAILED (see $rlog)"
        fi
        # This variant JUST finished -- if it was the current head of
        # the sequential queue, launch whatever's next in that queue
        # (giving it the tunnel to itself, per this section's own
        # reason for existing).
        if [[ "$SEQ_POS" -lt "${#SEQ_QUEUE[@]}" && "${SEQ_QUEUE[$SEQ_POS]}" == "$i" ]]; then
          SEQ_POS=$((SEQ_POS + 1))
          if [[ "$SEQ_POS" -lt "${#SEQ_QUEUE[@]}" ]]; then
            launch_variant "${SEQ_QUEUE[$SEQ_POS]}"
          fi
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
      # Queued-but-never-launched variants (deadline hit before their
      # turn in the sequential queue) never got a SEND_PID -- ":-"
      # keeps this a no-op kill instead of an unbound-variable error.
      kill -9 "${WATCHER_PID[$i]:-}" 2>/dev/null
      kill -9 "${SEND_PID[$i]:-}" 2>/dev/null
      # Without this, a variant killed by the deadline (rather than
      # finishing/failing on its own) would stay stuck at "running"
      # forever in the final table below -- caught by a local dry run
      # against fake background jobs before ever touching a real pod.
      [[ "${RESULT_STATUS[$i]}" == "running" || "${RESULT_STATUS[$i]}" == "queued" ]] && RESULT_STATUS[$i]="FAILED (global timeout)"
    done
    break
  fi
  sleep 5
done

for i in $(seq 0 $((N - 1))); do
  kill "${SEND_PID[$i]:-}" 2>/dev/null
done

hdr "FINAL RESULTS"
printf '%-28s %-22s %10s\n' "variant" "status" "MB/s"
printf '%s\n' "${C_DIM}------------------------------------------------------------${C_RESET}"
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

fi  # MODE_SPLIT_ONLY

if [[ "$SPLIT_LANES" == "0" ]]; then
  info "SPLIT_LANES=0 -- skipping the multi-lane phase."
  info "(pod teardown happens next, via the EXIT trap, unless KEEP_POD=1)"
else
  echo ""
  hdr "EXPERIMENTAL: $SPLIT_LANES independent lanes (own relay+tunnel each)"
  warn "Not yet confirmed at any scale -- see CROC_JOURNEY.md before trusting this number."

  SPLIT_DIR="$(mktemp -d)"
  PIECE_BYTES=$(( (EXPECTED_BYTES + SPLIT_LANES - 1) / SPLIT_LANES ))
  info "splitting $LOCAL_FILE into $SPLIT_LANES pieces (~$((PIECE_BYTES / 1048576)) MB each)..."
  split -b "$PIECE_BYTES" -d -a 2 "$LOCAL_FILE" "$SPLIT_DIR/piece_"

  SPLIT_REMOTE_DIR="$REMOTE_DIR/split-lanes"
  ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "mkdir -p '$SPLIT_REMOTE_DIR' && rm -f '$SPLIT_REMOTE_DIR'/*"

  declare -a LANE_RELAY_PIDS=() LANE_TUNNEL_PIDS=() LANE_SEND_PIDS=() LANE_SEND_LOGS=()
  LANES_OK=1

  info "starting $SPLIT_LANES independent relay+tunnel pairs..."
  for i in $(seq 0 $((SPLIT_LANES - 1))); do
    lane_port=$((SPLIT_BASE_PORT + i * SPLIT_PORT_STRIDE))
    if ! croc_relay_start_lane "$lane_port" 8 "LANE_RELAY_PIDS[$i]"; then
      LANES_OK=0
      break
    fi
    if ! croc_tunnel_start_lane "$lane_port" 8 "LANE_TUNNEL_PIDS[$i]"; then
      LANES_OK=0
      break
    fi
  done

  if [[ "$LANES_OK" == "1" ]]; then
    ok "all $SPLIT_LANES lanes' relay+tunnel established"
    info "sending all $SPLIT_LANES pieces CONCURRENTLY..."
    SPLIT_T0=$(date +%s)
    for i in $(seq 0 $((SPLIT_LANES - 1))); do
      lane_port=$((SPLIT_BASE_PORT + i * SPLIT_PORT_STRIDE))
      piece_file="$SPLIT_DIR/piece_$(printf '%02d' "$i")"
      slog="$(mktemp)"
      LANE_SEND_LOGS[$i]="$slog"
      croc_send "--relay localhost:$lane_port" "" "--no-local" "$piece_file" "$slog"
      LANE_SEND_PIDS[$i]=$!
    done

    declare -a LANE_RECV_WATCHER_PIDS=() LANE_RECV_LOGS=()
    for i in $(seq 0 $((SPLIT_LANES - 1))); do
      lane_port=$((SPLIT_BASE_PORT + i * SPLIT_PORT_STRIDE))
      piece_file="$SPLIT_DIR/piece_$(printf '%02d' "$i")"
      piece_name="piece_$(printf '%02d' "$i")"
      piece_bytes="$(stat -f%z "$piece_file" 2>/dev/null || stat -c%s "$piece_file")"
      code="$(croc_parse_code_from_log "${LANE_SEND_LOGS[$i]}" 15)"
      rlog="$(mktemp)"
      LANE_RECV_LOGS[$i]="$rlog"
      if [[ -z "$code" ]]; then
        # No process backgrounded for this lane -- leave its watcher
        # pid empty rather than capturing a stale/wrong $! from a
        # PRIOR iteration (or an unset one on the very first lane).
        echo "NEVER_GOT_CODE" >>"$rlog"
        LANE_RECV_WATCHER_PIDS[$i]=""
      else
        ( croc_receive "$SPLIT_TRANSFER_TIMEOUT" "$SPLIT_REMOTE_DIR" "$piece_name" "--relay localhost:$lane_port" "" \
            "$code" "$rlog" "$piece_bytes" ) &
        LANE_RECV_WATCHER_PIDS[$i]=$!
      fi
    done
    for i in $(seq 0 $((SPLIT_LANES - 1))); do
      wait "${LANE_RECV_WATCHER_PIDS[$i]:-}" 2>/dev/null || true
    done
    SPLIT_ELAPSED=$(( $(date +%s) - SPLIT_T0 ))

    for i in $(seq 0 $((SPLIT_LANES - 1))); do
      if grep -q "RC=0 " "${LANE_RECV_LOGS[$i]}" 2>/dev/null && grep -q "SIZE_OK" "${LANE_RECV_LOGS[$i]}" 2>/dev/null; then
        ok "lane $i: OK"
      else
        LANES_OK=0
        err "lane $i: FAILED -- $(cat "${LANE_RECV_LOGS[$i]}" 2>/dev/null | tail -3)"
      fi
      kill "${LANE_SEND_PIDS[$i]:-}" 2>/dev/null || true
    done
  fi

  for i in $(seq 0 $((SPLIT_LANES - 1))); do
    croc_stop_lane "${LANE_RELAY_PIDS[$i]:-}"
    croc_stop_lane "${LANE_TUNNEL_PIDS[$i]:-}"
  done

  if [[ "$LANES_OK" == "1" ]]; then
    info "reassembling $SPLIT_LANES pieces on the pod..."
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
      "cd '$SPLIT_REMOTE_DIR' && cat \$(ls piece_* | sort) > '$REMOTE_DIR/split_result.bin' && rm -f piece_*"
    SPLIT_REMOTE_SIZE="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
      "stat -c%s '$REMOTE_DIR/split_result.bin' 2>/dev/null")"
    if [[ "$SPLIT_REMOTE_SIZE" == "$EXPECTED_BYTES" ]]; then
      SPLIT_MBPS=$(awk -v b="$EXPECTED_BYTES" -v s="$SPLIT_ELAPSED" 'BEGIN{ if (s<1) s=1; printf "%.1f", (b/1048576)/s }')
      ok "reassembled size verified ($SPLIT_REMOTE_SIZE bytes)"
      echo ""
      ok "SPLIT RESULT: ${SPLIT_ELAPSED}s, ${SPLIT_MBPS} MB/s aggregate across $SPLIT_LANES lanes"
      if [[ "$BEST_NAME" != "none" ]]; then
        SPLIT_SPEEDUP=$(awk -v a="$SPLIT_MBPS" -v b="$BEST_MBPS" 'BEGIN{printf "%.1f", a/b}')
        info "that is ${SPLIT_SPEEDUP}x the best single-tunnel result ($BEST_MBPS MB/s, $BEST_NAME)"
      fi
    else
      err "reassembled size mismatch: got=${SPLIT_REMOTE_SIZE:-<none>} want=$EXPECTED_BYTES"
    fi
  else
    err "split-lanes phase failed -- see errors above. Not falling back to"
    err "anything; this phase is purely informational on top of the"
    err "already-completed main benchmark above."
  fi
  rm -rf "$SPLIT_DIR"
  ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "rm -f '$REMOTE_DIR/split_result.bin'" 2>/dev/null || true
  echo ""
  info "(pod teardown happens next, via the EXIT trap, unless KEEP_POD=1)"
fi
