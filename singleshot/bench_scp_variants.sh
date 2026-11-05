#!/usr/bin/env bash
# Parallel-scp version of bench_croc_variants.sh's split-lanes phase --
# split a real file into N pieces, send each through its OWN scp
# process (each `scp` call already opens its own independent ssh
# connection by default; no manual port-forwarding/tunnel machinery
# needed the way croc's self-hosted relay required), running all N
# concurrently, then reassemble and verify on the far end.
#
# WHY THIS EXISTS, INSTEAD OF MORE CROC TUNING
# =============================================
# croc's split-lanes phase (bench_croc_variants.sh) measured far below
# iperf3's raw-tunnel numbers at the same lane count (iperf3: 268.3
# MB/s at 10 independent tunnels; croc real-file split-lanes: timed
# out / near-zero on the run that prompted this script). Two concrete,
# non-speculative reasons croc pays overhead iperf3 never does:
#   1) DOUBLE ENCRYPTION FOR NOTHING -- croc encrypts its own traffic
#      (PAKE handshake + chacha20-poly1305) ON TOP OF the SSH tunnel
#      that's already encrypting everything between this machine and
#      the pod. On loopback-through-SSH like that setup, croc's crypto
#      layer buys zero additional security and costs full CPU, doubled
#      again by however many lanes run concurrently.
#   2) DISK + CPU, NOT JUST NETWORK -- croc also compresses (unless
#      --no-compress), hashes for integrity, and runs its PAKE
#      handshake per transfer; iperf3 never touches disk or does any
#      of that, it just measures raw socket throughput.
# Plain scp already gets an independent ssh connection per invocation
# (no relay, no PAKE, one layer of encryption, no local recompression
# on top of already-compressed data) -- this script checks whether
# that alone gets close to the ~312 MB/s link ceiling that croc's
# split-lanes phase could not reach.
#
# USAGE
# =====
#   LOCAL SMOKE TEST (default -- no pod, no cost, validates the split/
#   concurrent-launch/reassemble/verify logic against localhost via a
#   REAL ssh connection, not a simulated one):
#     bash singleshot/bench_scp_variants.sh
#   Requires macOS "Remote Login" enabled (System Settings > General >
#   Sharing) and this machine's own key in ~/.ssh/authorized_keys --
#   if `ssh -o BatchMode=yes localhost true` fails, this script says so
#   and exits before touching anything else.
#
#   AGAINST A REAL POD (after the local smoke test passes):
#     POD_HOST=1.2.3.4 POD_PORT=22222 POD_USER=root \
#       SSH_KEY=~/.runpod/ssh/runpodctl-ssh-key \
#       bash singleshot/bench_scp_variants.sh
#
# ENV VARS (all optional)
#   POD_HOST     default: localhost (the smoke-test target)
#   POD_PORT     default: 22
#   POD_USER     default: root, UNLESS POD_HOST is localhost/127.0.0.1,
#                in which case it defaults to `whoami` -- a real pod's
#                only login is root; your own Mac's is your own user.
#   SSH_KEY      default: ~/.ssh/id_ed25519
#   LANES        number of independent concurrent scp streams (default
#                10, matching bench_croc_variants.sh's confirmed-best
#                iperf3 lane count)
#   FILE_SIZE_MB size of the random test file (default 200 for the
#                local smoke test -- big enough to catch real
#                concurrency/ordering bugs without making every local
#                run slow; override to 1024 to mirror the croc bench's
#                real-scale file size once testing against a real pod)
#   SCP_TRANSFER_TIMEOUT  per-lane hard cutoff in seconds (default 60)
#   SSH_CHECK_TIMEOUT     seconds to wait for the initial reachability
#                probe before giving up (default 5)

set -uo pipefail  # NOT -e -- a single lane failing must not abort the
                   # others or skip the final per-lane report.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib_common.sh
source "$HERE/lib_common.sh"  # run_with_timeout
# shellcheck source=./lib_scp.sh
source "$HERE/lib_scp.sh"   # scp_send_split -- the actual split/send/
                             # reassemble/verify logic, shared with
                             # scp_data_files.sh so the two can never
                             # drift apart.

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ -z "${PFD_NO_COLOR:-}" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_CYAN=$'\033[36m'
else
  C_RESET=""; C_BOLD=""; C_DIM=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
fi
hdr()  { printf '%s\n' "${C_BOLD}${C_CYAN}=================================================================="; printf '%s\n' " $1${C_RESET}"; printf '%s\n' "${C_BOLD}${C_CYAN}==================================================================${C_RESET}"; }
ok()   { printf '%s\n' "${C_GREEN}[OK]${C_RESET} $1"; }
warn() { printf '%s\n' "${C_YELLOW}[WARN]${C_RESET} $1"; }
err()  { printf '%s\n' "${C_RED}[FAIL]${C_RESET} $1" >&2; }
info() { printf '%s\n' "${C_BLUE}[..]${C_RESET} $1"; }

POD_HOST="${POD_HOST:-localhost}"
POD_PORT="${POD_PORT:-22}"
if [[ -z "${POD_USER:-}" ]]; then
  if [[ "$POD_HOST" == "localhost" || "$POD_HOST" == "127.0.0.1" ]]; then
    POD_USER="$(whoami)"
  else
    POD_USER="root"
  fi
fi
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LANES="${LANES:-10}"
FILE_SIZE_MB="${FILE_SIZE_MB:-200}"
SCP_TRANSFER_TIMEOUT="${SCP_TRANSFER_TIMEOUT:-60}"
SSH_CHECK_TIMEOUT="${SSH_CHECK_TIMEOUT:-5}"

IS_LOCAL=0
[[ "$POD_HOST" == "localhost" || "$POD_HOST" == "127.0.0.1" ]] && IS_LOCAL=1

SSH_OPTS=(-p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes)
SCP_OPTS=(-P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o BatchMode=yes)

hdr "Preflight: reachability of $POD_USER@$POD_HOST:$POD_PORT"
if ! run_with_timeout "$SSH_CHECK_TIMEOUT" ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" true; then
  err "ssh $POD_USER@$POD_HOST:$POD_PORT (key: $SSH_KEY) is not reachable/authorized."
  if [[ "$IS_LOCAL" == "1" ]]; then
    err "This is the LOCAL smoke test path -- to fix:"
    err "  1) System Settings > General > Sharing > enable 'Remote Login'"
    err "  2) mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    err "  3) cat ${SSH_KEY}.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  else
    err "Check POD_HOST/POD_PORT/POD_USER/SSH_KEY against 'runpodctl ssh info <pod-id>'."
  fi
  exit 1
fi
ok "reachable"
echo ""

# $$ (this script's own pid) makes REMOTE_DIR unique per run, so it's
# always freshly created here -- no need to clear it first (and doing
# so via a glob on a directory that's sometimes empty broke under zsh,
# the default remote login shell on macOS for the local smoke-test
# path: "rm -f dir/*" with zero matches is a hard error under zsh's
# default nomatch setting, unlike bash which passes the literal
# unmatched pattern through and lets `rm -f` silently ignore it).
REMOTE_DIR="/tmp/scp_bench_$$"
ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "mkdir -p '$REMOTE_DIR'"

LOCAL_DIR="$(mktemp -d)"
LOCAL_FILE="$LOCAL_DIR/bench_${FILE_SIZE_MB}mb.bin"
cleanup() {
  ssh "${SSH_OPTS[@]}" "$POD_USER@$POD_HOST" "rm -rf '$REMOTE_DIR'" 2>/dev/null || true
  rm -rf "$LOCAL_DIR"
}
trap cleanup EXIT

hdr "Preparing test file"
info "generating a random ${FILE_SIZE_MB}MB local test file (random, not zeros)..."
dd if=/dev/urandom "of=$LOCAL_FILE" bs=1m count="$FILE_SIZE_MB" 2>/dev/null
ok "test file ready: $LOCAL_FILE"
echo ""

hdr "Sending via $LANES concurrent scp processes"
if scp_send_split "$LOCAL_FILE" "$REMOTE_DIR" "result.bin" "$LANES" "$SCP_TRANSFER_TIMEOUT"; then
  echo ""
  hdr "RESULT"
  ok "PASS -- $LANES-lane parallel scp, size and sha256 both verified"
else
  echo ""
  hdr "RESULT"
  err "FAIL -- see errors above"
  exit 1
fi
