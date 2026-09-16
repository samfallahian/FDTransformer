#!/usr/bin/env bash
# lib_croc.sh -- the reusable "class" for every proven-correct croc
# operation this repo uses: installing/configuring croc on a remote
# pod, running a self-hosted relay, tunneling its ports through SSH,
# and driving a send/receive pair safely.
#
# WHY THIS EXISTS
# ================
# Every function in this file encodes a real bug found and fixed
# during a multi-day investigation (documented in full, with rich
# narrative context, in CROC_JOURNEY.md next to this file) into
# getting croc's self-hosted relay working reliably against a RunPod
# pod. Duplicating this logic ad hoc in every script that touches croc
# is exactly how three of those bugs got introduced in the first place
# (bench_croc_variants.sh reinvented pieces of this three separate
# times before landing on the tunnel approach). Source this file once
# and call its functions instead.
#
# "CLASS" DESIGN NOTE -- bash has no real classes, and this repo only
# has bash 3.2 available locally (macOS's stock system bash, confirmed
# via `bash --version`; no Homebrew bash installed) -- no namerefs
# (`declare -n`, bash 4.3+), no associative arrays as function params.
# So "instance state" here is a small set of WELL-KNOWN GLOBAL
# variables, exactly the way a method implicitly reads/writes `self`'s
# attributes in a real class. Every function below documents which
# globals it reads and which it sets. The caller is responsible for
# setting the "constructor" globals (SSH_OPTS, POD_HOST) before calling
# anything else -- there is no init function that returns an object,
# just a contract on what must already be true.
#
# REQUIRED GLOBALS (set these before calling anything below):
#   SSH_OPTS   bash array, e.g. (-p "$POD_PORT" -i "$SSH_KEY"
#              -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
#   POD_HOST   the pod's SSH host/IP
#
# GLOBALS THIS FILE SETS (the closest thing to "instance attributes"):
#   RELAY_PID   set by croc_relay_start, read by croc_relay_stop
#   TUNNEL_PID  set by croc_tunnel_start, read by croc_tunnel_stop
#
# All functions print progress via whatever `ok`/`warn`/`err`/`info`
# functions are already defined by the CALLER (this file does not
# define its own -- it assumes the sourcing script already has them,
# matching bench_croc_variants.sh's convention; if they're undefined,
# bash just runs them as external commands and fails loudly, which is
# an acceptable "you forgot to define these" signal during development
# but should never happen in a finished caller).

# run_with_timeout lives in lib_common.sh now (it's generic, not
# croc-specific) -- source it so every function below that calls it
# keeps working without duplicating the definition.
# shellcheck source=./lib_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_common.sh"

# ---------------------------------------------------------------- #
# croc_require_local_binary
# Exits 1 if `croc` isn't installed on THIS machine.
# ---------------------------------------------------------------- #
croc_require_local_binary() {
  if ! command -v croc >/dev/null 2>&1; then
    err "croc not installed locally. brew install croc (macOS) or: curl https://getcroc.schollz.com | bash"
    exit 1
  fi
}

# ---------------------------------------------------------------- #
# croc_ensure_installed_remote
# Reads: SSH_OPTS, POD_HOST
# Installs croc on the pod via the official installer if missing --
# mirrors bootstrap_remote.sh's already-proven step (curl | bash, NOT
# apt -- croc isn't packaged there). An earlier version of this
# benchmark just ASSUMED croc was baked into the pod image (true once,
# by coincidence, on an H200 pod; false on every A40 pod since) and
# hard-failed instead of installing it.
# ---------------------------------------------------------------- #
croc_ensure_installed_remote() {
  if ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "command -v croc" >/dev/null 2>&1; then
    ok "croc present on pod: $(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "croc --version" 2>/dev/null)"
    return 0
  fi
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
}

# ---------------------------------------------------------------- #
# croc_ensure_classic_mode_remote
# Reads: SSH_OPTS, POD_HOST
# croc 11.x refuses a receive code as a bare CLI argument by default
# ("does not accept receive codes on the UNIX command line because
# they can appear in the process list") -- needed for any SCRIPTED
# receive. `--classic` is NOT a per-invocation flag despite looking
# like one: it TOGGLES a setting persisted at
# ~/.config/croc/classic_enabled, flipping it again every time it's
# passed. So this checks the flag file directly and only runs the
# toggle once, if needed -- calling `croc --classic` to "check" would
# just flip it.
# ---------------------------------------------------------------- #
croc_ensure_classic_mode_remote() {
  local flag
  flag="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
  if [[ "$flag" != "enabled" ]]; then
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "printf 'y\\n' | croc --classic" >/dev/null 2>&1 || true
    flag="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "cat /root/.config/croc/classic_enabled 2>/dev/null || echo missing")"
    if [[ "$flag" != "enabled" ]]; then
      err "could not enable croc's classic mode on the pod (needed for a scripted receive)."
      exit 1
    fi
  fi
  ok "croc classic mode enabled on pod"
}

# ---------------------------------------------------------------- #
# croc_relay_start PORT TRANSFERS
# Reads: SSH_OPTS, POD_HOST
# Sets:  RELAY_PID
# Starts `croc relay` on the pod, detached from the SSH session that
# launches it. `setsid` inside a nested `bash -c '... &'` is required
# -- a plain `ssh host "cmd &"` gets the background job SIGHUP'd the
# instant that ssh command's own shell exits (confirmed: the process
# was simply gone half a second later, no log file even created).
# `echo $!` right after backgrounding captures the pid (setsid execs
# straight into croc, so the pid doesn't change) for croc_relay_stop.
# Exits 1 if the relay doesn't report a pid within ~2s.
# ---------------------------------------------------------------- #
croc_relay_start() {
  local port="$1" transfers="$2"
  RELAY_PID="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
    "rm -f /tmp/croc_relay.log; bash -c 'setsid croc relay --port $port --transfers $transfers </dev/null >/tmp/croc_relay.log 2>&1 & echo \$!'" \
    2>/dev/null || true)"
  sleep 2
  if [[ -z "$RELAY_PID" ]]; then
    err "relay did not start -- see /tmp/croc_relay.log on the pod."
    exit 1
  fi
  ok "relay running, pid=$RELAY_PID, ports $port-$((port + transfers))"
}

# ---------------------------------------------------------------- #
# croc_relay_stop
# Reads: SSH_OPTS, POD_HOST, RELAY_PID
# Kills the relay by its EXACT pid. NEVER use `pkill -f "croc relay"`
# here, even for a one-off cleanup -- found the hard way that `pkill
# -f <pattern>` scans every process's FULL command line, which
# trivially includes the invoking shell's OWN command line (it
# literally contains the pkill invocation's text), and pkill's
# self-exclusion only covers the pkill process itself, not its parent
# shell -- it silently killed the whole SSH session instead (`ssh -v`
# showed `exit-signal`, not a clean exit code, and the calling script
# got back a bare `exit 255` with zero output). A numeric `kill <pid>`
# has no such hazard.
# ---------------------------------------------------------------- #
croc_relay_stop() {
  if [[ -n "${RELAY_PID:-}" ]]; then
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "kill $RELAY_PID 2>/dev/null; true" 2>/dev/null || true
    RELAY_PID=""
  fi
}

# ---------------------------------------------------------------- #
# croc_tunnel_start PORT TRANSFERS
# Reads: SSH_OPTS, POD_HOST
# Sets:  TUNNEL_PID
# Forwards EVERY relay port (PORT through PORT+TRANSFERS) from this
# machine's localhost, through the pod's SSH connection, to localhost
# on the pod -- exactly where croc_relay_start bound the relay. This
# is the actual fix for the multi-pod investigation documented in
# CROC_JOURNEY.md: RunPod's Secure Cloud pods do not expose arbitrary
# custom TCP ports the way earlier attempts assumed (neither a literal
# port number, which gets silently remapped with no way to discover
# the real external port, nor the documented "add 70000 for a
# symmetric mapping" trick, whose confirming env var never appeared
# even after a 30s retry loop against 5 separate real pods). Tunneling
# through SSH needs NOTHING beyond the SSH port that's always worked.
#
# `-o ExitOnForwardFailure=yes` makes the whole tunnel process exit
# immediately if ANY single `-L` can't bind, rather than silently
# running with some ports missing -- caught here, not three variants
# deep into a benchmark.
# ---------------------------------------------------------------- #
croc_tunnel_start() {
  local port="$1" transfers="$2"
  local tunnel_flags=()
  local i p
  for i in $(seq 0 "$transfers"); do
    p=$((port + i))
    tunnel_flags+=(-L "$p:localhost:$p")
  done
  ssh -N -o ExitOnForwardFailure=yes "${SSH_OPTS[@]}" "${tunnel_flags[@]}" "root@$POD_HOST" &
  TUNNEL_PID=$!
  sleep 2
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    err "SSH tunnel process exited immediately -- ExitOnForwardFailure means"
    err "at least one of the $((transfers + 1)) local port forwards couldn't"
    err "bind (maybe already in use on THIS machine -- check e.g."
    err "'lsof -i :$port')."
    exit 1
  fi
  ok "tunnel established, pid=$TUNNEL_PID, forwarding localhost:$port-$((port + transfers)) -> pod"

  if ! run_with_timeout 5 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
    err "cannot reach localhost:$port even through the SSH tunnel -- the"
    err "tunnel process is alive but the actual forward isn't answering."
    exit 1
  fi
  ok "port $port reachable via the tunnel from this machine"
}

# ---------------------------------------------------------------- #
# croc_tunnel_stop
# Reads: TUNNEL_PID
# ---------------------------------------------------------------- #
croc_tunnel_stop() {
  if [[ -n "${TUNNEL_PID:-}" ]]; then
    kill "$TUNNEL_PID" 2>/dev/null || true
    TUNNEL_PID=""
  fi
}

# ---------------------------------------------------------------- #
# croc_send RELAY_FLAG NOCOMPRESS SEND_ONLY_FLAGS FILE LOGFILE
# Starts `croc send` in the background against a LOCAL file. Prints
# nothing to stdout/stderr itself -- everything goes to LOGFILE so a
# caller running several of these concurrently can watch each one
# independently.
#
# `croc`'s usage is `[GLOBAL OPTIONS] [COMMAND] [COMMAND OPTIONS]
# [file]`. RELAY_FLAG/NOCOMPRESS are GLOBAL (must precede "send") --
# --relay, --yes, --no-compress. SEND_ONLY_FLAGS are `send`-SPECIFIC
# (must follow "send", never valid as a global flag or on receive) --
# --no-local, --transfers. Verified directly against the installed
# binary's own --help, not assumed -- mixing these two categories up
# is exactly what broke two variants the first time this ran ("flag
# provided but not defined: -transfers"), and a first draft of this
# very docstring made the identical mistake with --no-local before
# being checked against `croc --help` directly.
#
# Pass RELAY_FLAG as "" for the public relay (croc's real default,
# croc.schollz.com). SEND_ONLY_FLAGS should almost always include
# --no-local: without it, `croc send` ALSO tries to open a local-
# network-discovery relay on ports 9009-9013 -- harmless with one
# send running, but multiple concurrent sends on the same machine
# collide on those local binds ("address already in use", eventually
# a `panic:` in one log).
#
# Sets (via echo, caller captures with $()): nothing -- caller reads
# $! immediately after calling this for the send pid, same as before.
# Actually returns nothing meaningful; kept as a thin, testable wrapper
# purely to centralize the exact flag ordering in one place.
# ---------------------------------------------------------------- #
croc_send() {
  local relay_flag="$1" nocompress="$2" send_only_flags="$3" file="$4" logfile="$5"
  # shellcheck disable=SC2086
  croc --yes $relay_flag $nocompress send $send_only_flags "$file" >"$logfile" 2>&1 &
}

# ---------------------------------------------------------------- #
# croc_parse_code_from_log LOGFILE MAX_WAIT_SECS
# Polls LOGFILE (croc send's own log, as passed to croc_send) once a
# second for up to MAX_WAIT_SECS, looking for the "On the other
# computer, run:" line croc prints, then extracts the codephrase from
# the line right after it.
#
# NOT simply "the 2nd word" -- when --relay (or other flags) are in
# play, croc echoes them back on that same line too (e.g. "croc
# --relay 1.2.3.4:9019 --transfers 16 widen-try-grave"), so the code
# is the LAST token before any trailing "(code copied...)" annotation,
# never a fixed position. Verified against real croc 11.5.2 output,
# both with and without --relay, before trusting this in production.
#
# Prints the code to stdout (caller captures with $()), or nothing if
# it never showed up within MAX_WAIT_SECS.
# ---------------------------------------------------------------- #
croc_parse_code_from_log() {
  local logfile="$1" max_wait="$2"
  local code="" i
  for i in $(seq 1 "$max_wait"); do
    if grep -q "run:" "$logfile" 2>/dev/null; then
      code="$(grep -A1 "run:" "$logfile" | tail -1 | sed -E 's/^\s*croc\s+//; s/\s*\(.*$//' | awk '{print $NF}')"
      [[ -n "$code" ]] && break
    fi
    sleep 1
  done
  echo "$code"
}

# ---------------------------------------------------------------- #
# croc_receive TIMEOUT_SECS REMOTE_DIR REMOTE_FILENAME RELAY_FLAG NOCOMPRESS CODE LOGFILE [EXPECTED_BYTES]
# Reads: SSH_OPTS, POD_HOST
# Runs the receive side ON THE POD over ssh, bounded by
# run_with_timeout so a hung transfer fails loud instead of hanging
# the whole script. Appends output to LOGFILE, then appends a final
# "RC=<code> ELAPSED=<seconds>" line so a caller (or a human) can
# `grep` the outcome out of the log without needing this function's
# actual bash return value (useful when called from inside a
# backgrounded subshell, where a plain return code is otherwise
# awkward to observe from the parent).
#
# REMOTE_DIR should be UNIQUE PER CALLER when multiple receives run
# concurrently (e.g. "$REMOTE_DIR/$variant_name", not one shared
# directory) -- concurrent receives into the same path were racing
# against each other's `rm -f` + write with no isolation, and were
# never caught because nothing checked the result actually matched
# what was sent (see EXPECTED_BYTES below, added for exactly this).
#
# If EXPECTED_BYTES is given, appends a "SIZE_OK=<n>" or
# "SIZE_MISMATCH got=<n> want=<m>" line after a successful (rc=0)
# receive, checked via `stat` over the same ssh connection -- a
# passing exit code alone only means croc's OWN protocol thinks it
# finished; it says nothing about whether the bytes that arrived
# actually match what was sent, which matters a lot when several
# concurrent receives share infrastructure.
#
# RELAY_FLAG here should be the SAME "localhost:<port>" form used for
# the relay wherever this pod's relay is actually bound -- NOT the
# pod's public IP. Collapsing send-side and receive-side into one
# shared "$POD_HOST:<port>" flag was a real bug found here: the
# receive side (already running ON the pod) tried to dial its own
# public IP back at itself and got "i/o timeout" on every self-hosted
# variant, every time.
# ---------------------------------------------------------------- #
croc_receive() {
  local timeout_secs="$1" remote_dir="$2" remote_filename="$3" relay_flag="$4" nocompress="$5" code="$6" logfile="$7"
  local expected_bytes="${8:-}"
  local t0 rc elapsed got_bytes
  ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "mkdir -p $remote_dir" >/dev/null 2>&1
  t0=$(date +%s)
  # shellcheck disable=SC2086
  run_with_timeout "$timeout_secs" \
    ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
    "cd $remote_dir && rm -f $remote_filename && croc --yes $relay_flag $nocompress '$code'" \
    >>"$logfile" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  echo "RC=$rc ELAPSED=$elapsed" >>"$logfile"
  if [[ "$rc" == "0" && -n "$expected_bytes" ]]; then
    got_bytes="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
      "stat -c%s '$remote_dir/$remote_filename' 2>/dev/null" 2>/dev/null)"
    if [[ "$got_bytes" == "$expected_bytes" ]]; then
      echo "SIZE_OK=$got_bytes" >>"$logfile"
    else
      echo "SIZE_MISMATCH got=${got_bytes:-<none>} want=$expected_bytes" >>"$logfile"
    fi
  fi
}

# ------------------------------------------------------------------ #
# MULTI-LANE PRIMITIVES -- NEW, NOT YET CONFIRMED AT SCALE.
#
# Everything above this point is proven: confirmed 2.4x at real 1GB
# scale (CROC_JOURNEY.md). That result came from ONE relay, ONE SSH
# tunnel, croc's own default parallelism (4 real connections, capped
# at 8 by croc itself -- going higher measured SLOWER, not faster).
# Getting closer to a real ~312 MB/s link ceiling (2.5 Gb symmetric,
# confirmed by the operator) can't come from more connections through
# that ONE tunnel -- twice-confirmed not to help. The next lever is
# genuinely PARALLEL, INDEPENDENT tunnels: split a file into N pieces,
# send each through its OWN dedicated relay+tunnel pair (own local
# ports, own remote ports, own SSH process), running all N truly
# concurrently, then reassemble.
#
# These functions use an explicit OUTVAR parameter + `eval` to return
# a value (bash 3.2 has no namerefs) -- unlike the single-lane
# functions above, which set a single well-known global (RELAY_PID/
# TUNNEL_PID). Multiple simultaneous lanes each need their OWN pid
# tracked separately, so the caller passes the name of the variable
# (or array element, e.g. "MY_ARR[2]") it wants the result written
# into.
# ------------------------------------------------------------------ #

# croc_relay_start_lane PORT TRANSFERS OUTVAR
# Same mechanics as croc_relay_start (setsid inside a nested `bash -c
# '... &'`, see that function's docstring for why), but for a specific
# lane's port range, writing the resulting pid into $OUTVAR instead of
# the shared RELAY_PID global. Returns 1 (does not exit) on failure --
# multi-lane orchestration needs to clean up OTHER already-started
# lanes before giving up, unlike the single-lane path's hard exit.
croc_relay_start_lane() {
  local port="$1" transfers="$2" outvar="$3"
  local pid
  pid="$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" \
    "rm -f /tmp/croc_relay_$port.log; bash -c 'setsid croc relay --port $port --transfers $transfers </dev/null >/tmp/croc_relay_$port.log 2>&1 & echo \$!'" \
    2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    err "lane relay on port $port did not start"
    return 1
  fi
  eval "$outvar=\"\$pid\""
}

# croc_tunnel_start_lane PORT TRANSFERS OUTVAR
# Same mechanics as croc_tunnel_start, for one lane's port range,
# writing the resulting pid into $OUTVAR. Each lane MUST use a
# distinct PORT range (local ports can't be shared across separate
# ssh processes) -- the caller is responsible for spacing lanes far
# enough apart (e.g. stride of 20+ given TRANSFERS=8 needs 9 ports).
#
# Found the hard way running 4 lanes for real: this originally only
# checked "is the ssh process still alive", unlike croc_tunnel_start
# above, which ALSO probes actual port reachability. "Alive" is not
# "ready" -- with 4 tunnels' worth of SSH connections negotiating
# against the pod at nearly the same time, one lane's process was
# still alive at the 1s check but its forward wasn't actually
# accepting connections yet. Symptom: that lane's receiver connected
# to its own (local-to-the-pod) relay fine and sat at "waiting for
# sender...", while the corresponding local `croc send` -- going
# through the not-yet-ready tunnel -- never got through at all, timing
# out at TRANSFER_TIMEOUT. Now probes the SAME way croc_tunnel_start
# does, not just a liveness check.
croc_tunnel_start_lane() {
  local port="$1" transfers="$2" outvar="$3"
  local tunnel_flags=() i p pid
  for i in $(seq 0 "$transfers"); do
    p=$((port + i))
    tunnel_flags+=(-L "$p:localhost:$p")
  done
  ssh -N -o ExitOnForwardFailure=yes "${SSH_OPTS[@]}" "${tunnel_flags[@]}" "root@$POD_HOST" &
  pid=$!
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    err "lane tunnel for port $port failed to establish (process exited)"
    return 1
  fi
  # A few retries, not one hard check -- under real load (multiple
  # lanes' SSH connections negotiating against the pod at close to the
  # same time), one lane took longer than a single fixed wait to
  # actually start accepting connections even though its process was
  # already alive. Give it a real chance before declaring it dead.
  local _attempt
  for _attempt in 1 2 3; do
    if run_with_timeout 5 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
      eval "$outvar=\"\$pid\""
      return 0
    fi
    sleep 2
  done
  err "lane tunnel for port $port is alive but never became reachable"
  kill "$pid" 2>/dev/null || true
  return 1
}

# croc_stop_lane PID
# Kills one lane's relay or tunnel process by exact pid -- NEVER
# `pkill -f`, same reasoning as croc_relay_stop/croc_tunnel_stop above.
# For relay pids specifically this needs to run over ssh (the process
# is on the pod); for tunnel pids it's a local kill. Safe to call with
# either kind since a local `kill` on a pid that only exists remotely
# (or vice versa) just fails harmlessly.
croc_stop_lane() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  kill "$pid" 2>/dev/null || true
  ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "kill $pid 2>/dev/null; true" 2>/dev/null || true
}
