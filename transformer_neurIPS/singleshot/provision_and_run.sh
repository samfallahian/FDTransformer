#!/usr/bin/env bash
# Full pod lifecycle, driven entirely by runpodctl (no OAuth/MCP needed):
# create a pod -> wait for SSH -> build + scp the payload tarball -> run a time-boxed
# training job -> pull ONLY the new artifacts back -> terminate the pod.
#
# PREREQUISITE -- do this yourself, NOT through this script or in chat
# ======================================================================
# A RunPod API key must be configured for runpodctl before running this.
#
# 1. https://www.console.runpod.io/user/settings -> expand "API Keys" ->
#    "Create API Key".
# 2. Permissions: select "All (full access)", NOT "Restricted" or
#    "Read Only" -- runpodctl needs to create/list/delete pods and fetch
#    SSH info, all write/management operations. RunPod's "Restricted"
#    tier is scoped to per-Serverless-endpoint access, not documented to
#    cover Pod management, so "All" is the only tier confirmed to work
#    for what this script does.
# 3. Copy the key (shown once), then run ONE of these YOURSELF, in your
#    own terminal:
#
#      runpodctl doctor                    # interactive prompt, saves the key
#      export RUNPOD_API_KEY=your-key      # this shell session only
#
# Never paste the key into a chat message, a script argument, or a
# command line another process could log -- both methods above keep it
# out of this script entirely; runpodctl reads it from
# ~/.runpod/config.toml or the env var on its own. This script only ever
# calls `runpodctl user` to CHECK that credentials exist -- it never
# reads, prints, or stores the key itself.
#
# USAGE
#   bash singleshot/provision_and_run.sh
#
# ENV VARS (all optional)
#   GPU_ID             exact GPU type string from `runpodctl gpu list`
#                       (default: auto-picks the first available H200 or
#                       H300 listing; override if neither is in stock)
#   TEMPLATE_ID         RunPod template id (default: runpod-torch-v280,
#                       verified via `runpodctl template search pytorch`
#                       to actually exist -- do NOT hand-type an --image
#                       tag guessed from --help's illustrative example;
#                       an earlier version of this script did exactly
#                       that with a nonexistent tag and RunPod's backend
#                       surfaced it as an opaque "graphql_error" instead
#                       of "image not found"). bootstrap_remote.sh
#                       installs its own venv + torch on top regardless
#                       of which template's image is used, so this only
#                       needs to be ANY real, working CUDA+Python3 image.
#   IMAGE               set this instead of TEMPLATE_ID to use a specific
#                       docker image directly (`--image`) rather than a
#                       template -- verify it actually exists first, e.g.
#                       `runpodctl template search <name>` or
#                       `runpodctl template list --type official`.
#   CONTAINER_DISK_GB   container disk size in GB (default 60 -- payload is
#                       ~13.1GB now that train_80.h5/val_80.h5 are stored
#                       uncompressed, see decompress_h5.py; leaves headroom
#                       for the venv + checkpoints)
#   ARM                 arm to run for this budget (default h11_ridge_distill
#                       -- see rationale below; override for a different arm)
#   MAX_HOURS           wall-clock cap passed to the trainer (default 0.5 = 30 min)
#   MAX_STEPS           optimizer-step cap (default 2500 -- deliberately
#                       sized BELOW a conservative throughput estimate so
#                       WARMUP_FRAC doesn't break, same lesson as
#                       OVERVIEW.md v5.0 section 28.6)
#   WANDB_API_KEY       if set, wandb logs in non-interactively for this
#                       run (`WANDB_API_KEY=... wandb login` supports this
#                       with no prompt); if unset, the run uses --no-wandb
#                       -- an unattended pipeline cannot satisfy wandb's
#                       normal interactive/browser login
#   POD_ID              reuse an already-running pod instead of creating one
#   KEEP_POD            set to 1 to skip the final `pod delete` (for
#                       debugging a run without losing the box)
#   FORCE_TERMINATE_ON_PULL_FAILURE
#                       the final artifact pull-back (checkpoints,
#                       sweep_logs) retries a few times, then -- if it's
#                       still failing -- refuses to delete the pod so a
#                       transient SSH/network failure can't cost you the
#                       run's actual output. Set this to 1 to delete the
#                       pod anyway even if that pull-back never succeeded
#                       (i.e. you've confirmed there's nothing worth
#                       keeping, or already retrieved it manually).
#   SSH_KEY             local private key to use (default ~/.ssh/id_ed25519
#                       -- must match a key added via `runpodctl ssh add-key`
#                       or already present on the pod's image)
#   TRANSFER_METHOD     "scp" (default) or "croc" -- CONFIRMED faster,
#                       2.4x the public relay at real 1GB-file scale
#                       (93.1 vs 39.4 MB/s), see scp_data_files.sh's
#                       header comment and singleshot/CROC_JOURNEY.md
#                       for the full story. Uses a self-hosted relay
#                       tunneled through SSH -- needs NO extra ports
#                       exposed on the pod (an earlier version of this
#                       script requested CROC_RELAY_PORT's range at
#                       `pod create` time on the theory that croc
#                       needed them directly exposed; confirmed across
#                       5 real pod rentals that RunPod's Secure Cloud
#                       pods don't support that at all, which is why
#                       this now tunnels through SSH instead and no
#                       longer touches `--ports` for this). Deliberately
#                       NO scp fallback -- fails loud instead.
#   CROC_RELAY_PORT     base port for TRANSFER_METHOD=croc's self-hosted
#                       relay (default 9019, tunneled -- see above).
#
# WHY h11_ridge_distill BY DEFAULT
# ==================================
# It's the one arm from this session verified only on MPS (mechanically:
# loads, trains, resumes, doesn't crash) and has never touched real CUDA.
# OVERVIEW.md v6.1 section 31.5 explicitly said the next step was "a real
# CUDA shallow screen" -- 30 minutes is exactly that budget. Set ARM= to
# run something else instead (e.g. s8_h9_moreseqs_scaled for the standing
# production candidate).
#
# WHAT THIS SCRIPT DOES NOT DO
# ==============================
# It doesn't know RunPod's exact JSON field names for `gpu list --output
# json` / `pod create --output json` / `ssh info --output json` ahead of
# a real authenticated call -- runpodctl's --help text doesn't include a
# schema, and this session has no key to test against. Every parse below
# is defensive (tries several plausible field names, jq // fallbacks) and
# ALWAYS prints the raw command output too, so a schema mismatch is
# visible and fixable rather than silently wrong. If a jq parse comes up
# empty, the script stops and tells you exactly what to look at instead
# of guessing further.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"

RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
if ! command -v "$RUNPODCTL" >/dev/null 2>&1; then
  echo "runpodctl not found at $RUNPODCTL -- see singleshot/README.md's" >&2
  echo "'RunPod agent setup' section to install it." >&2
  exit 1
fi

echo "=================================================================="
echo " Checking RunPod credentials (never prints the key itself)..."
echo "=================================================================="
AUTH_CHECK="$("$RUNPODCTL" user -o json 2>&1 || true)"
if echo "$AUTH_CHECK" | jq -e '.error' >/dev/null 2>&1; then
  echo "No RunPod credentials configured. Run ONE of these yourself first:" >&2
  echo "  runpodctl doctor" >&2
  echo "  export RUNPOD_API_KEY=your-key   (create one -- 'All (full access)' permissions -- at https://www.console.runpod.io/user/settings)" >&2
  echo "" >&2
  echo "(raw check: $AUTH_CHECK)" >&2
  exit 1
fi
echo "  OK -- credentials present."
echo ""

GPU_ID="${GPU_ID:-}"
TEMPLATE_ID="${TEMPLATE_ID:-runpod-torch-v280}"
IMAGE="${IMAGE:-}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-60}"
ARM="${ARM:-h11_ridge_distill}"
MAX_HOURS="${MAX_HOURS:-0.5}"
MAX_STEPS="${MAX_STEPS:-2500}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
KEEP_POD="${KEEP_POD:-0}"
TRANSFER_METHOD="${TRANSFER_METHOD:-scp}"
CROC_RELAY_PORT="${CROC_RELAY_PORT:-9019}"

if [[ -z "$GPU_ID" && -z "${POD_ID:-}" ]]; then
  echo "Looking up an available H200/H300 GPU (override with GPU_ID=...)..."
  GPU_JSON="$("$RUNPODCTL" gpu list -o json 2>&1)"
  GPU_ID="$(echo "$GPU_JSON" | jq -r '
    [.[]? // .gpus[]? |
     select((.id // .gpuId // .displayName // .name // "") | test("H200|H300"; "i")) |
     (.id // .gpuId // .displayName // .name)
    ] | .[0] // empty' 2>/dev/null || true)"
  if [[ -z "$GPU_ID" ]]; then
    echo "" >&2
    echo "Could not auto-find an H200/H300 in the GPU list (either none are" >&2
    echo "in stock right now, or the JSON field names didn't match what this" >&2
    echo "script guessed -- see the raw output below). Set GPU_ID explicitly" >&2
    echo "and re-run:" >&2
    echo "" >&2
    echo "$GPU_JSON" >&2
    exit 1
  fi
  echo "  using GPU_ID=$GPU_ID"
fi
echo ""

if [[ -n "${POD_ID:-}" ]]; then
  echo "Reusing existing pod $POD_ID (POD_ID was set)."
  POD_JSON="$("$RUNPODCTL" pod get "$POD_ID" -o json)"
else
  if [[ -n "$IMAGE" ]]; then
    echo "Creating pod (gpu=$GPU_ID, image=$IMAGE, disk=${CONTAINER_DISK_GB}GB)..."
    IMAGE_FLAGS=(--image "$IMAGE")
  else
    echo "Creating pod (gpu=$GPU_ID, template=$TEMPLATE_ID, disk=${CONTAINER_DISK_GB}GB)..."
    IMAGE_FLAGS=(--template-id "$TEMPLATE_ID")
  fi
  # 22/tcp only -- TRANSFER_METHOD=croc's self-hosted relay is reached
  # through an SSH tunnel now, not a directly-exposed pod port (see
  # CROC_RELAY_PORT above). An earlier version of this line requested
  # CROC_RELAY_PORT's range directly on the theory that croc needed it
  # exposed that way; confirmed across 5 real pod rentals that RunPod's
  # Secure Cloud pods don't support arbitrary custom TCP port exposure
  # at all (see singleshot/CROC_JOURNEY.md), which is why the tunnel
  # approach exists and this no longer requests anything beyond SSH.
  echo "This blocks until SSH actually answers (--wait), up to 10 minutes."
  POD_JSON="$("$RUNPODCTL" pod create \
    --gpu-id "$GPU_ID" \
    "${IMAGE_FLAGS[@]}" \
    --container-disk-in-gb "$CONTAINER_DISK_GB" \
    --ports "22/tcp" \
    --public-ip \
    --wait --wait-timeout 10m \
    -o json)"
  POD_ID="$(echo "$POD_JSON" | jq -r '.id // .podId // empty')"
  if [[ -z "$POD_ID" ]]; then
    echo "Pod created but couldn't parse its id from the JSON below --" >&2
    echo "check manually with 'runpodctl pod list':" >&2
    echo "$POD_JSON" >&2
    exit 1
  fi
  echo "  pod id: $POD_ID"
fi
echo ""

PULL_BACK_OK=1  # flipped to 0 by rsync_with_retry() on a genuine failure
                # (see the pull-back step below) -- cleanup_pod() below
                # refuses to auto-terminate on 0, since the artifacts this
                # whole run was for might not have made it off the pod yet.

cleanup_pod() {
  if [[ "$KEEP_POD" == "1" ]]; then
    echo "KEEP_POD=1 set -- leaving pod $POD_ID running. Terminate it"
    echo "yourself when done: runpodctl pod delete $POD_ID"
    return
  fi
  if [[ "$PULL_BACK_OK" != "1" && "${FORCE_TERMINATE_ON_PULL_FAILURE:-0}" != "1" ]]; then
    echo ""
    echo "==================================================================" >&2
    echo " NOT terminating pod $POD_ID -- the artifact pull-back below" >&2
    echo " reported a genuine failure (not just \"no files matched\"), so" >&2
    echo " whatever this run produced may still only exist on the pod." >&2
    echo " Retrieve it yourself, THEN terminate:" >&2
    echo "" >&2
    echo "   POD_HOST=$POD_HOST POD_PORT=$POD_PORT SSH_KEY=$SSH_KEY \\" >&2
    echo "     rsync -avP -e \"ssh -p \$POD_PORT -i \$SSH_KEY\" \\" >&2
    echo "     \"root@\$POD_HOST:/workspace/cgan/transformer_neurIPS/saved_models/r2_${ARM}_*\" \\" >&2
    echo "     \"$TRANSFORMER_DIR/saved_models/\"" >&2
    echo "   runpodctl pod delete $POD_ID" >&2
    echo "" >&2
    echo " Or, if you're confident nothing of value is on this pod, re-run" >&2
    echo " with FORCE_TERMINATE_ON_PULL_FAILURE=1 to delete it anyway." >&2
    echo "==================================================================" >&2
    return
  fi
  echo ""
  echo "Terminating pod $POD_ID..."
  "$RUNPODCTL" pod delete "$POD_ID" || echo "  (delete failed -- terminate manually: runpodctl pod delete $POD_ID)" >&2
}
trap cleanup_pod EXIT

# Retries an rsync pull up to $3 (default 3) times with a short backoff,
# printing progress before/after every attempt -- fixes two real problems
# found in a live run: (1) almost no console output while this step ran,
# so a hang/slow transfer looked identical to nothing happening, and (2)
# a genuine SSH connection failure (exit 255, "kex_exchange_identification:
# read: Operation timed out") got silently swallowed by a bare `|| echo
# "no files found"`, which is flat-out WRONG for that failure -- it then
# let the pod get terminated with the run's actual artifacts possibly
# still stranded on it. rsync exit 24 ("vanishing source files", e.g. the
# archive's own rotation pruning a file mid-transfer, see OVERVIEW.md's
# rsync retrieval notes) is treated as success; every other nonzero exit
# is a real failure, retried, and -- if still failing after $3 attempts --
# reported loudly with PULL_BACK_OK=0 so cleanup_pod() above refuses to
# terminate the pod out from under undelivered data.
rsync_with_retry() {
  local desc="$1" src="$2" dest="$3" max_attempts="${4:-3}"
  local attempt=1 rc out
  while (( attempt <= max_attempts )); do
    echo "  [pull $attempt/$max_attempts] $desc ..."
    # `out=$(...) ; rc=$?` would trip `set -e` on a failing rsync (the
    # assignment's own exit status IS rsync's, and a plain failing
    # statement is not exempt from errexit) -- the `&&`/`||` chain keeps
    # this line itself always "successful" so rc is captured either way.
    out="$(rsync -avP -e "ssh -p $POD_PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new" \
      "$src" "$dest" 2>&1)" && rc=0 || rc=$?
    echo "$out"
    if [[ "$rc" == "0" || "$rc" == "24" ]]; then
      echo "  [pull $attempt/$max_attempts] $desc: OK (rsync exit $rc)"
      return 0
    fi
    echo "  [pull $attempt/$max_attempts] $desc: FAILED (rsync exit $rc)" >&2
    if (( attempt < max_attempts )); then
      echo "  retrying in 15s..." >&2
      sleep 15
    fi
    attempt=$((attempt + 1))
  done
  echo "  [pull] $desc: giving up after $max_attempts attempts (exit $rc)." >&2
  return 1
}

echo "Fetching SSH connection info..."
SSH_INFO_JSON="$("$RUNPODCTL" ssh info "$POD_ID" -o json 2>&1)"
echo "$SSH_INFO_JSON"
POD_HOST="$(echo "$SSH_INFO_JSON" | jq -r '.host // .ip // .sshHost // .publicIp // empty' 2>/dev/null || true)"
POD_PORT="$(echo "$SSH_INFO_JSON" | jq -r '.port // .sshPort // empty' 2>/dev/null || true)"
if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
  # Fall back to regex-extracting from a plain ssh command string, if
  # that's what this schema actually returns (e.g. a "command"/"sshCommand"
  # field, or plain text when -o json wasn't fully honoured for this
  # subcommand).
  CMD_LINE="$(echo "$SSH_INFO_JSON" | grep -oE 'ssh[^"]*' | head -1 || true)"
  POD_HOST="${POD_HOST:-$(echo "$CMD_LINE" | grep -oE '@[^ ]+' | tr -d '@' | head -1)}"
  POD_PORT="${POD_PORT:-$(echo "$CMD_LINE" | grep -oE '\-p ?[0-9]+' | grep -oE '[0-9]+' | head -1)}"
fi
if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
  echo "" >&2
  echo "Could not parse host/port out of the ssh-info output above." >&2
  echo "Read it yourself, then re-run the remaining steps manually:" >&2
  echo "  POD_HOST=<host> POD_PORT=<port> bash $HERE/scp_env_files.sh" >&2
  exit 1
fi
echo "  host=$POD_HOST port=$POD_PORT"
echo ""

export POD_HOST POD_PORT SSH_KEY TRANSFER_METHOD CROC_RELAY_PORT RELAY_TRANSFERS
echo "=================================================================="
echo " Phase 1/2: env files (code, checkpoint, ridge map, AE decoder)"
echo "=================================================================="
bash "$HERE/scp_env_files.sh"

echo ""
echo "=================================================================="
echo " Phase 2/2: data files, IN THE BACKGROUND -- bootstrap runs now,"
echo " concurrently, over a second ssh connection"
echo "=================================================================="
bash "$HERE/scp_data_files.sh" &
DATA_SCP_PID=$!

echo ""
echo "Bootstrapping the venv while the data transfer above runs in parallel..."
ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "root@$POD_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/cgan/transformer_neurIPS
bash singleshot/bootstrap_remote.sh
REMOTE

echo ""
echo "Waiting for the data transfer to finish (if it hasn't already)..."
wait "$DATA_SCP_PID"

echo ""
echo "=================================================================="
echo " Launching the ${MAX_HOURS}h-capped run (arm=$ARM)"
echo "=================================================================="
WANDB_FLAG="--no-wandb"
WANDB_SETUP=":"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_FLAG=""
  WANDB_SETUP="WANDB_API_KEY='$WANDB_API_KEY' wandb login --relogin"
fi

# Both bootstrap and the data transfer are confirmed done above -- this
# SSH command only launches training, nothing else. Blocks until the
# trainer exits.
#
# MANUAL FALLBACK -- keep this at your fingertips if this script ever
# dies/hangs mid-run again (see OVERVIEW.md v6.0 section 30.1 for the
# incident that made this whole package this cautious). SSH in yourself
# and run, verbatim (also documented in singleshot/README.md section
# 5a, kept in sync with this exact command so the two never drift):
#
#   cd /workspace/cgan/transformer_neurIPS
#   source /workspace/cgan/.venv/bin/activate
#   python train_production_transformer_deep_dive.py \
#     --arm h11_ridge_distill --round 2 \
#     --max-steps 2500 --max-hours 0.5 \
#     --val-every 500 --no-wandb
ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "root@$POD_HOST" bash -s <<REMOTE
set -euo pipefail
cd /workspace/cgan/transformer_neurIPS
# bootstrap_remote.sh's default VENV_DIR is \$REPO_ROOT/.venv, i.e.
# /workspace/cgan/.venv -- one level up from transformer_neurIPS/.
source ../.venv/bin/activate
$WANDB_SETUP
python train_production_transformer_deep_dive.py \\
  --arm $ARM --round 2 --max-steps $MAX_STEPS --max-hours $MAX_HOURS \\
  --val-every 500 $WANDB_FLAG
REMOTE

echo ""
echo "=================================================================="
echo " Pulling back ONLY the new artifacts (not the training data)"
echo "=================================================================="
CKPT_OK=1
if ! rsync_with_retry "checkpoints (saved_models/r2_${ARM}_*)" \
    "root@$POD_HOST:/workspace/cgan/transformer_neurIPS/saved_models/r2_${ARM}_"* \
    "$TRANSFORMER_DIR/saved_models/"; then
  CKPT_OK=0
  PULL_BACK_OK=0
fi
LOGS_OK=1
if ! rsync_with_retry "sweep_logs/" \
    "root@$POD_HOST:/workspace/cgan/transformer_neurIPS/sweep_logs/" \
    "$TRANSFORMER_DIR/sweep_logs/"; then
  LOGS_OK=0
  PULL_BACK_OK=0
fi

echo ""
echo "------------------------------------------------------------------"
echo " Pull-back summary"
echo "------------------------------------------------------------------"
echo "  checkpoints: $([[ "$CKPT_OK" == "1" ]] && echo OK || echo "FAILED -- see above")"
find "$TRANSFORMER_DIR/saved_models" -maxdepth 1 -name "r2_${ARM}_*" -newermt "-15 min" \
  -exec ls -la {} \; 2>/dev/null | sed 's/^/    /'
echo "  sweep_logs:  $([[ "$LOGS_OK" == "1" ]] && echo OK || echo "FAILED -- see above")"
echo ""
if [[ "$PULL_BACK_OK" == "1" ]]; then
  echo "Done. Results under saved_models/ and sweep_logs/ (local)."
  echo "Pod will now be terminated (trap on EXIT) unless KEEP_POD=1 was set."
else
  echo "AT LEAST ONE PULL-BACK STEP FAILED -- see cleanup_pod()'s message" >&2
  echo "below for how to retrieve manually before this pod is terminated." >&2
fi
