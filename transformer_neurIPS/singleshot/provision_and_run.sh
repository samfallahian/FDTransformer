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
#   bash singleshot/provision_and_run.sh --wandb=abc123yourkeyhere
#   bash singleshot/provision_and_run.sh --wandb=abc123yourkeyhere --round=3 --fresh
#
#   MULTIPLE EXPERIMENTS CONCURRENTLY on the SAME pod's GPU (OVERVIEW.md
#   section 32.3 -- one job alone showed only 24-47% GPU utilization and
#   ~34GB/144GB VRAM on an H200, real headroom for more): pass --arm2/
#   --round2 (and optionally --arm3/--round3) to launch 2nd/3rd training
#   processes on the same pod, at the same time, each over its own SSH
#   connection. This is PLAIN CO-LOCATION (bare `python` processes
#   sharing the GPU via default CUDA time-slicing), NOT NVIDIA MPS
#   spatial co-residency -- section 32.3 explicitly recommends measuring
#   this cheaper option first before setting up MPS, and each additional
#   concurrent job slices GPU compute further (3 jobs will each run
#   noticeably slower per-step than 1 alone -- this buys more EXPERIMENTS
#   per pod-hour, not more STEPS per experiment). Each process's own
#   output is prefixed ([arm/rN], [arm2/rN2], [arm3/rN3]) so interleaved
#   output stays distinguishable:
#     bash singleshot/provision_and_run.sh --wandb=KEY \
#       --arm=s8_h9_moreseqs_scaled --round=2 --max-steps=45000 --max-hours=0.5 \
#       --arm2=h11_ridge_distill --round2=2 --max-steps2=10000 --max-hours2=0.5 \
#       --arm3=s7_h9_scaled --round3=4 --fresh3 --max-steps3=20000 --max-hours3=0.5
#   Every active slot pulls back independently at the end (saved_models/
#   r${ROUND}_${ARM}_*, r${ROUND2}_${ARM2}_*, r${ROUND3}_${ARM3}_*) -- a
#   failure in one slot's pull-back does not skip the others'.
#
#   IMPORTANT -- MAX_STEPS is an ABSOLUTE target, not "how many more
#   steps": changing it between resumes of the SAME run re-triggers the
#   LR warmup/cosine schedule (make_lr_lambda computes its curve from
#   the CURRENT invocation's MAX_STEPS, not the checkpoint's original
#   target) and the AR-frames curriculum, both computed as fractions of
#   MAX_STEPS -- confirmed the hard way: resuming h11_ridge_distill's
#   round 2 with a bigger MAX_STEPS than its original run re-inflated
#   its LR back up from an already-annealed floor, roughly doubling
#   train_loss right after resume. Pick each arm's real target step
#   count ONCE and pass that same value on every future resume of that
#   run -- never bump MAX_STEPS mid-series.
#
# CLI FLAGS (all optional -- equivalent env vars also still work, a flag
# here just sets that same var for you; see ENV VARS below for details
# on what each one actually does)
#   --wandb=KEY   sets WANDB_API_KEY. Passed as a plain, UNQUOTED value
#                 on the command line (--wandb=abc123, not --wandb="abc123")
#                 -- note this DOES land in this shell's history and in
#                 `ps` output for the moment this script's own arg-parsing
#                 loop runs (unlike the RunPod API key -- see the
#                 PREREQUISITE section above, which deliberately never
#                 accepts that one as a script argument for exactly this
#                 reason). Accepted here anyway because it's what was
#                 asked for and a wandb key is lower-stakes than a cloud
#                 billing/infra credential; use WANDB_API_KEY=... as an
#                 env var instead if you'd rather avoid that exposure.
#   --wandb-group=NAME  sets WANDB_GROUP, applied to EVERY active slot via
#                 the trainer's existing `--set WANDB_GROUP=NAME` Config-
#                 override mechanism -- clusters this batch's runs
#                 together in the wandb UI under one shared, filterable
#                 label. Unset (default) means no group, unchanged
#                 behavior.
#   --round=N         sets ROUND (default 2 -- see ROUND below)
#   --arm=NAME        sets ARM (default h11_ridge_distill -- see ARM below)
#   --fresh           cold-start control run: adds --fresh --no-warm-start
#                     to the trainer invocation (see ROUND below for when
#                     to use this -- e.g. h11_ridge_distill's "exp-b")
#   --max-steps=N     sets MAX_STEPS (default 2500 -- see MAX_STEPS below)
#   --max-hours=H     sets MAX_HOURS (default 0.5 -- see MAX_HOURS below)
#   --extra="..."     raw string appended verbatim to THIS slot's python
#                     invocation -- e.g. --extra="--set WEIGHT_DECAY=0.05
#                     --set DROPOUT=0.05" for an ad-hoc regularization
#                     experiment, using the trainer's own generic --set
#                     KEY=VALUE Config-override mechanism, without needing
#                     a dedicated flag here for every possible field.
#                     Quote the whole value as ONE shell argument.
#   --arm2=NAME       sets ARM2 -- if given, launches a SECOND training
#                     process concurrently on the same pod (see
#                     "MULTIPLE EXPERIMENTS CONCURRENTLY" above). Unset
#                     (default) means single-arm mode, unchanged from before.
#   --round2=N        sets ROUND2 (default 2 if ARM2 is set)
#   --fresh2          cold-start for the SECOND slot
#   --max-steps2=N    sets MAX_STEPS2 (default 2500 if ARM2 is set)
#   --max-hours2=H    sets MAX_HOURS2 (default 0.5 if ARM2 is set)
#   --extra2="..."    same as --extra, for the SECOND slot
#   --arm3=NAME       sets ARM3 -- if given, launches a THIRD concurrent
#                     training process (works with or without ARM2 set).
#   --round3=N        sets ROUND3 (default 2 if ARM3 is set)
#   --fresh3          cold-start for the THIRD slot
#   --max-steps3=N    sets MAX_STEPS3 (default 2500 if ARM3 is set)
#   --max-hours3=H    sets MAX_HOURS3 (default 0.5 if ARM3 is set)
#   --extra3="..."    same as --extra, for the THIRD slot
#
# ENV VARS (all optional)
#   GPU_ID             exact GPU type string from `runpodctl gpu list`
#                       (default: auto-picks the first available match
#                       for GPU_AUTO_DETECT_PATTERN below; override if
#                       nothing in that pool is in stock)
#   GPU_AUTO_DETECT_PATTERN  case-insensitive regex against gpu list's
#                       id/gpuId/displayName/name, used by the auto-pick
#                       above (default "H200|H300|RTX PRO 6000 Blackwell
#                       (Server|Workstation) Edition$" -- the trailing
#                       `Edition$` anchor is load-bearing, it excludes
#                       that card's much-smaller MIG-sliced variants,
#                       24GB/48GB, which would NOT fit this workload).
#                       Widen or narrow the auto-detect pool by
#                       overriding this instead of editing the script.
#   GPU_WAIT_SECS       when auto-picking (GPU_ID unset), how long to
#                       keep polling for matching stock before giving up
#                       (default 300 = 5min). Real stock -- ESPECIALLY
#                       when pinned to one datacenter via
#                       NETWORK_VOLUME_ID/DATA_CENTER_IDS -- can be
#                       genuinely zero for a few minutes and then free
#                       up; only ever applies to the auto-detect path,
#                       an explicit GPU_ID never waits.
#   GPU_POLL_INTERVAL_SECS  how often to recheck while waiting above
#                       (default 10)
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
#   ROUND               --round value passed to the trainer (default 2,
#                       i.e. h11_ridge_distill's warm-started "exp-a" --
#                       see singleshot/README.md's exp-a/exp-b table).
#                       Also controls which checkpoint prefix this script
#                       pulls back and reports on (saved_models/r${ROUND}_
#                       ${ARM}_*) -- round 2's own checkpoint completed
#                       successfully (2500/2500 steps, +30-37% over
#                       persistence, confirmed under the corrected
#                       DELTA_ANCHOR='persistence' config from OVERVIEW.md
#                       v6.5). The remaining undone half of that arm's own
#                       two-experiment design is round 3, the cold-start
#                       control -- pass --round=3 --fresh for that.
#   FRESH               set (or pass --fresh) for a cold-start control run
#                       -- adds --fresh --no-warm-start to the trainer
#                       invocation instead of warm-starting from the
#                       existing round's checkpoint. Use with ROUND=3 to
#                       run h11_ridge_distill's "exp-b", per README.
#   MAX_HOURS           wall-clock cap passed to the trainer (default 0.5 = 30 min)
#   MAX_STEPS           optimizer-step cap (default 2500 -- deliberately
#                       sized BELOW a conservative throughput estimate so
#                       WARMUP_FRAC doesn't break, same lesson as
#                       OVERVIEW.md v5.0 section 28.6). NOTE: this is an
#                       ABSOLUTE global step target, not "how many more
#                       steps to run" -- `train_production_transformer_
#                       deep_dive.py`'s loop is `while step < MAX_STEPS`,
#                       so resuming a checkpoint already at step 4000
#                       with MAX_STEPS=2500 is a silent no-op (the loop
#                       exits immediately). To continue a run further,
#                       pass a MAX_STEPS bigger than its last checkpoint's
#                       own step count.
#   ARM2/ROUND2/FRESH2/MAX_STEPS2/MAX_HOURS2
#   ARM3/ROUND3/FRESH3/MAX_STEPS3/MAX_HOURS3
#                       same meaning as ARM/ROUND/FRESH/MAX_STEPS/
#                       MAX_HOURS, for OPTIONAL 2nd/3rd training
#                       processes launched concurrently on the same pod
#                       -- see "MULTIPLE EXPERIMENTS CONCURRENTLY" above.
#                       Only ARM2/ARM3 need to be set to enable each
#                       slot; that slot's own ROUND defaults to 2,
#                       MAX_STEPS to 2500, MAX_HOURS to 0.5 if left unset.
#   WANDB_API_KEY       if set (or passed as --wandb=KEY, see CLI FLAGS
#                       above), wandb logs in non-interactively for this
#                       run (`WANDB_API_KEY=... wandb login` supports this
#                       with no prompt); if unset, the run uses --no-wandb
#                       -- an unattended pipeline cannot satisfy wandb's
#                       normal interactive/browser login
#   POD_ID              reuse an already-running pod instead of creating one
#   KEEP_POD            set to 1 to skip the final `pod delete` (for
#                       debugging a run without losing the box)
#   NETWORK_VOLUME_ID   attach a persistent RunPod network volume instead
#                       of re-uploading train_80.h5/val_80.h5 every run
#                       (~14GB, minutes each time otherwise). A network
#                       volume is pinned to ONE datacenter -- setting
#                       this looks that datacenter up automatically and
#                       pins BOTH pod creation and H200/H300 GPU
#                       selection to it (stock elsewhere doesn't help;
#                       the pod can't be created anywhere else and still
#                       attach the volume), and replaces the whole
#                       "Phase 2/2: data files" step with a quick remote
#                       size check against what's already on the volume
#                       -- training refuses to launch if that check
#                       fails, rather than silently training on stale or
#                       missing data. Unset (default) preserves the
#                       original re-upload-every-run behavior exactly,
#                       with no datacenter pinning. One-time setup to
#                       populate a fresh volume: create/attach a pod
#                       with this same --network-volume-id (e.g. via
#                       this script's own pod-creation step, or by hand)
#                       and run scp_data_files.sh against it once.
#   DATA_CENTER_IDS     override/skip NETWORK_VOLUME_ID's automatic
#                       datacenter lookup -- set this directly if you
#                       already know it, or to pin a datacenter for a
#                       run that isn't using a network volume at all.
#   FORCE_TERMINATE_ON_PULL_FAILURE
#                       DEFAULTS TO 1 -- the pod is terminated on exit
#                       regardless of whether the artifact pull-back
#                       (checkpoints, sweep_logs) succeeded. This was
#                       flipped from an earlier default of 0 (which left
#                       a failed-pull-back pod running, billing, for a
#                       human to clean up by hand) after exactly that
#                       happened on a real run and the cost/inconvenience
#                       of a dangling pod was judged worse than the risk
#                       of losing a run's local artifacts -- wandb's own
#                       versioned Artifact upload (every save_checkpoint()
#                       call) is the fallback recovery path in that case.
#                       Set this to 0 explicitly if you want the OLD
#                       protective behavior back for a specific run (e.g.
#                       a long/expensive run where you'd rather babysit a
#                       failed pull-back than risk losing the result).
#   SSH_KEY             local private key to use (default ~/.ssh/id_ed25519
#                       -- must match a key added via `runpodctl ssh add-key`
#                       or already present on the pod's image)
#   LANES               concurrent scp streams per data file (default 10,
#                       see scp_data_files.sh's header comment -- croc
#                       was tried and dropped from this pipeline
#                       entirely, see singleshot/CROC_JOURNEY.md for
#                       that investigation and bench_scp_variants.sh for
#                       why plain parallel scp beat it).
#   EARLY_STOP_PATIENCE_STEPS  passed to every slot as
#                       --early-stop-patience-steps (default 500,
#                       OVERVIEW.md §42's own stated "going forward"
#                       value). Set to "" to disable and always run the
#                       full step budget. Confirmed the hard way this
#                       needed a default, not opt-in: a real run peaked
#                       at step 500/2500, never re-promoted, and burned
#                       the remaining ~2/3 of its wall-clock on a
#                       strictly worse checkpoint because nothing passed
#                       this flag (OVERVIEW.md §46).
#   GLOBAL_TIMEOUT_SECS  hard wall-clock cap on this ENTIRE script, no
#                       matter what it's stuck on -- see "GLOBAL TIMEOUT
#                       WRAPPER" near the top of this file for the
#                       mechanism (same two-layer pattern as
#                       bench_croc_variants.sh). DERIVED by default, not
#                       a fixed number: the longest MAX_HOURS across all
#                       active slots (they run concurrently) plus
#                       SETUP_OVERHEAD_SECS. Set this directly to bypass
#                       that derivation entirely.
#   SETUP_OVERHEAD_SECS  seconds of margin added on top of the longest
#                       slot's MAX_HOURS when DERIVING GLOBAL_TIMEOUT_SECS
#                       (default 900 = 15 min, for pod boot + bootstrap +
#                       data transfer -- "30 minutes of GPU compute
#                       should be over in 45 minutes of wall-clock time,
#                       no matter what"). Has no effect if
#                       GLOBAL_TIMEOUT_SECS is set directly.
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

# ---------------------------------------------------------------- #
# GLOBAL TIMEOUT WRAPPER -- same two-layer pattern as bench_croc_
# variants.sh (see that file's own header for the full rationale): a
# stuck pod is real money, and MAX_HOURS only self-polices INSIDE each
# training process's own loop -- it does nothing if bootstrap hangs, an
# SSH connection stalls, or a process gets stuck somewhere that never
# reaches that check. Confirmed directly: "30 minutes of GPU compute
# should be over in 45 minutes of wall-clock time, no matter what."
#
#   Layer 1: this process re-execs ITSELF as a backgrounded child, then
#   a watchdog subshell sleeps GLOBAL_TIMEOUT_SECS and, if the child is
#   still alive, sends SIGTERM (bash still runs cleanup_pod()'s EXIT
#   trap on SIGTERM -- not a bypass), waits 10s grace, then SIGKILL.
#
#   Layer 2 (armed later, once POD_ID is known -- see "host=$POD_HOST"
#   below): a fully DETACHED background job that calls `runpodctl pod
#   delete` at a fixed deadline slightly AFTER layer 1's own --
#   independent of this script's signal handling or control flow
#   entirely, in case layer 1's SIGTERM->cleanup_pod->pod-delete path
#   itself somehow hangs (a stuck `ssh` inside cleanup_pod, say).
#   IMPORTANT: cleanup_pod() explicitly CANCELS this timer (kills its
#   pid) whenever it decides to deliberately KEEP the pod alive (the
#   pull-back-failure branch) -- confirmed the hard way that without
#   this cancellation, a purely wall-clock-based Layer 2 doesn't know
#   about that decision and silently overrides it once its deadline
#   passes, force-deleting a pod that was intentionally being preserved
#   for manual recovery (and killing whatever was still training on it).
#
# GLOBAL_TIMEOUT_SECS is DERIVED by default, not a fixed number: the
# longest MAX_HOURS across all active slots (they run concurrently, so
# the longest one determines when the pipeline should naturally finish)
# plus SETUP_OVERHEAD_SECS (pod boot + bootstrap + data transfer). This
# needs each active slot's --max-hoursN BEFORE the real argument parser
# (below) has a lot more context to work with -- rather than duplicate
# that whole parser, this is a small, SEPARATE, one-purpose pre-scan of
# "$@" for just the --max-hours*/--armN flags. The real parser re-parses
# the identical "$@" a second time inside the re-exec'd child --
# pure/deterministic, so re-parsing twice is safe and far simpler than
# exporting bash arrays across a re-exec (which doesn't work cleanly
# anyway -- arrays aren't exportable the normal way).
#
# The re-exec only happens ONCE: _PROV_INNER marks "this is the real
# work", so the child doesn't re-wrap itself again.
# ---------------------------------------------------------------- #
SETUP_OVERHEAD_SECS="${SETUP_OVERHEAD_SECS:-900}"
if [[ -z "${GLOBAL_TIMEOUT_SECS:-}" ]]; then
  _peek_hours=0.5; _peek_hours2=""; _peek_hours3=""; _peek_arm2=""; _peek_arm3=""
  for _a in "$@"; do
    case "$_a" in
      --max-hours=*)  _peek_hours="${_a#--max-hours=}" ;;
      --max-hours2=*) _peek_hours2="${_a#--max-hours2=}" ;;
      --max-hours3=*) _peek_hours3="${_a#--max-hours3=}" ;;
      --arm2=*)       _peek_arm2="${_a#--arm2=}" ;;
      --arm3=*)       _peek_arm3="${_a#--arm3=}" ;;
    esac
  done
  [[ -n "$_peek_arm2" && -z "$_peek_hours2" ]] && _peek_hours2=0.5
  [[ -n "$_peek_arm3" && -z "$_peek_hours3" ]] && _peek_hours3=0.5
  _peek_max_hours="$(awk -v a="$_peek_hours" -v b="${_peek_hours2:-0}" -v c="${_peek_hours3:-0}" \
    'BEGIN{m=a; if(b>m)m=b; if(c>m)m=c; print m}')"
  GLOBAL_TIMEOUT_SECS="$(awk -v h="$_peek_max_hours" -v o="$SETUP_OVERHEAD_SECS" \
    'BEGIN{printf "%d", h*3600 + o}')"
fi
if [[ -z "${_PROV_INNER:-}" ]]; then
  export _PROV_INNER=1
  echo "=================================================================="
  echo " Global wall-clock budget: ${GLOBAL_TIMEOUT_SECS}s (derived: longest"
  echo " active slot's MAX_HOURS + ${SETUP_OVERHEAD_SECS}s setup overhead --"
  echo " override with GLOBAL_TIMEOUT_SECS=<secs> to bypass this derivation)"
  echo "=================================================================="
  "$0" "$@" &
  _inner_pid=$!
  (
    sleep "$GLOBAL_TIMEOUT_SECS"
    if kill -0 "$_inner_pid" 2>/dev/null; then
      echo "" >&2
      echo "[FAIL] GLOBAL ${GLOBAL_TIMEOUT_SECS}s TIMEOUT HIT -- sending SIGTERM (10s grace for pod cleanup via the EXIT trap, then SIGKILL)." >&2
      kill -TERM "$_inner_pid" 2>/dev/null || true
      sleep 10
      kill -KILL "$_inner_pid" 2>/dev/null || true
    fi
  ) &
  _watchdog_pid=$!
  _rc=0
  wait "$_inner_pid" 2>/dev/null || _rc=$?
  kill "$_watchdog_pid" 2>/dev/null || true
  wait "$_watchdog_pid" 2>/dev/null || true
  exit "$_rc"
fi
SCRIPT_START_TS=$(date +%s)

FRESH="${FRESH:-0}"
FRESH2="${FRESH2:-0}"
FRESH3="${FRESH3:-0}"
for _arg in "$@"; do
  case "$_arg" in
    --wandb=*)       WANDB_API_KEY="${_arg#--wandb=}" ;;
    --wandb-group=*) WANDB_GROUP="${_arg#--wandb-group=}" ;;
    --round=*)     ROUND="${_arg#--round=}" ;;
    --arm=*)       ARM="${_arg#--arm=}" ;;
    --fresh)       FRESH=1 ;;
    --max-steps=*) MAX_STEPS="${_arg#--max-steps=}" ;;
    --max-hours=*) MAX_HOURS="${_arg#--max-hours=}" ;;
    --extra=*)      EXTRA="${_arg#--extra=}" ;;
    --arm2=*)       ARM2="${_arg#--arm2=}" ;;
    --round2=*)     ROUND2="${_arg#--round2=}" ;;
    --fresh2)       FRESH2=1 ;;
    --max-steps2=*) MAX_STEPS2="${_arg#--max-steps2=}" ;;
    --max-hours2=*) MAX_HOURS2="${_arg#--max-hours2=}" ;;
    --extra2=*)     EXTRA2="${_arg#--extra2=}" ;;
    --arm3=*)       ARM3="${_arg#--arm3=}" ;;
    --round3=*)     ROUND3="${_arg#--round3=}" ;;
    --fresh3)       FRESH3=1 ;;
    --max-steps3=*) MAX_STEPS3="${_arg#--max-steps3=}" ;;
    --max-hours3=*) MAX_HOURS3="${_arg#--max-hours3=}" ;;
    --extra3=*)     EXTRA3="${_arg#--extra3=}" ;;
    *)
      echo "Unknown argument: $_arg" >&2
      echo "Supported: --wandb=KEY --wandb-group=NAME --round=N --arm=NAME --fresh --max-steps=N --max-hours=H --extra=\"...\"" >&2
      echo "           --arm2=NAME --round2=N --fresh2 --max-steps2=N --max-hours2=H --extra2=\"...\"" >&2
      echo "           --arm3=NAME --round3=N --fresh3 --max-steps3=N --max-hours3=H --extra3=\"...\"" >&2
      exit 1
      ;;
  esac
done

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
ROUND="${ROUND:-2}"
MAX_HOURS="${MAX_HOURS:-0.5}"
MAX_STEPS="${MAX_STEPS:-2500}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
KEEP_POD="${KEEP_POD:-0}"
LANES="${LANES:-10}"
WANDB_GROUP="${WANDB_GROUP:-}"
# Defaults to 500 (OVERVIEW.md §42's own stated "going forward" value,
# matching --val-every 500 below -- stops on the very next non-promoting
# eval after a promotion, one eval of slack) -- confirmed the hard way
# this was worth defaulting rather than leaving opt-in-only: a real
# m3_ar_combined run (OVERVIEW.md §46) peaked at step 500 of a 2500-step
# budget, never re-promoted, and ran the full remaining ~1500 steps (2/3
# of its wall-clock) for a strictly worse checkpoint, purely because
# neither its launch command nor this script ever passed
# --early-stop-patience-steps. Set to empty ("") to disable and run the
# full step budget regardless, matching the pre-this-default behavior.
EARLY_STOP_PATIENCE_STEPS="${EARLY_STOP_PATIENCE_STEPS-500}"
# NETWORK_VOLUME_ID (optional): attach a persistent RunPod network volume
# instead of re-uploading train_80.h5/val_80.h5 every run. A network
# volume is pinned to ONE datacenter -- setting this looks that
# datacenter up automatically (below) and pins pod creation + GPU
# selection to it, and skips scp_data_files.sh entirely in favor of a
# quick remote existence/size check against what's already on the
# volume. Unset (default) preserves the original re-upload-every-run
# behavior exactly. DATA_CENTER_IDS can be set directly instead (or in
# addition, to override the auto-lookup) if you already know it.
NETWORK_VOLUME_ID="${NETWORK_VOLUME_ID:-}"
DATA_CENTER_IDS="${DATA_CENTER_IDS:-}"

# Slots 2 and 3 are entirely optional, only active if ARM2/ARM3 end up
# non-empty (set via --arm2=/--arm3= or ARM2=/ARM3=). SLOT_* arrays
# below are what every later stage (launch, wait, pull-back, summary)
# actually loops over -- slot 1 (ARM/ROUND/...) is always index 0, so
# there's exactly ONE code path for "1, 2, or 3 concurrent experiments"
# instead of hand-duplicating the same logic per slot (the same
# duplication lesson this repo already learned the hard way with croc,
# see lib_croc.sh's own header).
EXTRA="${EXTRA:-}"
ARM2="${ARM2:-}"
if [[ -n "$ARM2" ]]; then
  ROUND2="${ROUND2:-2}"
  MAX_STEPS2="${MAX_STEPS2:-2500}"
  MAX_HOURS2="${MAX_HOURS2:-0.5}"
fi
EXTRA2="${EXTRA2:-}"
ARM3="${ARM3:-}"
if [[ -n "$ARM3" ]]; then
  ROUND3="${ROUND3:-2}"
  MAX_STEPS3="${MAX_STEPS3:-2500}"
  MAX_HOURS3="${MAX_HOURS3:-0.5}"
fi
EXTRA3="${EXTRA3:-}"

SLOT_ARMS=("$ARM"); SLOT_ROUNDS=("$ROUND")
SLOT_STEPS=("$MAX_STEPS"); SLOT_HOURS=("$MAX_HOURS"); SLOT_FRESH=("$FRESH"); SLOT_EXTRA=("$EXTRA")
if [[ -n "$ARM2" ]]; then
  SLOT_ARMS+=("$ARM2"); SLOT_ROUNDS+=("$ROUND2")
  SLOT_STEPS+=("$MAX_STEPS2"); SLOT_HOURS+=("$MAX_HOURS2"); SLOT_FRESH+=("$FRESH2"); SLOT_EXTRA+=("$EXTRA2")
fi
if [[ -n "$ARM3" ]]; then
  SLOT_ARMS+=("$ARM3"); SLOT_ROUNDS+=("$ROUND3")
  SLOT_STEPS+=("$MAX_STEPS3"); SLOT_HOURS+=("$MAX_HOURS3"); SLOT_FRESH+=("$FRESH3"); SLOT_EXTRA+=("$EXTRA3")
fi

if [[ -n "$NETWORK_VOLUME_ID" && -z "$DATA_CENTER_IDS" && -z "${POD_ID:-}" ]]; then
  echo "NETWORK_VOLUME_ID=$NETWORK_VOLUME_ID set -- looking up its datacenter"
  echo "(a network volume is pinned to one datacenter; the pod has to be"
  echo "created there too, or the volume can't be attached)..."
  VOL_JSON="$("$RUNPODCTL" network-volume get "$NETWORK_VOLUME_ID" -o json 2>&1)"
  DATA_CENTER_IDS="$(echo "$VOL_JSON" | jq -r '.dataCenterId // empty' 2>/dev/null)"
  if [[ -z "$DATA_CENTER_IDS" ]]; then
    echo "" >&2
    echo "Could not look up NETWORK_VOLUME_ID=$NETWORK_VOLUME_ID's datacenter --" >&2
    echo "check it exists ('runpodctl network-volume get $NETWORK_VOLUME_ID')," >&2
    echo "or set DATA_CENTER_IDS explicitly if this keeps failing:" >&2
    echo "$VOL_JSON" >&2
    exit 1
  fi
  echo "  volume is in datacenter $DATA_CENTER_IDS -- pod will be pinned there"
  echo ""
fi

GPU_WAIT_SECS="${GPU_WAIT_SECS:-300}"            # 5 min default
GPU_POLL_INTERVAL_SECS="${GPU_POLL_INTERVAL_SECS:-10}"
# Case-insensitive regex against gpu list's id/gpuId/displayName/name --
# override to widen/narrow the auto-detect pool without touching this
# script. Default includes the RTX PRO 6000 Blackwell 96GB cards
# (Server AND Workstation edition -- both real 96GB parts, $2.09-2.19/hr,
# CHEAPER than H100 at $3.49/hr) alongside H200/H300: AR-rollout-loss
# memory scales with an arm's AR_SEQS, not micro_batch, so h9_ar_freq1/
# h11_ridge_distill/m1_ar_weight/m2_ar_curriculum (all AR_SEQS=2, ~34GB
# observed on H200) fit comfortably in 96GB. The trailing `Edition$`
# anchor is load-bearing -- WITHOUT it this would also match the MIG-
# sliced variants ("... Edition MIG 1g.24gb" / "... MIG 2g.48gb", 24GB/
# 48GB respectively), which are real, much smaller partitions of the
# same physical card and would NOT fit this workload.
# NOTE: m3_ar_combined (AR_SEQS=16) is estimated at ~150-200GB actual --
# it does NOT fit on these 96GB cards, and likely not even on a 141GB
# H200 either. That's a pre-existing sizing question independent of
# this GPU pool, not something widening the pool here fixes.
GPU_AUTO_DETECT_PATTERN="${GPU_AUTO_DETECT_PATTERN:-H200|H300|RTX PRO 6000 Blackwell (Server|Workstation) Edition\$}"
if [[ -z "$GPU_ID" && -z "${POD_ID:-}" ]]; then
  if [[ -n "$DATA_CENTER_IDS" ]]; then
    echo "Looking up an available GPU ($GPU_AUTO_DETECT_PATTERN) in datacenter"
    echo "$DATA_CENTER_IDS (pinned there by NETWORK_VOLUME_ID/DATA_CENTER_IDS;"
    echo "override with GPU_ID=... or widen/narrow via GPU_AUTO_DETECT_PATTERN=...)."
  else
    echo "Looking up an available GPU ($GPU_AUTO_DETECT_PATTERN)."
    echo "Override with GPU_ID=... or widen/narrow via GPU_AUTO_DETECT_PATTERN=...)."
  fi
  # Stock -- ESPECIALLY pinned to ONE datacenter via a network volume --
  # can be genuinely zero for a while and then free up minutes later --
  # confirmed hitting exactly this against US-GA-2 in practice. Poll
  # instead of failing on the very first empty check: every
  # GPU_POLL_INTERVAL_SECS (default 10s) for up to GPU_WAIT_SECS
  # (default 300s = 5min) before giving up for real. A GPU_ID explicitly
  # set by the caller never reaches this block at all (the outer `if`
  # above), so this retry only ever applies to the auto-detect path.
  echo "Will retry every ${GPU_POLL_INTERVAL_SECS}s for up to ${GPU_WAIT_SECS}s if none is available yet."
  SECONDS=0
  while true; do
    GPU_JSON="$("$RUNPODCTL" gpu list -o json 2>&1)"
    # $dc empty (the common, non-network-volume case) matches every GPU
    # regardless of per-datacenter stock, exactly like the original
    # global search -- $dc non-empty additionally requires THIS specific
    # datacenter to actually carry real stock ("none" means zero, an
    # empty stockStatus string appears to mean "not currently tracked",
    # NOT "unlimited" -- confirmed empirically against US-GA-2's own
    # listing, so only "none" is excluded here, not blank).
    GPU_ID="$(echo "$GPU_JSON" | jq -r --arg dc "$DATA_CENTER_IDS" --arg pat "$GPU_AUTO_DETECT_PATTERN" '
      [.[]? // .gpus[]? |
       select((.id // .gpuId // .displayName // .name // "") | test($pat; "i")) |
       select($dc == "" or ((.dataCenterAvailability // []) | any(.dataCenterId == $dc and .stockStatus != "none"))) |
       (.id // .gpuId // .displayName // .name)
      ] | .[0] // empty' 2>/dev/null || true)"
    if [[ -n "$GPU_ID" ]]; then
      break
    fi
    if [[ "$SECONDS" -ge "$GPU_WAIT_SECS" ]]; then
      break
    fi
    echo "  no stock yet (${SECONDS}s/${GPU_WAIT_SECS}s elapsed) -- retrying in ${GPU_POLL_INTERVAL_SECS}s..."
    sleep "$GPU_POLL_INTERVAL_SECS"
  done
  if [[ -z "$GPU_ID" ]]; then
    echo "" >&2
    if [[ -n "$DATA_CENTER_IDS" ]]; then
      echo "Still no matching GPU stock in $DATA_CENTER_IDS after ${GPU_WAIT_SECS}s" >&2
      echo "of polling -- a network volume pins the pod to this one datacenter, so" >&2
      echo "stock elsewhere doesn't help. Set GPU_ID explicitly to whatever IS in" >&2
      echo "stock there, or re-run (optionally with a longer GPU_WAIT_SECS) once" >&2
      echo "stock frees up -- see the raw list below:" >&2
    else
      echo "Still no matching GPU stock after ${GPU_WAIT_SECS}s of polling (pattern:" >&2
      echo "$GPU_AUTO_DETECT_PATTERN -- or the JSON field names didn't match what" >&2
      echo "this script guessed -- see the raw output below). Set GPU_ID explicitly" >&2
      echo "and re-run:" >&2
    fi
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
  # Plain scalar, spliced in UNQUOTED below (not a bash array) -- matches
  # this file's own existing convention for optional flags
  # ($fresh_flags/$group_flags in launch_training()). NOT an array:
  # macOS's stock bash (3.2.57) raises "unbound variable" under `set -u`
  # when expanding "${arr[@]}" on a genuinely EMPTY array (confirmed
  # directly, see select_fastest_pod.sh's own history of this exact
  # bug) -- an empty unquoted scalar just expands to zero words instead,
  # which is exactly "no flag" here. Safe to word-split unquoted:
  # NETWORK_VOLUME_ID/DATA_CENTER_IDS are plain IDs, never contain
  # spaces or glob characters.
  NETWORK_VOLUME_FLAG=""
  [[ -n "$NETWORK_VOLUME_ID" ]] && NETWORK_VOLUME_FLAG="--network-volume-id $NETWORK_VOLUME_ID"
  DATA_CENTER_FLAG=""
  [[ -n "$DATA_CENTER_IDS" ]] && DATA_CENTER_FLAG="--data-center-ids $DATA_CENTER_IDS"
  # 22/tcp only -- plain scp needs nothing beyond SSH itself. An earlier
  # version of this line requested a croc relay's port range directly;
  # confirmed across 5 real pod rentals that RunPod's Secure Cloud pods
  # don't support arbitrary custom TCP port exposure at all (see
  # singleshot/CROC_JOURNEY.md) -- one more reason croc was dropped
  # from this pipeline entirely in favor of parallel scp.
  echo "This blocks until SSH actually answers (--wait), up to 10 minutes."
  POD_JSON="$("$RUNPODCTL" pod create \
    --gpu-id "$GPU_ID" \
    "${IMAGE_FLAGS[@]}" \
    $NETWORK_VOLUME_FLAG \
    $DATA_CENTER_FLAG \
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
                # (see the pull-back step below). FORCE_TERMINATE_ON_PULL_
                # FAILURE defaults to 1 (see its header doc above) --
                # cleanup_pod() below terminates the pod regardless, unless
                # that's explicitly set to 0 for this run.

cleanup_pod() {
  if [[ "$KEEP_POD" == "1" ]]; then
    echo "KEEP_POD=1 set -- leaving pod $POD_ID running. Terminate it"
    echo "yourself when done: runpodctl pod delete $POD_ID"
    return
  fi
  if [[ "$PULL_BACK_OK" != "1" && "${FORCE_TERMINATE_ON_PULL_FAILURE:-1}" != "1" ]]; then
    # CRITICAL: cancel Layer 2's blind timer (see "GLOBAL TIMEOUT WRAPPER"
    # near the top of this file) -- without this, a real incident: Layer 2
    # doesn't know this branch just decided to KEEP the pod for manual
    # recovery, and its own deadline firing later force-deletes it anyway,
    # silently overriding this exact protection and killing whatever
    # training was still running mid-flight. Confirmed happening for real:
    # three runs showed up in wandb as "crashed" (abrupt kill, not a
    # graceful stop) right around when Layer 2's timer would have fired.
    kill "${SAFETY_NET_PID:-}" 2>/dev/null || true
    echo ""
    echo "==================================================================" >&2
    echo " NOT terminating pod $POD_ID -- the artifact pull-back below" >&2
    echo " reported a genuine failure (not just \"no files matched\"), so" >&2
    echo " whatever this run produced may still only exist on the pod." >&2
    echo " Retrieve it yourself, THEN terminate:" >&2
    echo "" >&2
    echo "   POD_HOST=$POD_HOST POD_PORT=$POD_PORT SSH_KEY=$SSH_KEY \\" >&2
    for _i in "${!SLOT_ARMS[@]}"; do
      echo "     rsync -avP -e \"ssh -p \$POD_PORT -i \$SSH_KEY\" \\" >&2
      echo "     \"root@\$POD_HOST:/workspace/cgan/transformer_neurIPS/saved_models/r${SLOT_ROUNDS[$_i]}_${SLOT_ARMS[$_i]}_*\" \\" >&2
      echo "     \"$TRANSFORMER_DIR/saved_models/\"" >&2
    done
    echo "   runpodctl pod delete $POD_ID" >&2
    echo "" >&2
    echo " (You're seeing this because FORCE_TERMINATE_ON_PULL_FAILURE=0" >&2
    echo " was explicitly set for this run -- it defaults to 1, i.e." >&2
    echo " terminate regardless, since a dangling pod costs more than a" >&2
    echo " lost local artifact wandb likely still has a copy of.)" >&2
    echo "==================================================================" >&2
    echo "" >&2
    echo " (Layer 2's backup pod-deletion timer has been cancelled -- this" >&2
    echo " pod will NOT be auto-deleted later. It's now entirely on you to" >&2
    echo " retrieve and/or delete it -- also worth checking wandb: every" >&2
    echo " save_checkpoint() call uploads a versioned Artifact, so recent" >&2
    echo " checkpoints are very likely recoverable there even if this pod" >&2
    echo " is lost -- see 'wandb: queued artifact upload' lines in this" >&2
    echo " run's own console output for the exact artifact names.)" >&2
    return
  fi
  # Also cancel Layer 2 on the NORMAL success path -- tidiness (no
  # dangling disowned sleep outliving this script for no reason), not
  # correctness: the pod's already gone by the time this line runs, so a
  # late-firing Layer 2 would just be a harmless no-op delete-of-nothing
  # either way.
  kill "${SAFETY_NET_PID:-}" 2>/dev/null || true
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

# Layer 2 of the global timeout (see "GLOBAL TIMEOUT WRAPPER" near the
# top of this file): a fully detached backup that deletes THIS pod at a
# fixed deadline no matter what happens to the rest of this script --
# does not depend on cleanup_pod()'s trap firing, this shell's job
# control, or anything else in this process. Deadline is set slightly
# AFTER layer 1's own SIGTERM/SIGKILL deadline, so the normal cleanup
# path gets a real chance to run first; this only actually deletes
# anything if that path somehow failed. Skipped when KEEP_POD=1,
# matching cleanup_pod()'s own first check exactly -- this script
# doesn't distinguish reused vs. newly-created pods anywhere else
# either, so neither does this.
if [[ "$KEEP_POD" != "1" ]]; then
  SAFETY_NET_SECS=$(( GLOBAL_TIMEOUT_SECS + 30 - ($(date +%s) - SCRIPT_START_TS) ))
  [[ "$SAFETY_NET_SECS" -lt 1 ]] && SAFETY_NET_SECS=1
  ( sleep "$SAFETY_NET_SECS"; "$RUNPODCTL" pod delete "$POD_ID" >/dev/null 2>&1 ) &
  # SAFETY_NET_PID captured BEFORE disown (disown only removes it from
  # this shell's job table / SIGHUP-on-exit list -- the pid itself stays
  # valid for `kill` regardless) -- cleanup_pod() uses this to CANCEL
  # this timer whenever it decides to deliberately keep the pod alive
  # (pull-back failure). Without that cancellation, this blind timer
  # would silently override that exact decision later -- confirmed
  # happening for real, see cleanup_pod()'s own comment on this.
  SAFETY_NET_PID=$!
  disown 2>/dev/null || true
  echo "  backup pod-deletion safety net armed: $POD_ID will be deleted in ${SAFETY_NET_SECS}s regardless, unless cleanup already ran"
else
  echo "  backup pod-deletion safety net skipped (KEEP_POD=1)"
fi
echo ""

export POD_HOST POD_PORT SSH_KEY LANES
DATA_SCP_PID=""
if [[ -n "$NETWORK_VOLUME_ID" ]]; then
  echo "=================================================================="
  echo " Phase 2/2: data files -- SKIPPED (NETWORK_VOLUME_ID=$NETWORK_VOLUME_ID"
  echo " is attached). Verifying the expected files are already on the"
  echo " mounted volume instead of re-uploading ~14GB every run."
  echo "=================================================================="
  DATA_OK=1
  for f in train_80.h5 val_80.h5; do
    local_path="$REPO_ROOT/transformer_neurIPS/data/$f"
    local_size="$(stat -f%z "$local_path" 2>/dev/null || stat -c%s "$local_path" 2>/dev/null || echo "")"
    remote_size="$(ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
      "root@$POD_HOST" "stat -c%s '/workspace/cgan/transformer_neurIPS/data/$f' 2>/dev/null" 2>/dev/null || echo "")"
    if [[ -n "$remote_size" && -n "$local_size" && "$remote_size" == "$local_size" ]]; then
      echo "  [OK]      $f ($remote_size bytes, matches local)"
    else
      echo "  [MISSING] $f (local=${local_size:-<none>} remote=${remote_size:-<none>})" >&2
      DATA_OK=0
    fi
  done
  if [[ "$DATA_OK" != "1" ]]; then
    echo "" >&2
    echo "==================================================================" >&2
    echo " Expected data not found (or size mismatch) on volume" >&2
    echo " $NETWORK_VOLUME_ID -- training will NOT launch. Upload it once" >&2
    echo " (from a pod with this SAME --network-volume-id attached):" >&2
    echo "   POD_HOST=<host> POD_PORT=<port> SSH_KEY=<key> bash $HERE/scp_data_files.sh" >&2
    echo "==================================================================" >&2
    exit 1
  fi
  echo ""
else
  echo "=================================================================="
  echo " Phase 2/2: data files, IN THE BACKGROUND, starting IMMEDIATELY --"
  echo " it only needs SSH (creates its own remote directory), no"
  echo " dependency on phase 1 or bootstrap. Now that scp replaced croc,"
  echo " there's no relay/tunnel setup to wait on either."
  echo "=================================================================="
  bash "$HERE/scp_data_files.sh" &
  DATA_SCP_PID=$!
fi

echo ""
echo "=================================================================="
echo " Phase 1/2: env files (code, checkpoint, ridge map, AE decoder) --"
echo " runs now, concurrently with the data transfer above. This DOES"
echo " need to finish before bootstrap, below: bootstrap_remote.sh and"
echo " requirements.txt are themselves among the files phase 1 ships."
echo "=================================================================="
bash "$HERE/scp_env_files.sh"

echo ""
if [[ -n "$DATA_SCP_PID" ]]; then
  echo "Bootstrapping the venv while the data transfer above keeps running..."
else
  echo "Bootstrapping the venv (data already verified on the attached volume)..."
fi
ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "root@$POD_HOST" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/cgan/transformer_neurIPS
bash singleshot/bootstrap_remote.sh
REMOTE

if [[ -n "$DATA_SCP_PID" ]]; then
  echo ""
  echo "Waiting for the data transfer to finish (if it hasn't already)..."
  # NOT a bare `wait` -- under `set -euo pipefail`, a nonzero exit from
  # scp_data_files.sh (e.g. a lane's retries all exhausted -- see
  # lib_scp.sh's scp_lane_with_retry) would trip errexit RIGHT HERE and
  # silently kill the entire pipeline before training ever launches, with
  # no clearer diagnostic than whatever scp_data_files.sh itself printed.
  # The pod still gets cleanly terminated either way (the EXIT trap fires
  # regardless), so this isn't an orphaned-billing risk -- but a pod
  # rental producing zero training results with no loud, unambiguous
  # reason why is exactly the kind of failure worth naming explicitly.
  if ! wait "$DATA_SCP_PID"; then
    echo "" >&2
    echo "==================================================================" >&2
    echo " DATA TRANSFER FAILED -- see scp_data_files.sh's own output above" >&2
    echo " for which file/lane. Training will NOT launch. This pod will" >&2
    echo " still be terminated normally (nothing to pull back yet), but" >&2
    echo " this run produced no results -- re-run once the underlying" >&2
    echo " network/SSH issue is understood (a single transient failure" >&2
    echo " should have been absorbed by scp_lane_with_retry's 3 attempts;" >&2
    echo " if you're seeing this, all 3 attempts failed for at least one" >&2
    echo " lane, which is worth a closer look, not just a blind re-run)." >&2
    echo "==================================================================" >&2
    exit 1
  fi
fi

echo ""
echo "=================================================================="
if [[ "${#SLOT_ARMS[@]}" -gt 1 ]]; then
  echo " Launching ${#SLOT_ARMS[@]} concurrent runs, plain co-location on one GPU:"
  for _i in "${!SLOT_ARMS[@]}"; do
    echo "   ${SLOT_HOURS[$_i]}h arm=${SLOT_ARMS[$_i]} (r${SLOT_ROUNDS[$_i]})"
  done
else
  echo " Launching the ${MAX_HOURS}h-capped run (arm=$ARM)"
fi
echo "=================================================================="
WANDB_FLAG="--no-wandb"
WANDB_SETUP=": # no WANDB_API_KEY set -- running with --no-wandb"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_FLAG=""
  # `export` here, NOT `WANDB_API_KEY=... wandb login --relogin` (an
  # earlier version of this line) -- that form only sets the var for
  # the `wandb login` COMMAND ITSELF, not for anything after it in this
  # same script, including the `python` invocation below. Confirmed the
  # hard way: training failed with "wandb: ERROR ... No API key
  # configured" even though --wandb=KEY was passed, because `wandb
  # login`'s own non-interactive auth didn't persist anything the
  # later `python` process could see. `export` makes the var part of
  # this whole script's environment instead, which the wandb SDK reads
  # directly at `wandb.init()` time -- no separate `wandb login` step
  # needed at all for this non-interactive path.
  WANDB_SETUP="export WANDB_API_KEY='$WANDB_API_KEY'"
fi

# launch_training ARM ROUND MAX_STEPS MAX_HOURS FRESH OUTVAR
#
# Runs one training invocation over its own SSH connection,
# BACKGROUNDED (the caller gets the pid via $OUTVAR and must `wait` on
# it). Output is tagged "[arm/rN] " per line (via `sed`) so two
# concurrent invocations' interleaved output stays distinguishable.
#
# Exit-code note: `(...) | sed ...` on its own would make `wait` report
# sed's exit code, not ssh's (i.e. always 0, since sed itself basically
# never fails) -- `set -o pipefail` INSIDE the subshell makes the
# subshell's own exit status reflect ssh's real exit code instead
# (pipefail returns the rightmost non-zero status in the pipe, or 0 if
# everything succeeded), which is what $OUTVAR's pid actually needs to
# mean anything when `wait`ed on later.
#
# MANUAL FALLBACK -- keep this at your fingertips if this script ever
# dies/hangs mid-run again (see OVERVIEW.md v6.0 section 30.1 for the
# incident that made this whole package this cautious). SSH in yourself
# and run, verbatim (substitute ROUND/ARM/MAX_STEPS/MAX_HOURS and add
# --fresh --no-warm-start if that's what this invocation used -- also
# documented in singleshot/README.md section 5a):
#
#   cd /workspace/cgan/transformer_neurIPS
#   source /workspace/cgan/.venv/bin/activate
#   python train_production_transformer_deep_dive.py \
#     --arm h11_ridge_distill --round 2 \
#     --max-steps 2500 --max-hours 0.5 \
#     --val-every 500 --no-wandb
launch_training() {
  local arm="$1" round="$2" max_steps="$3" max_hours="$4" fresh="$5" outvar="$6" extra="${7:-}"
  local fresh_flags="" group_flags="" patience_flags="" tag="[${arm}/r${round}]"
  local num_slots="${#SLOT_ARMS[@]}"
  [[ "$fresh" == "1" ]] && fresh_flags="--fresh --no-warm-start"
  # Uses the trainer's own existing generic `--set KEY=VALUE` Config-
  # override mechanism (train_production_transformer_deep_dive.py's
  # `WANDB_GROUP` field already existed, just unwired to any CLI flag)
  # instead of adding a dedicated flag there -- one fewer thing to keep
  # in sync between the two scripts.
  [[ -n "$WANDB_GROUP" ]] && group_flags="--set WANDB_GROUP=$WANDB_GROUP"
  # Defaults to 500 (see EARLY_STOP_PATIENCE_STEPS above) -- appended
  # BEFORE $extra below, so a slot's own --extra="--early-stop-patience-
  # steps N" still wins (argparse takes the last occurrence of a flag).
  [[ -n "$EARLY_STOP_PATIENCE_STEPS" ]] && patience_flags="--early-stop-patience-steps $EARLY_STOP_PATIENCE_STEPS"
  # $extra is a raw, per-slot string appended verbatim to this slot's
  # python invocation -- e.g. "--set WEIGHT_DECAY=0.05 --set DROPOUT=0.05"
  # for an ad-hoc regularization experiment on ONE slot without needing
  # a dedicated CLI flag here for every possible Config field.
  (
    set -o pipefail
    # NOT `sed "s/^/$tag /"` -- $tag is "[arm/rN]", and that literal `/`
    # collides with sed's own `/`-delimited substitute syntax (confirmed
    # the hard way: BSD sed on macOS hard-errors "bad flag in substitute
    # command" and exits immediately without processing any input,
    # which breaks this pipe, which SIGPIPEs the local ssh client, which
    # tears down the SSH session -- likely killing the REMOTE training
    # process too, mid-run, with no screen/nohup on this channel to
    # protect it). `awk` does plain string concatenation instead, so it
    # has no delimiter to collide with regardless of what's in $tag.
    ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
      "root@$POD_HOST" bash -s <<REMOTE 2>&1 | awk -v tag="$tag" '{print tag, $0; fflush()}'
set -euo pipefail
cd /workspace/cgan/transformer_neurIPS
# bootstrap_remote.sh's default VENV_DIR is \$REPO_ROOT/.venv, i.e.
# /workspace/cgan/.venv -- one level up from transformer_neurIPS/.
source ../.venv/bin/activate
$WANDB_SETUP
# Force ANSI color even though this remote script's stdout is piped
# through ssh into the local awk tagger below, not a real tty --
# train_production_transformer_deep_dive.py's own _COLOR_ON check
# (sys.stdout.isatty()) is False in exactly that situation, so the
# per-epoch report's green/red Δ coloring was silently rendering as
# plain text on every real run until this was added. The escape codes
# survive the awk tagging (it only prepends "tag " to each line) and
# render fine both in a real local terminal and in wandb's own console-
# capture view (which supports ANSI), so this is safe to force
# unconditionally here.
export PFD_FORCE_COLOR=1
# PyTorch (and numpy/h5py's own BLAS backend) default to grabbing a CPU
# thread PER CORE for their intra-op thread pool, with no constraint --
# confirmed nothing in this trainer ever calls torch.set_num_threads()
# or sets OMP_NUM_THREADS/MKL_NUM_THREADS. Harmless with exactly one
# training process on the box; with $num_slots CONCURRENT processes
# (this pipeline's whole "multiple experiments per pod" point), every
# one of them independently tries to claim every core, causing real CPU
# thread thrashing on top of the already-measured GPU contention
# (OVERVIEW.md v6.9 section 39.1's 2.4x same-arm slowdown) -- for zero
# benefit, since the actual model compute is GPU-bound; CPU threads
# here only serve host-side data feeding/eval, which doesn't need a
# full core count. Cap each process to its fair share instead.
_NPROC="\$(nproc 2>/dev/null || echo 4)"
_THREADS_PER_SLOT=\$(( _NPROC / $num_slots ))
[[ "\$_THREADS_PER_SLOT" -lt 1 ]] && _THREADS_PER_SLOT=1
export OMP_NUM_THREADS="\$_THREADS_PER_SLOT"
export MKL_NUM_THREADS="\$_THREADS_PER_SLOT"
# SEPARATE from OMP/MKL above -- torch._inductor's own compile-worker
# subprocess pool (the \`compile_worker/__main__.py --workers=N\`
# processes visible in \`ps aux\` during "getting the model ready")
# defaults to min(32, cpu_count) PER PROCESS via its own
# TORCHINDUCTOR_COMPILE_THREADS knob (torch/_inductor/config.py,
# decide_compile_threads()) -- confirmed reading that source directly:
# it is NOT affected by OMP_NUM_THREADS/MKL_NUM_THREADS at all, those
# only govern the steady-state tensor-math thread pool. Left uncapped,
# $num_slots concurrent processes can spin up $num_slots x 32 compile-
# worker subprocesses all competing for the same cores during each
# process's own compilation burst -- a separate, more concentrated
# contention spike than the steady-state threads, and very likely why
# "getting models ready" felt slower once concurrent batches became
# routine even after the OMP/MKL fix above.
export TORCHINDUCTOR_COMPILE_THREADS="\$_THREADS_PER_SLOT"
# Persist this arm's own full stdout/stderr on the pod (sweep_logs/
# already exists and is already rsync'd back by this script's own
# pull-back step below) -- confirmed the hard way this session that a
# crash's traceback was otherwise UNRECOVERABLE: this SSH pipe's
# output only ever reached whichever local terminal launched it, and
# once that process exits (crash, or the pod being torn down before
# anyone thought to scroll back), the actual error is gone for good.
# \`tee\` still forwards everything to this pipe's stdout too, so the
# local \"[arm/rN] \"-tagged live view is unaffected; \`set -o pipefail\`
# (already active) means a failing python still fails this pipeline
# even though tee itself always exits 0.
mkdir -p sweep_logs
python train_production_transformer_deep_dive.py \\
  --arm $arm --round $round --max-steps $max_steps --max-hours $max_hours \\
  --val-every 500 $WANDB_FLAG $fresh_flags $group_flags $patience_flags $extra \\
  2>&1 | tee -a "sweep_logs/r${round}_${arm}.log"
REMOTE
  ) &
  eval "$outvar=\$!"
}

# Launch every active slot, each over its own SSH connection, all near-
# simultaneously -- SLOT_PIDS[$i] holds slot $i's background pid.
declare -a SLOT_PIDS=() SLOT_RCS=()
for _i in "${!SLOT_ARMS[@]}"; do
  launch_training "${SLOT_ARMS[$_i]}" "${SLOT_ROUNDS[$_i]}" "${SLOT_STEPS[$_i]}" \
    "${SLOT_HOURS[$_i]}" "${SLOT_FRESH[$_i]}" "SLOT_PIDS[$_i]" "${SLOT_EXTRA[$_i]}"
done

for _i in "${!SLOT_ARMS[@]}"; do
  SLOT_RCS[$_i]=0
  wait "${SLOT_PIDS[$_i]}" || SLOT_RCS[$_i]=$?
  if [[ "${SLOT_RCS[$_i]}" != "0" ]]; then
    echo "" >&2
    echo "[${SLOT_ARMS[$_i]}/r${SLOT_ROUNDS[$_i]}] training exited non-zero" >&2
    echo "(${SLOT_RCS[$_i]}) -- pull-back below still runs, to grab whatever" >&2
    echo "checkpoint progress exists." >&2
  fi
done

echo ""
echo "=================================================================="
echo " Pulling back ONLY the new artifacts (not the training data)"
echo "=================================================================="
declare -a CKPT_OKS=()
for _i in "${!SLOT_ARMS[@]}"; do
  CKPT_OKS[$_i]=1
  if ! rsync_with_retry "checkpoints (saved_models/r${SLOT_ROUNDS[$_i]}_${SLOT_ARMS[$_i]}_*)" \
      "root@$POD_HOST:/workspace/cgan/transformer_neurIPS/saved_models/r${SLOT_ROUNDS[$_i]}_${SLOT_ARMS[$_i]}_"* \
      "$TRANSFORMER_DIR/saved_models/"; then
    CKPT_OKS[$_i]=0
    PULL_BACK_OK=0
  fi
done
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
for _i in "${!SLOT_ARMS[@]}"; do
  echo "  checkpoints (${SLOT_ARMS[$_i]}/r${SLOT_ROUNDS[$_i]}): $([[ "${CKPT_OKS[$_i]}" == "1" ]] && echo OK || echo "FAILED -- see above")"
  find "$TRANSFORMER_DIR/saved_models" -maxdepth 1 -name "r${SLOT_ROUNDS[$_i]}_${SLOT_ARMS[$_i]}_*" -newermt "-15 min" \
    -exec ls -la {} \; 2>/dev/null | sed 's/^/    /'
done
echo "  sweep_logs:  $([[ "$LOGS_OK" == "1" ]] && echo OK || echo "FAILED -- see above")"
echo ""
if [[ "$PULL_BACK_OK" == "1" ]]; then
  echo "Done. Results under saved_models/ and sweep_logs/ (local)."
  echo "Pod will now be terminated (trap on EXIT) unless KEEP_POD=1 was set."
else
  echo "AT LEAST ONE PULL-BACK STEP FAILED -- see cleanup_pod()'s message" >&2
  echo "below for how to retrieve manually before this pod is terminated." >&2
fi
