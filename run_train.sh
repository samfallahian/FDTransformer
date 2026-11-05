#!/usr/bin/env bash
# =============================================================================
# run_train.sh -- launch ONE training run of the FDTransformer on ONE GPU.
#
# Thin wrapper around train_production_transformer_deep_dive.py. It does not
# change any training logic: it only picks the GPU, points the trainer at the
# data/decoder/warm-start files, runs pre-flight checks, and logs to runs/.
#
# Default (2026-11-05, see docs/CHANGES_2026-11-05.md):
#   arm h11_ridge_distill + "trial112" hyper-parameters, COLD start (random
#   init), latent-consistency loss on, 20000 steps, eval every 500 steps on
#   256 param-stratified rollout sequences, early stop after 5000 steps
#   without a new promoted checkpoint (never before the curricula finish).
#   Warm starting is opt-in: --warm-start [PATH].
#
# Examples
#   ./run_train.sh                                  # default recipe on GPU 0
#   ./run_train.sh --gpu 1                          # same, on GPU 1
#   ./run_train.sh --gpu 1 --round 2 --set USE_META_COLS=False   # meta-cols ablation
#   ./run_train.sh --warm-start                     # opt-in warm start (assets/ ckpt)
#   ./run_train.sh --warm-start saved_models/r400_h11_ridge_distill_rollout_best.pt
#   ./run_train.sh --arm s7_h9_scaled --preset none --round 3
#   ./run_train.sh --detach --wandb-group myexp     # background, survives logout
#   ./run_train.sh --smoke                          # 1-min synthetic check, no data
#   ./run_train.sh --dry-run                        # print checks + command only
#   ./run_train.sh --help
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="$HERE/train_production_transformer_deep_dive.py"

# ----------------------------------------------------------------- defaults --
GPU="${GPU:-0}"
ARM="h11_ridge_distill"
PRESET="auto"                      # auto -> trial112 for h11_ridge_distill, else none
ROUND="1"
DATA_DIR="${PFD_DATA_DIR:-/data/ptv-092026}"
DECODER="${PFD_DECODER_PATH:-$HERE/assets/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt}"
DEFAULT_WARM_START="$HERE/assets/r2_h11_ridge_distill_rollout_best.pt"
WARM_START=""                      # empty = cold start (the default since 2026-11-05)
KEEP_SCHEDULE=0                    # 1 = do not apply the warm-start fine-tune schedule
MAX_STEPS=20000
MAX_HOURS=6.0                      # wall-clock kill switch only; LR schedule uses MAX_STEPS
VAL_EVERY=500
ROLLOUT_SEQS=256
PATIENCE=5000                      # 0 = no early stopping
WANDB_GROUP=""
NO_WANDB=0
FRESH=0
MODE="train"                       # train | smoke | diagnostics | list-arms
DETACH=0
DRY_RUN=0
SETS=()
EXTRA=()

# trial 112 of the Optuna study (OVERVIEW.md §68.7 / §73.1 / §81.6)
TRIAL112=(
  "WEIGHT_DECAY=0.0966331656012948"
  "DROPOUT=0.07345842794271963"
  "EMA_DECAY=0.98"
  "AR_LOSS_WEIGHT=0.6828481795652924"
)

# Applied (before any --set) only when warm starting, unless
# --warm-start-keep-schedule. A warm start restarts the step counter at 0, so
# without this every curriculum restarts too: LR re-warms to its 1e-3 peak,
# the AR-loss weight drops to 0 and the AR horizon to 2 frames -- i.e. the
# already-trained model is fine-tuned mostly teacher-forced at peak LR, which
# is what produced the "peak at step 250, then decline" in the 2026-11-05 run.
WARMSTART_SETS=(
  "AR_FRAMES_START=None"
  "AR_WEIGHT_WARMUP_FRAC=0.0"
  "RIDGE_DISTILL_WARMUP_FRAC=0.0"
  "LEARNING_RATE=3e-4"
  "WARMUP_FRAC=0.01"
)

usage() {
  sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options
  --gpu N              GPU index to train on (default: $GPU). Sets CUDA_VISIBLE_DEVICES=N.
  --arm NAME           arm to train (default: $ARM). See --list-arms.
  --preset NAME        trial112 | none   (default: trial112 for h11_ridge_distill, else none)
  --round N            run tag: checkpoints are saved_models/r<N>_<arm>_*.pt (default: $ROUND).
                       Re-using a round RESUMES from r<N>_<arm>_latest.pt unless --fresh.
  --data-dir DIR       folder holding train_80.h5 / val_80.h5 (default: $DATA_DIR)
  --decoder PATH       frozen GEN3 scripted decoder (default: assets/Model_GEN3_05_..._scripted.pt)
  --warm-start [PATH]  OPT-IN warm start. Without PATH: assets/r2_h11_ridge_distill_rollout_best.pt.
                       Also applies a fine-tune schedule (no curriculum restart, LR 3e-4,
                       1% warmup) before your --set values; see WARMSTART_SETS.
  --warm-start-keep-schedule   warm start WITHOUT that fine-tune schedule
  --no-warm-start      cold start (the default; accepted for old command lines)
  --max-steps N        optimizer steps; also sizes the LR schedule (default: $MAX_STEPS)
  --max-hours H        wall-clock safety net (default: $MAX_HOURS)
  --val-every N        eval cadence in steps (default: $VAL_EVERY)
  --rollout-seqs N     validation rollout sequences per eval (default: $ROLLOUT_SEQS)
  --patience N         early-stop after N steps with no new promoted checkpoint; 0 = off (default: $PATIENCE)
  --set KEY=VALUE      extra Config override (repeatable, applied after the preset)
  --wandb-group NAME   wandb group label
  --no-wandb           disable wandb
  --fresh              ignore an existing r<round>_<arm>_latest.pt (start over)
  --smoke              trainer --smoke-test: synthetic data, proves model builds/is causal/trains
  --diagnostics        only run diagnostics (causality probes + ridge baseline; writes
                       saved_models/ridge_frame_map.pt), then exit
  --list-arms          list every arm and exit
  --detach             run in the background (nohup); follow with: tail -f <log>
  --dry-run            run the checks and print the command, but do not launch
  -- ARGS...           anything after -- is passed verbatim to the trainer

Environment
  PYTHON               interpreter to use (default: .venv/bin/python, else 'uv run python')
  PFD_DATA_LOAD_THREADS, PFD_CUDA_MICRO_BATCH, PFD_COMPILE_MODEL, WANDB_API_KEY, ...
                       are passed through to the trainer unchanged.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------- parse --
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)           GPU="$2"; shift 2 ;;
    --arm)           ARM="$2"; shift 2 ;;
    --preset)        PRESET="$2"; shift 2 ;;
    --round)         ROUND="$2"; shift 2 ;;
    --data-dir)      DATA_DIR="$2"; shift 2 ;;
    --decoder)       DECODER="$2"; shift 2 ;;
    --warm-start)
      if [[ $# -ge 2 && "$2" != -* ]]; then WARM_START="$2"; shift 2
      else WARM_START="$DEFAULT_WARM_START"; shift; fi ;;
    --warm-start-keep-schedule) KEEP_SCHEDULE=1; shift ;;
    --no-warm-start) WARM_START=""; shift ;;
    --max-steps)     MAX_STEPS="$2"; shift 2 ;;
    --max-hours)     MAX_HOURS="$2"; shift 2 ;;
    --val-every)     VAL_EVERY="$2"; shift 2 ;;
    --rollout-seqs)  ROLLOUT_SEQS="$2"; shift 2 ;;
    --patience)      PATIENCE="$2"; shift 2 ;;
    --set)           SETS+=("$2"); shift 2 ;;
    --wandb-group)   WANDB_GROUP="$2"; shift 2 ;;
    --no-wandb)      NO_WANDB=1; shift ;;
    --fresh)         FRESH=1; shift ;;
    --smoke)         MODE="smoke"; shift ;;
    --diagnostics)   MODE="diagnostics"; shift ;;
    --list-arms)     MODE="list-arms"; shift ;;
    --detach)        DETACH=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; EXTRA+=("$@"); break ;;
    *)               die "unknown option '$1' (see --help)" ;;
  esac
done

[[ "$GPU" =~ ^[0-9]+$ ]] || die "--gpu must be a single GPU index (e.g. 0 or 1), got '$GPU'"
[[ "$ROUND" =~ ^[0-9]+$ ]] || die "--round must be an integer, got '$ROUND'"

# -------------------------------------------------------------- interpreter --
if [[ -n "${PYTHON:-}" ]]; then
  read -r -a PY <<< "$PYTHON"
elif [[ -x "$HERE/.venv/bin/python" ]]; then
  PY=("$HERE/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PY=(uv run --project "$HERE" python)
else
  PY=(python3)
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PFD_DATA_DIR="$DATA_DIR"
export PFD_DECODER_PATH="$DECODER"
export PYTHONUNBUFFERED=1

# ----------------------------------------------------------------- list arms --
if [[ "$MODE" == "list-arms" ]]; then
  exec "${PY[@]}" "$TRAINER" --list-arms
fi

# ------------------------------------------------------------ build command --
if [[ "$PRESET" == "auto" ]]; then
  [[ "$ARM" == "h11_ridge_distill" ]] && PRESET="trial112" || PRESET="none"
fi
case "$PRESET" in
  trial112) PRESET_SETS=("${TRIAL112[@]}") ;;
  none)     PRESET_SETS=() ;;
  *)        die "unknown --preset '$PRESET' (trial112 | none)" ;;
esac
[[ -n "$WANDB_GROUP" ]] && SETS+=("WANDB_GROUP=$WANDB_GROUP")
WS_SETS=()
if [[ "$MODE" == "train" && -n "$WARM_START" && $KEEP_SCHEDULE -eq 0 ]]; then
  WS_SETS=("${WARMSTART_SETS[@]}")
fi
# order matters: preset, then warm-start schedule, then the user's --set (last wins)
ALL_SETS=("${PRESET_SETS[@]+"${PRESET_SETS[@]}"}" "${WS_SETS[@]+"${WS_SETS[@]}"}" "${SETS[@]+"${SETS[@]}"}")

STAMP="$(date +%Y%m%d_%H%M%S)"
case "$MODE" in
  train)       RUN_ID="r${ROUND}_${ARM}_gpu${GPU}_${STAMP}" ;;
  smoke)       RUN_ID="smoke_gpu${GPU}_${STAMP}" ;;
  diagnostics) RUN_ID="diagnostics_gpu${GPU}_${STAMP}" ;;
esac
RUN_DIR="$HERE/runs/$RUN_ID"
LOG="$RUN_DIR/train.log"

CMD=("${PY[@]}" -u "$TRAINER" --out-dir "$RUN_DIR")
case "$MODE" in
  smoke)
    CMD+=(--smoke-test --arm "$ARM") ;;
  diagnostics)
    CMD+=(--diagnostics-only) ;;
  train)
    CMD+=(--arm "$ARM" --round "$ROUND"
          --max-steps "$MAX_STEPS" --max-hours "$MAX_HOURS"
          --val-every "$VAL_EVERY" --rollout-seqs "$ROLLOUT_SEQS")
    [[ "$PATIENCE" != "0" ]] && CMD+=(--early-stop-patience-steps "$PATIENCE")
    if [[ -n "$WARM_START" ]]; then CMD+=(--warm-start "$WARM_START"); else CMD+=(--no-warm-start); fi
    [[ $NO_WANDB -eq 1 ]] && CMD+=(--no-wandb)
    [[ $FRESH -eq 1 ]] && CMD+=(--fresh)
    ;;
esac
# NOTE: the trainer's --set is nargs="*": all overrides MUST go in ONE --set.
[[ ${#ALL_SETS[@]} -gt 0 && "$MODE" != "diagnostics" ]] && CMD+=(--set "${ALL_SETS[@]}")
CMD+=("${EXTRA[@]+"${EXTRA[@]}"}")

# --------------------------------------------------------- pre-flight checks --
echo "================================================================"
echo " FDTransformer launcher -- mode: $MODE"
echo "================================================================"
FAIL=0
ok()   { printf "  [OK]      %s\n" "$*"; }
bad()  { printf "  [MISSING] %s\n" "$*"; FAIL=1; }
warn() { printf "  [WARN]    %s\n" "$*"; }

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total \
              --format=csv,noheader -i "$GPU" 2>/dev/null || true)"
  if [[ -n "$GPU_NAME" ]]; then ok "GPU $GPU_NAME"
  else bad "GPU index $GPU not found by nvidia-smi ($(nvidia-smi -L 2>/dev/null | wc -l) GPU(s) present)"; fi
else
  warn "nvidia-smi not found -- the trainer will fall back to MPS/CPU"
fi

if [[ "$MODE" != "smoke" ]]; then
  for f in "$DATA_DIR/train_80.h5" "$DATA_DIR/val_80.h5"; do
    [[ -f "$f" ]] && ok "$f ($(du -h "$f" | cut -f1))" || bad "$f"
  done
fi
if [[ -f "$DECODER" ]]; then ok "decoder $DECODER"
elif [[ "$MODE" == "smoke" ]]; then warn "decoder $DECODER missing (not needed for --smoke, needed for training)"
else bad "decoder $DECODER  (see assets/README.md)"; fi

if [[ "$MODE" == "train" ]]; then
  LATEST="$HERE/saved_models/r${ROUND}_${ARM}_latest.pt"
  if [[ -f "$LATEST" && $FRESH -eq 0 ]]; then
    warn "RESUMING from $LATEST (pass --fresh or a new --round to start over)"
  elif [[ -n "$WARM_START" ]]; then
    [[ -f "$WARM_START" ]] && ok "warm-start $WARM_START" \
      || bad "warm-start $WARM_START  (see assets/README.md, or drop --warm-start)"
    [[ $KEEP_SCHEDULE -eq 0 ]] && warn "warm start: fine-tune schedule applied (${WARMSTART_SETS[*]})"
  else
    ok "cold start (random init) -- pass --warm-start [PATH] to opt in"
  fi
  LOCK="$HERE/saved_models/r${ROUND}_${ARM}.lock"
  [[ -f "$LOCK" ]] && warn "lock file $LOCK exists -- another run of r${ROUND}_${ARM} may be live (stale locks >300s are reclaimed)"
fi

"${PY[@]}" - <<'PYEOF' || FAIL=1
import importlib, sys
mods = ["torch", "numpy", "h5py"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        extra = ""
        if m == "torch":
            extra = f" cuda_build={mod.version.cuda} cuda_available={mod.cuda.is_available()}"
        print(f"  [OK]      {m} {mod.__version__}{extra}")
    except Exception as e:
        print(f"  [MISSING] python package {m}: {e}"); sys.exit(1)
try:
    import wandb; print(f"  [OK]      wandb {wandb.__version__}")
except Exception:
    print("  [WARN]    wandb not importable -- pass --no-wandb")
PYEOF

[[ $FAIL -eq 0 ]] || die "pre-flight checks failed (see above)"

# h11_ridge_distill (and any RIDGE_DISTILL/ridge-anchor arm) needs the fitted
# ridge map that only --diagnostics produces. Fit it once, on this GPU, if absent.
NEED_RIDGE=0
if [[ "$MODE" == "train" && ! -f "$HERE/saved_models/ridge_frame_map.pt" ]]; then
  if [[ "$ARM" == h11_* || "$ARM" == *ridge* ]] || printf '%s\n' "${ALL_SETS[@]+"${ALL_SETS[@]}"}" | grep -q '^RIDGE_DISTILL_WEIGHT='; then
    NEED_RIDGE=1
    warn "saved_models/ridge_frame_map.pt missing -- diagnostics will run first to fit it"
  fi
fi

echo
echo "  GPU          : $GPU   (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  data dir     : $PFD_DATA_DIR"
[[ "$MODE" == "train" ]] && echo "  arm / round  : $ARM / r$ROUND   preset: $PRESET"
[[ ${#ALL_SETS[@]} -gt 0 ]] && echo "  overrides    : ${ALL_SETS[*]}"
echo "  run dir      : $RUN_DIR"
echo "  command      : ${CMD[*]}"
echo

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry run -- nothing launched)"
  exit 0
fi

mkdir -p "$RUN_DIR"
{
  echo "run_id=$RUN_ID"; echo "started=$(date -Is)"; echo "host=$(hostname)"
  echo "gpu=$GPU"; echo "data_dir=$PFD_DATA_DIR"; echo "decoder=$PFD_DECODER_PATH"
  echo "git_head=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "git_dirty=$(git -C "$HERE" status --porcelain 2>/dev/null | wc -l | tr -d ' ') files"
  printf 'command='; printf '%q ' "${CMD[@]}"; echo
} > "$RUN_DIR/run_info.txt"

run_all() {
  cd "$HERE"
  if [[ $NEED_RIDGE -eq 1 ]]; then
    echo "### fitting ridge map (diagnostics) ###"
    "${PY[@]}" -u "$TRAINER" --diagnostics-only --out-dir "$RUN_DIR" || return $?
    echo "### diagnostics done ###"
  fi
  "${CMD[@]}"
}

if [[ $DETACH -eq 1 ]]; then
  export HERE TRAINER RUN_DIR NEED_RIDGE
  # functions/arrays can't be exported reliably -- re-serialise them for the child shell
  CHILD="$(declare -f run_all); $(declare -p PY CMD); run_all; rc=\$?; echo \"finished=\$(date -Is) rc=\$rc\" >> \"$RUN_DIR/run_info.txt\"; exit \$rc"
  nohup setsid bash -c "$CHILD" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$RUN_DIR/pid"
  echo "Launched in background: pid $PID (process group $PID)"
  echo "  follow : tail -f $LOG"
  echo "  stop   : kill -- -$PID"
  echo "           (_latest.pt is written every ~10 min; re-run with the same --round to resume)"
else
  echo "Logging to $LOG"
  set +e
  run_all 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "finished=$(date -Is) rc=$RC" >> "$RUN_DIR/run_info.txt"
  exit "$RC"
fi
