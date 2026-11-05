"""One-off analysis script (OVERVIEW.md v8.1, strategy #1): evaluate
whether "model soup" weight averaging (Wortsman et al.) helps on top of
the best checkpoints branches M/P/Q/U/W have already produced -- zero new
training compute, pure local inference against real weights already on
disk.

Two averageable clusters exist among this session's checkpoints (verified
by inspecting each checkpoint's own stored `config` blob -- same
TOKENIZATION/VARIANT/EMBED_SIZE/N_LAYERS/N_HEADS/AUX_HEAD_FRAMES):
  - token-native: p1_bestyet_adamw, p2_bestyet_lion, p3_bestyet_sophia,
    q1_sophia_ema, q3_sophia_rollout_dominant
  - frame-native (AUX_HEAD_FRAMES=0): r1_spatial_smooth, u2_frame_control,
    u3_frame_rollout_dominant, s1_aropt_frame
`q2_sophia_auxhead` (AUX_HEAD_FRAMES=8, extra `aux_head.*` params) is a
DIFFERENT architecture and is evaluated standalone, never souped.

Evaluated at FULL resolution (Config.HALF_TIME_MODE left at its default
False) -- every ingredient here was trained at NUM_TIME=80, and
HALF_TIME_MODE re-indexes frames to a shorter 0..(NUM_TIME/2-1) position
axis, which would look up the WRONG position-embedding rows for a model
that learned the full 0..79 axis (not a cost-saving shortcut, an actual
correctness bug for this specific use). HALF_TIME_MODE is for NEW
training runs, not re-scoring already-trained checkpoints.

For each cluster: evaluates every individual ingredient (fresh, via the
same evaluate() every arm's own training run uses, so numbers here are
directly comparable to earlier reported ones), then builds a GREEDY soup
(Wortsman et al.'s own selection procedure, not blind uniform averaging):
sort ingredients by their own individual rollout_mse, start the soup with
the best one, then try adding each remaining ingredient in order --
keep the addition only if it does not make the running soup's rollout_mse
worse.
"""
import os
import sys

os.environ.setdefault("NO_COLOR", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from train_production_transformer_deep_dive import (  # noqa: E402
    Config, TransformerDataset, average_state_dicts, evaluate,
    resolve_train_regime,
)
from model_variants import get_model  # noqa: E402

CKPT_DIR = Config.CHECKPOINT_DIR

TOKEN_NATIVE_CLUSTER = [
    "r2_p1_bestyet_adamw_best",
    "r2_p2_bestyet_lion_best",
    "r2_p3_bestyet_sophia_best",
    "r2_q1_sophia_ema_best",
    "r2_q3_sophia_rollout_dominant_best",
]
FRAME_NATIVE_CLUSTER = [
    "r2_r1_spatial_smooth_best",
    "r2_u2_frame_control_best",
    "r2_u3_frame_rollout_dominant_best",
    "r2_s1_aropt_frame_best",
]


def load_ckpt(name):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    # These checkpoints were trained with torch.compile ON (before OVERVIEW.md
    # v7.19 defaulted it off), and the raw model_state_dict payload was saved
    # straight from the still-compiled wrapper -- every key carries a
    # "_orig_mod." prefix (torch.compile's OptimizedModule). Strip it, same
    # unwrap convention this file already uses on the live model object
    # elsewhere (getattr(model, "_orig_mod", model)), just applied to a
    # loaded state_dict's keys instead.
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd, ckpt["config"]



# Absolute filesystem paths stored in a checkpoint's own config_dict()
# reflect wherever it was TRAINED (usually a pod's /workspace/... layout),
# not this machine -- always use the live local Config's own paths for
# these instead of trusting the checkpoint's snapshot.
_LOCAL_PATH_FIELDS = (
    "DECODER_SCRIPTED_PATH", "TRAIN_H5", "VAL_H5", "CHECKPOINT_DIR",
    "RIDGE_MAP_PATH", "WARM_START_PATH",
)


def build_and_eval(state_dict, cfg_dict, device, regime, val_data):
    """Build a fresh model from cfg_dict's architecture fields, load
    state_dict, run the real evaluate() against val_data."""
    ns = type("Cfg", (), {})()
    for k, v in cfg_dict.items():
        setattr(ns, k, v)
    for k in _LOCAL_PATH_FIELDS:
        if hasattr(Config, k):
            setattr(ns, k, getattr(Config, k))
    # VAL_ROLLOUT_SEQS: ALWAYS use the live Config's current value (this
    # script's own cost-control knob, set in main()), never the checkpoint's
    # own historical value -- unlike _LOCAL_PATH_FIELDS this key IS always
    # present in a real checkpoint's config_dict(), so the generic
    # missing-field fallback below would never reach it.
    ns.VAL_ROLLOUT_SEQS = Config.VAL_ROLLOUT_SEQS
    # evaluate()/teacher_forced() read a handful of fields not always
    # present in an older checkpoint's own config_dict() snapshot -- fall
    # back to the live Config's current value for anything missing.
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k)):
            continue
        if not hasattr(ns, k):
            setattr(ns, k, getattr(Config, k))

    model = get_model(ns).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    amp_dtype = regime.amp_dtype if regime.use_amp else None
    # regime.eval_micro_batch is 1 on MPS/CPU (chosen conservatively for the
    # AR rollout's activation growth against a full 25k-row val set) -- for
    # THIS script's much smaller val subset + VAL_ROLLOUT_SEQS, the real
    # bottleneck measured was per-call MPS dispatch overhead, not memory or
    # raw FLOPs (11s of actual CPU time over 4.5 minutes wall-clock for one
    # ingredient at micro_batch=1). Batching more sequences per call cuts
    # the NUMBER of calls without changing total work -- overridable via
    # MODEL_SOUP_BATCH for this exploratory script specifically; the real
    # per-arm training runs are unaffected, they never import this file.
    batch = int(os.environ.get("MODEL_SOUP_BATCH", "16"))
    metrics = evaluate(model, val_data, ns, str(device),
                       amp_dtype=amp_dtype,
                       chunk=batch,
                       tf_batch_size=batch)
    return metrics


QUICK_MODE = os.environ.get("MODEL_SOUP_QUICK", "1") == "1"


def run_cluster(name, members, device, regime, val_data, known_best=None):
    """QUICK_MODE (default): evaluate only `known_best` (the individual
    ingredient with the best previously-reported CUDA production number --
    hardcoded from the real numbers already reported this session, not
    re-derived here) plus ONE uniform soup of every member -- 2 evaluations
    total instead of the full greedy search's ~2N-1. MPS at micro_batch-ish
    scale proved too slow per-evaluation for the full search in one sitting
    (a single ingredient took several minutes even after batching); this
    trades search thoroughness for a real, same-subset, apples-to-apples
    answer to "does souping help at all" within a reasonable wall-clock
    budget. Set MODEL_SOUP_QUICK=0 for the original full greedy search.
    """
    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}", flush=True)
    loaded = [(m, *load_ckpt(m)) for m in members]

    if QUICK_MODE and known_best is not None:
        best_name = known_best
        best_sd, best_cfg = next((sd, cfg) for m, sd, cfg in loaded if m == best_name)
        print(f"  Evaluating known-best ingredient only: {best_name}", flush=True)
        best_metrics = build_and_eval(best_sd, best_cfg, device, regime, val_data)
        print(f"    rollout_mse={best_metrics['rollout_mse']:.6f}  "
             f"improvement_pct={best_metrics['improvement_pct']:+.2f}%", flush=True)

        print(f"\n  Evaluating ONE uniform soup of all {len(loaded)} members:", flush=True)
        uniform_sd = average_state_dicts([sd for _, sd, _ in loaded])
        soup_metrics = build_and_eval(uniform_sd, best_cfg, device, regime, val_data)
        soup_members = [m for m, _, _ in loaded]
        print(f"    rollout_mse={soup_metrics['rollout_mse']:.6f}  "
             f"improvement_pct={soup_metrics['improvement_pct']:+.2f}%", flush=True)
    else:
        loaded_with_metrics = []
        for m, sd, cfg_dict in loaded:
            metrics = build_and_eval(sd, cfg_dict, device, regime, val_data)
            loaded_with_metrics.append((m, sd, cfg_dict, metrics))
            print(f"  {m:38s} rollout_mse={metrics['rollout_mse']:.6f}  "
                  f"improvement_pct={metrics['improvement_pct']:+.2f}%", flush=True)

        # Greedy soup: sort by individual rollout_mse ascending (best first).
        loaded_with_metrics.sort(key=lambda t: t[3]["rollout_mse"])
        best_name, best_sd, best_cfg, best_metrics = loaded_with_metrics[0]
        soup_members = [best_name]
        soup_sd = best_sd
        soup_metrics = best_metrics
        print(f"\n  Greedy soup, starting from best ingredient ({best_name}):", flush=True)
        for m, sd, cfg_dict, metrics in loaded_with_metrics[1:]:
            candidate_sd = average_state_dicts([soup_sd, sd])
            candidate_metrics = build_and_eval(candidate_sd, best_cfg, device, regime, val_data)
            if candidate_metrics["rollout_mse"] <= soup_metrics["rollout_mse"]:
                print(f"    + {m:36s} -> rollout_mse={candidate_metrics['rollout_mse']:.6f} "
                     f"(improved from {soup_metrics['rollout_mse']:.6f}) -- KEPT", flush=True)
                soup_sd = candidate_sd
                soup_metrics = candidate_metrics
                soup_members.append(m)
            else:
                print(f"    - {m:36s} -> rollout_mse would be {candidate_metrics['rollout_mse']:.6f} "
                     f"(worse than {soup_metrics['rollout_mse']:.6f}) -- SKIPPED", flush=True)

    print(f"\n  Final soup: {soup_members}", flush=True)
    print(f"  Best individual ingredient: {best_name}  "
         f"improvement_pct={best_metrics['improvement_pct']:+.2f}%", flush=True)
    print(f"  Greedy soup result:                     "
         f"improvement_pct={soup_metrics['improvement_pct']:+.2f}%", flush=True)
    delta = soup_metrics["improvement_pct"] - best_metrics["improvement_pct"]
    verdict = "SOUP WINS" if delta > 0 else ("TIE" if delta == 0 else "soup loses")
    print(f"  Delta vs. best individual: {delta:+.2f} points -- {verdict}", flush=True)
    return best_metrics, soup_metrics, soup_members


def main():
    # MODEL_SOUP_DEVICE override: a real MPS stall was observed running this
    # script (the AR rollout loop inside evaluate() sat "uninterruptible
    # sleep," barely consuming CPU, for 5+ minutes on a single evaluation --
    # a backend-level dispatch/sync pathology, not something fixable by
    # tuning batch/subset size). CPU's single-queue, fully synchronous
    # execution is more predictable for this exact sequential-rollout
    # workload -- defaulting to it here for reliability over raw speed.
    device_name = os.environ.get(
        "MODEL_SOUP_DEVICE",
        "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    regime = resolve_train_regime(device)
    print(regime.banner, flush=True)

    # Exploratory/directional check, not the final production number: the
    # teacher-forced pass iterates over ALL rows of whatever val_data it's
    # given, at micro_batch=1 on MPS -- the full 25,410-row val_80.h5 would
    # take a very long time per ingredient x ~9 ingredients x ~7 greedy-soup
    # candidate evals (confirmed: even a 2032-row subset took ~4 minutes for
    # ONE ingredient). Cut hard on both axes for a first pass -- a few
    # hundred TF rows and a much smaller rollout population than the
    # production VAL_ROLLOUT_SEQS=64 default. Override via
    # MODEL_SOUP_VAL_SUBSET_RATIO=1.0 / MODEL_SOUP_ROLLOUT_SEQS=64 for the
    # full, slow, rigorous number once a candidate looks promising here.
    subset_ratio = float(os.environ.get("MODEL_SOUP_VAL_SUBSET_RATIO", "0.012"))
    Config.VAL_ROLLOUT_SEQS = int(os.environ.get("MODEL_SOUP_ROLLOUT_SEQS", "12"))
    val_ds = TransformerDataset(Config.VAL_H5, subset_ratio=subset_ratio)
    val_data = val_ds.data
    print(f"val data: {tuple(val_data.shape)} (subset_ratio={subset_ratio})", flush=True)

    # Known-best per cluster, from this session's own real CUDA production
    # numbers (§57/59/61): q1_sophia_ema +40.8% (token-native),
    # u3_frame_rollout_dominant +41.3% (frame-native) -- used only to pick
    # which single ingredient QUICK_MODE evaluates fresh here, not trusted
    # as the actual comparison number (that comes from THIS script's own
    # same-subset evaluation of it, below).
    run_cluster("Token-native cluster (p1/p2/p3/q1/q3)",
               TOKEN_NATIVE_CLUSTER, device, regime, val_data,
               known_best="r2_q1_sophia_ema_best")
    run_cluster("Frame-native cluster (r1/u2/u3/s1)",
               FRAME_NATIVE_CLUSTER, device, regime, val_data,
               known_best="r2_u3_frame_rollout_dominant_best")


if __name__ == "__main__":
    main()
