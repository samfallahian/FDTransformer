"""
Concurrent arm launcher + aggregator for the deep-dive sweep.

    # everything: diagnostics, then 5 arms in parallel, then one report
    python sweep_deep_dive.py --round 1 --max-parallel 5 --max-steps 6000

    # a cheap smoke test first -- STRONGLY recommended before the real run
    python sweep_deep_dive.py --round 1 --smoke

    # round 2, from the branch the round-1 report identified
    python sweep_deep_dive.py --round 2 --branch D --max-parallel 5

The single artefact to upload back is

    sweep_logs/<run_id>/UPLOAD_ME.md

which is written incrementally, so it is worth reading even if the sweep is
still running or died. It contains the diagnostics, a per-arm table, the
per-frame improvement breakdown, the auto-classified decision-tree branch and
the recommended Round 2 arms. Full per-arm stdout stays in
`sweep_logs/<run_id>/<arm>.log` and is NOT needed for the next decision.

WHY THE CLOCK IS STEPS, NOT TIME
================================
Arms sharing a GPU slow each other down unequally (the frame-tokenised arm is
~26x cheaper per rollout than the token-level arms). Under a wall-clock budget
that silently hands some arms more gradient steps than others, so the ranking
would partly measure scheduling luck. Every arm gets the same `--max-steps`, and
wall time is recorded as an observation rather than used as the budget.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(HERE, "train_production_transformer_deep_dive.py")
SWEEP_ROOT = os.path.join(HERE, "sweep_logs")

# Works whether this is launched as a script from inside transformer_neurIPS/ or
# imported as part of the namespace package from the repo root.
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from train_production_transformer_deep_dive import ROUND1_ARMS, ROUND2_ARMS  # noqa: E402
except ImportError:
    from transformer_neurIPS.train_production_transformer_deep_dive import (  # noqa: E402
        ROUND1_ARMS, ROUND2_ARMS)


# --------------------------------------------------------------------------- #
# Decision tree
# --------------------------------------------------------------------------- #
# Thresholds are on `improvement_pct` -- rollout MSE versus the persistence
# baseline over the full 28-frame horizon, both scored on the same rows.
BEATS = 5.0          # below this an arm has not meaningfully beaten persistence
STRONG = 30.0        # above this the plateau is broken and the job is to scale
SEPARATES = 3.0      # an arm must beat the control by this much to count as a win
GAP_OVERFIT = 0.25   # val_tf_loss - train_loss, relative to train_loss
# "Close enough to persistence that the interesting question is the task, not the
# model." Below this the model is simply broken, and a weak linear baseline is no
# longer evidence that persistence is hard to beat.
NEAR_PARITY = -50.0


def classify(diag, results):
    """Pick the Round 2 branch from the Round 1 outcome.

    Returns (branch_key, list_of_reason_strings). Mirrors
    Documentation/deep_dive_decision_tree.md -- keep the two in step.
    """
    reasons = []
    ok = {name: r for name, r in results.items() if r.get("final_improvement_pct") is not None}
    if not ok:
        return "L", ["No arm produced a rollout score at all; check the per-arm logs."]

    # A failed causality probe invalidates every number downstream, so it is
    # checked before anything is interpreted.
    leaky = [n for n, r in results.items()
             if r.get("causality_probe", {}).get("causal") is False]
    if leaky:
        return "L", [f"Causality probe FAILED for: {', '.join(sorted(leaky))}. "
                     f"No metric below can be trusted."]

    imp = {n: r["final_improvement_pct"] for n, r in ok.items()}
    control = imp.get("a0_control")
    best_arm = max(imp, key=imp.get)
    best = imp[best_arm]
    reasons.append(f"best arm = {best_arm} at {best:+.2f}% vs persistence"
                   + (f"; control = {control:+.2f}%" if control is not None else ""))

    # centroid-space (decoded m/s), NOT raw-latent -- this is the field that's
    # apples-to-apples with `best`/`imp`, which are also centroid-space (see
    # evaluate()'s IMPROV%). See linear_frame_baseline()'s docstring.
    lin = (diag or {}).get("linear_baseline", {}).get("improvement_pct_centroid")
    if lin is not None:
        reasons.append(f"ridge linear frame-map baseline = {lin:+.2f}% (centroid space)")

    # THE SANITY FLOORS, checked before anything else is interpreted. A model that
    # cannot beat the best CONSTANT on its own training objective has learned
    # nothing, and no reasoning about exposure bias or capacity applies to it.
    # This is the check whose absence let a badly broken run read as a plateau:
    # the previously saved checkpoint recorded train L2 = 0.4266 while predicting
    # a constant scores ~0.070 and copying the previous frame scores 0.0174 on the
    # same objective.
    def _floors(label):
        out = []
        for n, r in ok.items():
            tl = (r.get("best") or {}).get("train_loss")
            fl = r.get("constant_floor" if label == "constant" else "anchor_floor")
            if tl is not None and fl is not None:
                out.append(f"{n}: best train {tl:.5g} vs {label} floor {fl:.5g}")
        return out

    beat_const = [n for n, r in ok.items() if r.get("beat_constant_predictor")]
    if not beat_const:
        return "N", reasons + ["NO arm beat the best CONSTANT predictor on its own "
                               "training objective -- nothing was learned at all."] \
            + _floors("constant")
    if len(beat_const) < len(ok):
        reasons.append("arms that failed to beat a constant predictor: "
                       + ", ".join(sorted(set(ok) - set(beat_const))))

    # The previous-frame anchor IS the persistence baseline expressed in the
    # training objective. An arm that cannot beat it in-sample cannot beat
    # persistence in rollout, so a large negative improvement here is a broken
    # model -- NOT evidence that persistence is hard to beat.
    anchored = [n for n, r in ok.items() if r.get("beat_frame_anchor")]
    reasons.append(f"beat the previous-frame anchor on the training objective: "
                   f"{len(anchored)}/{len(ok)}")
    if not anchored and best < NEAR_PARITY:
        return "N", reasons + [
            f"No arm beat the previous-frame anchor even in-sample, and the best "
            f"rollout is {best:+.1f}% (far below parity). The models are broken, not "
            f"the task."] + _floors("anchor")

    # A one-step-good / horizon-bad profile is a different disease from a
    # uniformly-flat one, so check the per-frame shape before the levels.
    br = ok.get(best_arm, {})
    per_frame = br.get("final_per_frame_improvement_pct") or []
    frame1 = per_frame[0] if per_frame else None

    # Overfitting is orthogonal to the rest and is diagnosed from the control
    # arm's own train/val gap -- no arm had to be spent measuring it.
    ctrl = ok.get("a0_control", {})
    gap, tr = ctrl.get("final_train_val_gap"), (ctrl.get("final", {}) or {}).get("train_loss")
    if gap is not None and tr:
        rel = gap / abs(tr)
        reasons.append(f"control train/val gap = {gap:+.4f} ({rel:+.1%} of train loss)")
        if rel > GAP_OVERFIT and best < STRONG:
            return "G", reasons + ["Val loss sits far above train loss: the model is "
                                   "memorising ~15k sequences before it generalises."]

    if lin is not None and lin > 15.0 and best < lin * 0.5:
        return "R", reasons + ["A ridge regression beats the transformer by 2x. The "
                               "model or the framing is broken, not the task."]

    if best >= STRONG:
        return "S", reasons + ["The plateau is broken; the remaining work is scaling "
                               "and refinement."]

    if best >= BEATS and (control is None or best - control >= SEPARATES):
        # a1_nonorm winning would mean normalisation HURT, and a2_mse winning
        # means the objective geometry mattered -- both are optimisation stories,
        # so both land in O.
        branch = {"a3_delta": "D", "a4_frame": "F",
                  "a1_nonorm": "O", "a2_mse": "O"}.get(best_arm, "O")
        return branch, reasons + [f"{best_arm} separated from the control by "
                                  f"{best - (control or 0):+.2f} points."]

    if frame1 is not None and frame1 >= 15.0 and best < BEATS:
        return "A", reasons + [f"One frame ahead is {frame1:+.2f}% but the full horizon "
                               f"is {best:+.2f}%: single-step prediction works and error "
                               f"accumulation destroys it."]

    # Only conclude "the task is the problem" when the models are actually NEAR
    # persistence. A weak linear baseline plus a model that is 1000x worse than
    # persistence is a broken model, not a hard task.
    if lin is not None and lin < BEATS and best >= NEAR_PARITY:
        return "T", reasons + [f"Neither the transformer ({best:+.2f}%) NOR a linear map "
                               f"({lin:+.2f}%) beats persistence, and the models ARE near "
                               f"parity. Persistence is genuinely strong at this sampling "
                               f"rate; change the problem, not the model."]
    if best < NEAR_PARITY:
        return "N", reasons + [f"Best arm is {best:+.1f}%, far below parity with a trivial "
                               f"baseline. Fix conditioning before interpreting anything "
                               f"else."] + _floors("anchor")

    return "O", reasons + ["No arm separated from the control by a meaningful margin."]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v, spec="+.2f", na="   --  "):
    if v is None:
        return na
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _pct(v):
    """Improvement percentages span 0 to -10^7 when a model is broken.

    A fixed '+.2f' overflows its column and destroys the table alignment, so
    switch to a compact exponent past four digits.
    """
    if v is None:
        return "  --  "
    try:
        return f"{v:+.2f}" if abs(v) < 10000 else f"{v:+.2e}"
    except (TypeError, ValueError):
        return str(v)


def _sparkline(values, lo=None, hi=None):
    """Compact per-frame trace, so a 28-number curve costs one line not 28."""
    if not values:
        return ""
    # No space as the lowest level: in a report a blank reads as missing data
    # rather than as "worst".
    blocks = "._-=+*#%@"
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi - lo < 1e-9:
        return blocks[len(blocks) // 2] * len(values)
    return "".join(blocks[min(len(blocks) - 1,
                              int((v - lo) / (hi - lo) * (len(blocks) - 1)))]
                   for v in values)


def render_report(run_dir, meta, diag, results, statuses):
    L = []
    add = L.append

    branch, reasons = classify(diag, results)
    blob = ROUND2_ARMS.get(branch, {})

    add(f"# Deep-dive sweep report -- round {meta['round']}  (`{meta['run_id']}`)")
    add("")
    add("## VERDICT")
    add("")
    add(f"**Branch `{branch}`: {blob.get('title', 'unknown')}**")
    add("")
    for r in reasons:
        add(f"- {r}")
    add("")
    if blob.get("arms"):
        add("Recommended next command:")
        add("")
        add("```")
        add(f"python sweep_deep_dive.py --round 2 --branch {branch} "
            f"--max-parallel {meta['max_parallel']} --max-steps {meta['max_steps']}")
        add("```")
        add("")
        add("which runs:")
        add("")
        for name, spec in blob["arms"].items():
            add(f"- `{name}` -- {spec['desc']}")
    else:
        add("No Round 2 arms: fix the blocker above first, then re-run Round 1.")
    add("")

    # ------------------------------------------------------------------ setup
    add("## Run settings")
    add("")
    add("```")
    for k in ("run_id", "round", "branch_requested", "max_parallel", "max_steps",
              "max_hours", "seed", "subset_ratio", "rollout_seqs", "gpus",
              "started", "finished", "trainer_git_head"):
        if meta.get(k) is not None:
            add(f"{k:<18} = {meta[k]}")
    add("```")
    add("")

    # ------------------------------------------------------------ diagnostics
    add("## Diagnostics (run once, before any training)")
    add("")
    if not diag:
        add("_Diagnostics did not complete -- see `diagnostics.log`._")
    else:
        add("```")
        add(f"torch            = {diag.get('torch_version')}")
        add(f"device           = {diag.get('device')}  gpu = {diag.get('gpu')} "
            f"x{diag.get('gpu_count')}  bf16 = {diag.get('bf16_supported')}")
        add(f"train sequences  = {diag.get('train_sequences')}")
        add(f"val sequences    = {diag.get('val_sequences')}")
        add("```")
        add("")
        add("### 1. Was the old attention call actually leaking the future?")
        add("")
        add("```")
        add(json.dumps(diag.get("legacy_mha_probe", {}), indent=2))
        add("```")
        add("`nn.MultiheadAttention(attn_mask=None, is_causal=True)` -- for the module "
            "API `is_causal` is only a HINT that `attn_mask` already is the causal mask. "
            "`causal: false` here means the previous runs were trained with the future "
            "visible; `outcome: raised` means that call could never have run and "
            "something else was happening.")
        add("")
        add("### 2. The old ConvBlock padding")
        add("")
        add("```")
        for k, v in (diag.get("conv_padding_probe") or {}).items():
            add(f"{k:<28} max change in PAST outputs from a FUTURE perturbation = {v:.3e}")
        add("```")
        add("Non-zero for `symmetric_padding_1 (old)` confirms `padding=1` with "
            "`kernel_size=3` let every token see t+1, once per block.")
        add("")
        add("### 3. Causality of each configuration we are about to train")
        add("")
        add("```")
        for name, p in (diag.get("model_probes") or {}).items():
            if "error" in p:
                add(f"{name:<16} ERROR {p['error']}")
            else:
                add(f"{name:<16} causal={str(p['causal']):<5} "
                    f"before_cut={p['max_change_before_cut']:.3e} "
                    f"after_cut={p['max_change_after_cut']:.3e}")
        add("```")
        add("")
        add("### 4. Sanity floor: what trivial predictors score on the training objective")
        add("")
        for tok, nb in (diag.get("null_baselines") or {}).items():
            add(f"`tokenization={tok}` (target std = {nb.get('_target_std')})")
            add("")
            add("```")
            for name in [k for k in nb if not k.startswith("_")]:
                add(f"{name:<16} l2norm={nb[name]['l2norm']:.6f}  mse={nb[name]['mse']:.3e}")
            add("```")
            add("")
        add("Compare each arm's `train` column against these. A train loss ABOVE the "
            "zero-predictor means the model learned nothing at all, and the cause is "
            "conditioning or optimisation -- not capacity, not exposure bias, not "
            "architecture. The previously saved checkpoint recorded train L2 = 0.4266 "
            "against a zero-predictor score of ~0.074 and a previous-frame score of "
            "~0.017, i.e. it was ~6x worse than outputting nothing.")
        add("")
        add("### 5. Is there learnable temporal structure beyond persistence?")
        add("")
        lin = diag.get("linear_baseline", {})
        add("```")
        add(f"[raw latent space, 470-dim -- NOT comparable to an arm's IMPROV%]")
        add(f"persistence MSE                = {lin.get('persistence_mse')}")
        add(f"ridge linear frame-map MSE     = {lin.get('linear_mse')}")
        add(f"linear improvement             = {_fmt(lin.get('improvement_pct'))}%")
        add(f"linear improvement, 1 frame    = {_fmt(lin.get('improvement_pct_frame1'))}%")
        add("")
        add(f"[decoded centroid velocity space, m/s -- apples-to-apples with IMPROV%]")
        add(f"persistence MSE                = {lin.get('persistence_mse_centroid')}")
        add(f"ridge linear frame-map MSE     = {lin.get('linear_mse_centroid')}")
        add(f"linear improvement             = {_fmt(lin.get('improvement_pct_centroid'))}%")
        add(f"linear improvement, 1 frame    = {_fmt(lin.get('improvement_pct_frame1_centroid'))}%")
        add(f"fit on                         = {lin.get('fit_transitions')} frame transitions")
        add("```")
        add("The CENTROID figure is the floor a competent model must clear -- it is in "
            "the same units evaluate() scores arms in (see `IMPROV%`/`roll MSE`/`pers "
            "MSE` below). A ridge regression that beats persistence there while the "
            "transformer does not means the transformer is broken. The raw-latent figure "
            "above it is informational only: the decoder is a nonlinear map, so a raw-"
            "latent improvement is not the same claim as a decoded-velocity improvement.")
        add("")
        fs = diag.get("feature_stats", {})
        if fs:
            add("### 6. Input feature scales")
            add("")
            add("```")
            add(f"latent std (mean over 47 dims) = {fs.get('latent_std_mean')}")
            add(f"columns 47:52 mean             = {fs.get('meta_cols_mean')}")
            add(f"columns 47:52 std              = {fs.get('meta_cols_std')}")
            add("```")
            add("Those columns (x, y, z, t, param) went through the same `nn.Linear` as "
                "the latents. The size of that mismatch is what `NORMALIZE_FEATURES` "
                "removes.")
    add("")

    # ------------------------------------------------------------------ table
    add("## Arm results")
    add("")
    hdr = (f"{'arm':<14} {'status':<9} {'steps':>7} {'mins':>6} {'params':>7} "
           f"{'IMPROV%':>11} {'frame1%':>11} {'last%':>11} {'roll MSE':>11} "
           f"{'pers MSE':>11} {'val tf':>10} {'train':>10} {'>const?':>8} {'>anch?':>7}")
    add("```")
    add(hdr)
    add("-" * len(hdr))
    order = sorted(results, key=lambda n: -(results[n].get("final_improvement_pct") or -1e9))
    for name in list(order) + [n for n in statuses if n not in results]:
        r = results.get(name, {})
        pf = r.get("final_per_frame_improvement_pct") or []
        fin = r.get("final", {}) or {}
        beat_zero = {True: "yes", False: "NO"}.get(r.get("beat_constant_predictor"), "-")
        beat_anch = {True: "yes", False: "NO"}.get(r.get("beat_frame_anchor"), "-")
        add(f"{name:<14} {statuses.get(name, '?'):<9} "
            f"{str(r.get('steps_completed', '-')):>7} "
            f"{_fmt((r.get('wall_seconds') or 0) / 60.0, '.1f'):>6} "
            f"{_fmt(r.get('params_m'), '.2f'):>7} "
            f"{_pct(r.get('final_improvement_pct')):>11} "
            f"{_pct(pf[0] if pf else None):>11} "
            f"{_pct(pf[-1] if pf else None):>11} "
            f"{_fmt(fin.get('rollout_mse'), '.6f'):>11} "
            f"{_fmt(fin.get('persistence_mse'), '.6f'):>11} "
            f"{_fmt(fin.get('val_tf_loss'), '.6f'):>10} "
            f"{_fmt(fin.get('train_loss'), '.6f'):>10} "
            f"{beat_zero:>8} {beat_anch:>7}")
    add("```")
    add("")
    add("`IMPROV%` is the headline: rollout MSE vs the persistence baseline over the "
        "full 28-frame horizon, with model and baseline scored on the SAME validation "
        "rows. Positive means better than doing nothing.")
    add("")
    add("`>const?` is the sanity floor: did this arm's training loss beat the best "
        "CONSTANT predictor? `NO` means the arm learned nothing at all and its "
        "`IMPROV%` is not worth interpreting. `>anch?` is the same question against "
        "copying the previous time frame -- that anchor IS the persistence baseline "
        "expressed in the training objective, so an arm that cannot beat it in-sample "
        "cannot beat persistence in rollout.")
    add("")
    add("Note: `train` and `val tf` are not comparable between token-level and "
        "frame-level arms -- their loss is a norm over 47 and 1222 dimensions "
        "respectively. `IMPROV%`, `roll MSE` and `pers MSE` are comparable across all "
        "arms, which is why the ranking uses them.")
    add("")

    # -------------------------------------------------------- per-frame shape
    add("## Improvement by rollout horizon")
    add("")
    n_fr = max((len(results[n].get("final_per_frame_improvement_pct") or [])
                for n in order), default=0)
    add(f"One row per arm, {n_fr} predicted time frames left to right. This is the shape "
        f"that tells you whether the model predicts well and then drifts, or never "
        f"predicted well.")
    add("")
    add("```")
    allv = [v for n in order for v in (results[n].get("final_per_frame_improvement_pct") or [])]
    lo, hi = (min(allv), max(allv)) if allv else (0, 1)
    add(f"scale: '{_sparkline([lo], lo, hi)}' = {lo:+.1f}%   "
        f"'{_sparkline([hi], lo, hi)}' = {hi:+.1f}%")
    add("")
    for name in order:
        pf = results[name].get("final_per_frame_improvement_pct") or []
        add(f"{name:<14} |{_sparkline(pf, lo, hi)}|  "
            f"f1={_pct(pf[0] if pf else None)}  f28={_pct(pf[-1] if pf else None)}")
    add("```")
    add("")
    for name in order:
        pf = results[name].get("final_per_frame_improvement_pct") or []
        if pf:
            add(f"- `{name}`: {', '.join(f'{v:+.1f}' for v in pf)}")
    add("")

    # ------------------------------------------------------------------ curves
    add("## Training curves (subsampled)")
    add("")
    for name in order:
        curves = results[name].get("curves") or []
        if not curves:
            continue
        keep = curves if len(curves) <= 12 else \
            [curves[i] for i in range(0, len(curves), max(1, len(curves) // 12))] + [curves[-1]]
        add(f"### `{name}`")
        add("")
        add("```")
        add(f"{'step':>7} {'train':>10} {'val_tf':>10} {'roll MSE':>11} {'IMPROV%':>9} {'lr':>9} {'min':>6}")
        for c in keep:
            add(f"{c['step']:>7} {c['train_loss']:>10.6f} {c['val_tf_loss']:>10.6f} "
                f"{c['rollout_mse']:>11.6f} {c['improvement_pct']:>+9.2f} "
                f"{c['lr']:>9.2e} {c['wall_seconds'] / 60:>6.1f}")
        add("```")
        add("")

    # ------------------------------------------------------------------- specs
    add("## What each arm was")
    add("")
    for name in list(order) + [n for n in statuses if n not in results]:
        spec = ROUND1_ARMS.get(name) or {}
        if not spec:
            for b in ROUND2_ARMS.values():
                if name in b["arms"]:
                    spec = b["arms"][name]
        add(f"### `{name}`")
        add("")
        add(f"- **what**: {spec.get('desc', '?')}")
        if spec.get("hypothesis"):
            add(f"- **hypothesis**: {spec['hypothesis']}")
        if spec.get("reads_as"):
            add(f"- **reads as**: {spec['reads_as']}")
        cfg = (results.get(name, {}) or {}).get("config", {})
        if cfg:
            shown = {k: cfg.get(k) for k in (
                "VARIANT", "TOKENIZATION", "EMBED_SIZE", "N_LAYERS", "N_HEADS",
                "PREDICT_DELTA", "DELTA_ANCHOR", "NORMALIZE_FEATURES", "USE_ROPE",
                "NOISE_STD", "AR_MODE", "AR_LOSS_WEIGHT", "AR_FRAMES",
                "AR_FEEDBACK_NOISE_STD", "LOSS", "LEARNING_RATE",
                "DROPOUT", "WEIGHT_DECAY", "BATCH_SIZE", "ACCUMULATION_STEPS")}
            add(f"- **config**: `{shown}`")
        st = statuses.get(name)
        if st and st != "ok":
            add(f"- **status**: {st} -- see `{name}.log`")
        add("")

    failed = [n for n, s in statuses.items() if s != "ok"]
    if failed:
        add("## Failures")
        add("")
        for n in failed:
            add(f"- `{n}`: {statuses[n]} (tail of `{n}.log`)")
            tail = os.path.join(run_dir, f"{n}.log")
            if os.path.exists(tail):
                with open(tail, errors="replace") as f:
                    lines = f.read().splitlines()[-25:]
                add("")
                add("```")
                L.extend(lines)
                add("```")
            add("")

    return "\n".join(L) + "\n"


def write_report(run_dir, meta, diag, results, statuses):
    text = render_report(run_dir, meta, diag, results, statuses)
    path = os.path.join(run_dir, "UPLOAD_ME.md")
    with open(path, "w") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------- #
# Launching
# --------------------------------------------------------------------------- #
def gpu_list():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g.strip() for g in vis.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            got = [l.strip() for l in out.stdout.splitlines() if l.strip()]
            if got:
                return got
    except (OSError, subprocess.SubprocessError):
        pass
    return ["0"]


def arm_command(arm, args, run_dir, round_no):
    cmd = [sys.executable, "-u", TRAINER, "--arm", arm, "--round", str(round_no),
           "--out-dir", run_dir,
           "--max-steps", str(args.max_steps), "--max-hours", str(args.max_hours),
           "--val-every", str(args.val_every), "--seed", str(args.seed),
           "--rollout-seqs", str(args.rollout_seqs),
           "--subset-ratio", str(args.subset_ratio)]
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.cpu_data:
        cmd.append("--cpu-data")
    if args.fresh:
        cmd.append("--fresh")
    if args.no_warm_start:
        cmd.append("--no-warm-start")
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.accum:
        cmd += ["--accum", str(args.accum)]
    for s in args.set:
        cmd += ["--set", s]
    return cmd


def run_concurrent(arms, args, run_dir, round_no, gpus, log):
    """Launch arms with at most `max_parallel` alive, round-robin across GPUs."""
    pending = list(arms)
    running = {}          # arm -> (Popen, file handle, gpu, t0)
    statuses = {}
    launched = 0

    while pending or running:
        while pending and len(running) < args.max_parallel:
            arm = pending.pop(0)
            gpu = gpus[launched % len(gpus)]
            launched += 1
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            path = os.path.join(run_dir, f"{arm}.log")
            fh = open(path, "w")
            cmd = arm_command(arm, args, run_dir, round_no)
            fh.write(f"$ CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}\n\n")
            fh.flush()
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=HERE, env=env)
            running[arm] = (proc, fh, gpu, time.time())
            log(f"[launch] {arm} on GPU {gpu} (pid {proc.pid}) -> {path}")

        time.sleep(5.0)

        for arm in list(running):
            proc, fh, gpu, t0 = running[arm]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[arm]
            statuses[arm] = "ok" if rc == 0 else f"exit {rc}"
            log(f"[done]   {arm} rc={rc} after {(time.time() - t0) / 60:.1f} min "
                f"({len(pending)} queued, {len(running)} running)")
    return statuses


def collect(run_dir, arms):
    out = {}
    for arm in arms:
        p = os.path.join(run_dir, f"{arm}.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                out[arm] = json.load(f)
        except (OSError, ValueError):
            pass
    return out


def git_head():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=HERE, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--branch", default=None,
                   help="round 2 only: which decision-tree branch to run (S/D/E/F/A/R/T/O/G)")
    p.add_argument("--arms", nargs="*", default=None,
                   help="explicit arm list, overriding --round/--branch")
    p.add_argument("--max-parallel", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--max-hours", type=float, default=12.0)
    p.add_argument("--val-every", type=int, default=400)
    p.add_argument("--rollout-seqs", type=int, default=64)
    p.add_argument("--subset-ratio", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--cpu-data", action="store_true")
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints for these arms")
    p.add_argument("--no-warm-start", action="store_true",
                   help="cold-start every arm instead of warm-starting from "
                        "DEFAULT_WARM_START_CKPT. Recommended for any sweep "
                        "meant as a controlled comparison BETWEEN arms -- "
                        "without this, every arm silently warm-starts from "
                        "the same external checkpoint (if it exists), which "
                        "biases the comparison rather than testing each "
                        "arm's mechanism from a clean baseline. Also avoids "
                        "a hard failure when that default checkpoint isn't "
                        "present on this box at all.")
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end run (60 steps, 10%% of data) to prove the "
                        "pipeline works before committing GPU-hours")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra Config overrides forwarded to every arm")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        # Small enough to finish in minutes, big enough to exercise every code
        # path: diagnostics, rollout eval, checkpointing, the JSON dump and the
        # report. The NUMBERS from a smoke run are meaningless -- it exists to
        # prove nothing crashes before a real run is committed to.
        args.max_steps = min(args.max_steps, 60)
        args.val_every = 30
        args.rollout_seqs = 8
        args.subset_ratio = min(args.subset_ratio, 0.1)
        args.max_hours = min(args.max_hours, 1.0)
        args.no_wandb = True
        args.fresh = True
        # A 12-frame context means a 728-token sequential rollout per evaluation,
        # which dominates a smoke run. 32 frames of context leaves an 8-frame
        # horizon: same code path, ~3.5x less of it.
        if not any(s.startswith("VAL_CONTEXT_STEPS=") for s in args.set):
            args.set = list(args.set) + ["VAL_CONTEXT_STEPS=32"]

    if args.arms:
        arms = list(args.arms)
    elif args.round == 1:
        arms = list(ROUND1_ARMS)
    else:
        if not args.branch:
            raise SystemExit("round 2 needs --branch (see the round-1 UPLOAD_ME.md verdict) "
                             "or an explicit --arms list")
        blob = ROUND2_ARMS.get(args.branch.upper())
        if not blob:
            raise SystemExit(f"unknown branch {args.branch!r}; "
                             f"expected one of {sorted(ROUND2_ARMS)}")
        if not blob["arms"]:
            raise SystemExit(f"branch {args.branch.upper()} has no arms: "
                             f"{blob['title']}")
        arms = list(blob["arms"])

    run_id = args.run_id or (f"round{args.round}"
                             + (f"_{args.branch.upper()}" if args.branch else "")
                             + ("_smoke" if args.smoke else "")
                             + f"_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir = os.path.join(SWEEP_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    driver_path = os.path.join(run_dir, "sweep.log")
    driver = open(driver_path, "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        driver.write(line + "\n")
        driver.flush()

    gpus = gpu_list()
    meta = {
        "run_id": run_id, "round": args.round,
        "branch_requested": args.branch.upper() if args.branch else None,
        "max_parallel": args.max_parallel, "max_steps": args.max_steps,
        "max_hours": args.max_hours, "seed": args.seed,
        "subset_ratio": args.subset_ratio, "rollout_seqs": args.rollout_seqs,
        "gpus": ",".join(gpus), "trainer_git_head": git_head(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None,
        "arms": arms, "smoke": args.smoke,
    }
    log(f"run_dir  = {run_dir}")
    log(f"arms     = {arms}")
    log(f"gpus     = {gpus}   max_parallel = {args.max_parallel}")
    log(f"budget   = {args.max_steps} optimizer steps/arm, {args.max_hours}h wall safety net")

    # ------------------------------------------------------------ diagnostics
    diag = None
    if not args.skip_diagnostics:
        log("running diagnostics (causality probes + linear frame baseline) ...")
        dlog = os.path.join(run_dir, "diagnostics.log")
        with open(dlog, "w") as fh:
            cmd = [sys.executable, "-u", TRAINER, "--diagnostics-only",
                   "--out-dir", run_dir, "--subset-ratio", str(args.subset_ratio)]
            fh.write(f"$ {' '.join(cmd)}\n\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=HERE, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        dpath = os.path.join(run_dir, "diagnostics.json")
        if rc == 0 and os.path.exists(dpath):
            with open(dpath) as f:
                diag = json.load(f)
            lin = diag.get("linear_baseline", {})
            log(f"diagnostics ok: linear baseline beats persistence by "
                f"{lin.get('improvement_pct_centroid', float('nan')):+.2f}% "
                f"(centroid space; raw-latent was "
                f"{lin.get('improvement_pct', float('nan')):+.2f}%)")
            bad = [n for n, p in (diag.get("model_probes") or {}).items()
                   if p.get("causal") is False]
            if bad:
                log(f"*** CAUSALITY PROBE FAILED for {bad} -- arms will refuse to train ***")
        else:
            log(f"diagnostics FAILED (rc={rc}); see {dlog}. Continuing, but the "
                f"report will be missing its anchors.")

    # Write an early report so there is something uploadable even if the sweep
    # is interrupted.
    write_report(run_dir, meta, diag, {}, {a: "queued" for a in arms})

    if args.diagnostics_only:
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = write_report(run_dir, meta, diag, {}, {})
        log(f"diagnostics-only; report -> {path}")
        driver.close()
        return 0

    # -------------------------------------------------------------- the arms
    statuses = run_concurrent(arms, args, run_dir, args.round, gpus, log)
    results = collect(run_dir, arms)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    path = write_report(run_dir, meta, diag, results, statuses)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"meta": meta, "diagnostics": diag, "results": results,
                   "statuses": statuses}, f, indent=2, default=str)

    branch, reasons = classify(diag, results)
    log("")
    log("=" * 72)
    log(f"VERDICT: branch {branch} -- {ROUND2_ARMS.get(branch, {}).get('title', '?')}")
    for r in reasons:
        log(f"  - {r}")
    log("=" * 72)
    log(f"UPLOAD THIS FILE: {path}")
    size = os.path.getsize(path) / 1024.0
    log(f"({size:.0f} KB)")
    if ROUND2_ARMS.get(branch, {}).get("arms"):
        log(f"Next: python sweep_deep_dive.py --round 2 --branch {branch} "
            f"--max-parallel {args.max_parallel} --max-steps {args.max_steps}")

    # Convenience copy at a stable path, so "upload the latest report" needs no
    # timestamp lookup.
    try:
        shutil.copyfile(path, os.path.join(SWEEP_ROOT, "LATEST_UPLOAD_ME.md"))
        log(f"also copied to {os.path.join(SWEEP_ROOT, 'LATEST_UPLOAD_ME.md')}")
    except OSError:
        pass

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
