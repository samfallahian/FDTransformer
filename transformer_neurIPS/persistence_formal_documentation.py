"""
Formal documentation figures: r1_a3b_delta_ar_rollout_best.pt vs. persistence.

`tests/test_model_vs_baseline.py` (as currently defaulted, TX_MODELS =
"r1_a3b_delta_ar_rollout_best.pt") answers PASS/FAIL: does the model beat
persistence. This script produces the documentation-grade evidence behind
that answer -- run once, remotely, on the GPU box (it is not a test and is
not run in CI).

WHAT "40-frame setup" MEANS HERE
=================================
Every validation sequence is 40 frames: 12 frames of context, 28 predicted.
At the data's native cadence (12 context frames = 100 ms, i.e. dt = 8.3(3) ms
per frame -- see Documentation/persistence_baseline_design.md for the same
100 ms context convention used elsewhere in this project):

    context : 12 frames =  100.0 ms
    forecast: 28 frames =  233.3 ms
    window  : 40 frames =  333.3 ms

Rather than push the forecast horizon further, this script spends the budget
on evaluation rigor instead:

  - ALL available validation sequences (165 at the default 0.2 subset ratio
    of val_40.h5's 829 sequences -- the same subset TX_DEEPDIVE_SUBSET_RATIO
    uses in the test file), not a handful.
  - Two disjoint sub-populations, reported SEPARATELY rather than pooled:
    "wake-targeted" sequences (centered on one of the 24 (y, z) coordinates
    prepare_data.py deliberately samples in the vortex wake) and
    "random-location" sequences (sampled uniformly outside the wake set).
    They are NOT a 50/50 split of this val file -- see the printed group
    sizes -- because random-location sequences are dropped more often for
    landing outside the data's valid domain (see prepare_data.py).
  - Six horizons: frames [1, 6, 12, 18, 24, 28] = [8.3, 50, 100, 150, 200,
    233.3] ms past the context boundary.
  - Mean, a 95% bootstrap CI (resampling over sequences), and the fraction
    of sequences where the model beats persistence -- at every (group,
    horizon), and per velocity component (vx, vy, vz) as well as overall.
  - Three metrics, all from the same rollout (no separate passes): MAE,
    RMSE, and L2 (mean per-coordinate Euclidean norm of the velocity error,
    i.e. the trainer's own `l2_loss = mean(norm(diff, dim=-1))` geometry).
    RMSE and L2 are NOT derivable after the fact from a saved MAE scalar --
    they need the raw squared errors / per-token vector norms -- so all
    three are computed together here rather than bolted on later.

Reuses (does not reimplement) the checkpoint loading, AE decode, and rollout
machinery already in the test file / trainer:
    tests/test_model_vs_baseline.py : CHECKPOINT_DIR, load_checkpoint,
        normalize_state_dict, apply_checkpoint_config, load_autoencoder,
        decode_latents_to_centroid, BENIGN_MISSING_KEYS
    train_production_transformer_deep_dive.py : Config, TransformerDataset,
        rollout_frames, probe_causality
    model_variants.py : get_model
    prepare_data.py : WAKE_COORDS (the sampling-time wake/random label)

Outputs (under Documentation/persistence_formal/, created if missing):
    results_by_horizon.csv   tidy long-format table: one row per
                              (group, horizon, component)
    horizon_summary.pdf      MAE-vs-horizon (with 95% CI) and fraction-
                              beating-persistence, one row per group
    velocity_components.pdf  MAE-vs-horizon per velocity component, one row
                              per group, one column per component
    report.md                narrative summary of the above, in the same
                              vocabulary (frame1%/last%) the rest of the repo
                              uses

Tunable via environment variables (all optional):
    PFD_RUN              target run name                (default "r1_a3b_delta_ar")
    PFD_KIND             target checkpoint kind          (default "rollout_best")
    PFD_SUBSET_RATIO     val-set fraction to load        (default 0.2)
    PFD_SAMPLES          sequences to use, 0 = all loaded (default 0)
    PFD_HORIZON_FRAMES   comma list of horizon frames    (default "1,6,12,18,24,28")
    PFD_CONTEXT_MS       wall-clock ms the context spans (default 100.0)
    PFD_N_BOOTSTRAP      bootstrap resamples for the CI  (default 10000)
    PFD_BATCH_SIZE       sequences per rollout batch     (default 32)
    PFD_SEED             RNG seed (bootstrap only)       (default 0)
    PFD_OUT_DIR          output directory                (default Documentation/persistence_formal)
"""

import os
import sys
import time
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- path setup, mirrors tests/test_model_vs_baseline.py ---------------------
HERE = os.path.dirname(os.path.abspath(__file__))              # .../transformer_neurIPS
PROJECT_ROOT = os.path.dirname(HERE)                            # .../cgan
TESTS_DIR = os.path.join(HERE, "tests")
for p in (PROJECT_ROOT, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformer_neurIPS.train_production_transformer_deep_dive import (
    Config, TransformerDataset, rollout_frames, probe_causality)
from transformer_neurIPS.model_variants import get_model
from transformer_neurIPS.prepare_data import WAKE_COORDS
from test_model_vs_baseline import (
    CHECKPOINT_DIR, load_checkpoint, normalize_state_dict, apply_checkpoint_config,
    load_autoencoder, decode_latents_to_centroid, BENIGN_MISSING_KEYS)

# --- validated categorical palette (dataviz skill, references/palette.md) ---
# Slots 1 (blue) and 2 (orange) are the model/persistence pair used
# throughout; color follows the entity (model vs. persistence), never the
# group -- wake vs. random-location is encoded as row facets instead of a
# third color, per the "one axis / color follows the entity" rule.
COLOR_MODEL = "#2a78d6"
COLOR_PERSISTENCE = "#eb6834"
COLOR_REFERENCE = "#52514e"   # muted ink for the 50% reference line

VELOCITY_COMPONENTS = ("vx", "vy", "vz")
# "l2" has no per-component variant: it is the vector norm ACROSS vx/vy/vz,
# so it only exists at component="all".
METRICS = ("mae", "rmse", "l2")
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "l2": "L2 (mean vector norm)"}


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


PFD_RUN = os.environ.get("PFD_RUN", "r1_a3b_delta_ar")
PFD_KIND = os.environ.get("PFD_KIND", "rollout_best")
PFD_SUBSET_RATIO = _env_float("PFD_SUBSET_RATIO", 0.2)
PFD_SAMPLES = _env_int("PFD_SAMPLES", 0)  # 0 = all loaded
PFD_HORIZON_FRAMES = [int(x) for x in os.environ.get(
    "PFD_HORIZON_FRAMES", "1,6,12,18,24,28").split(",") if x.strip()]
PFD_CONTEXT_MS = _env_float("PFD_CONTEXT_MS", 100.0)
PFD_N_BOOTSTRAP = _env_int("PFD_N_BOOTSTRAP", 10000)
PFD_BATCH_SIZE = _env_int("PFD_BATCH_SIZE", 32)
PFD_SEED = _env_int("PFD_SEED", 0)
PFD_OUT_DIR = os.environ.get("PFD_OUT_DIR",
                              os.path.join(PROJECT_ROOT, "Documentation", "persistence_formal"))

GROUPS = ("all", "wake", "random")
GROUP_TITLES = {"all": "All sequences", "wake": "Wake-targeted", "random": "Random-location"}


def bootstrap_ci(values, n_boot, ci, rng):
    """Percentile bootstrap CI on the mean of `values` (resampling sequences)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def label_wake_or_random(dataset, num_avail):
    """Per-sequence bool array: True if the sequence's (y, z) is a WAKE_COORDS entry.

    Mirrors prepare_data.py's own wake/random split at generation time: (y, z)
    is stored in feature columns 48/49, constant across every token in a
    sequence, so it is read back directly rather than re-derived.
    """
    wake_set = {(int(y), int(z)) for y, z in WAKE_COORDS}
    ys = dataset.data[:num_avail, 0, 48].numpy().astype(int)
    zs = dataset.data[:num_avail, 0, 49].numpy().astype(int)
    return np.array([(y, z) in wake_set for y, z in zip(ys, zs)])


def load_model():
    ckpt_name = f"{PFD_RUN}_{PFD_KIND}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"{ckpt_path} not found. This script targets one specific "
            f"checkpoint on the remote GPU box; set PFD_RUN/PFD_KIND to "
            f"point elsewhere if you meant a different one.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device              : {device}")
    print(f"checkpoint          : {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    apply_checkpoint_config(ckpt)
    model = get_model(Config)
    state_dict, stripped = normalize_state_dict(ckpt["model_state_dict"])
    if stripped:
        print(f"  stripped state-dict prefix(es): {stripped}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if k not in BENIGN_MISSING_KEYS]
    if missing:
        raise SystemExit(f"{ckpt_name}: {len(missing)} missing key(s), cannot be "
                          f"scored: {missing[:4]}")
    model.eval()
    model.to(device)

    probe = probe_causality(model, Config, device)
    print(f"causal              : {'OK' if probe['causal'] else 'LEAK'} "
          f"(past outputs moved {probe['max_change_before_cut']:.3e}, "
          f"tolerance {probe['tolerance']:.3e})")
    if not probe["causal"]:
        raise SystemExit("Architecture leaks the future; every downstream metric "
                          "would be meaningless. Fix the leak before documenting it.")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"architecture        : variant={getattr(Config, 'VARIANT', '?')} "
          f"embed={getattr(Config, 'EMBED_SIZE', '?')} layers={getattr(Config, 'N_LAYERS', '?')} "
          f"params={params_m:.2f}M epoch={ckpt.get('epoch')}")
    return model, device, ckpt_name


def run_rollouts(model, device, ae, converter, dataset, n_samples, num_x, latent_dim,
                  context_steps, n_frames):
    """Full-horizon rollout for every sequence, batched for speed.

    Returns a dict keyed by metric name ("mae", "rmse", "l2"), each value a
    dict with:
      "model", "pers"       shape (n_samples, n_frames)     -- overall, over 26 coords x 3 comps
      "model_c", "pers_c"   shape (n_samples, n_frames, 3)  -- per velocity component (mae/rmse only;
                                                                l2 is a cross-component vector norm,
                                                                so it has no per-component variant)
    All three metrics come from the SAME rollout -- no separate passes.
    """
    context_len = num_x * context_steps
    metrics = {
        "mae": dict(model=np.zeros((n_samples, n_frames)), pers=np.zeros((n_samples, n_frames)),
                    model_c=np.zeros((n_samples, n_frames, 3)), pers_c=np.zeros((n_samples, n_frames, 3))),
        "rmse": dict(model=np.zeros((n_samples, n_frames)), pers=np.zeros((n_samples, n_frames)),
                     model_c=np.zeros((n_samples, n_frames, 3)), pers_c=np.zeros((n_samples, n_frames, 3))),
        "l2": dict(model=np.zeros((n_samples, n_frames)), pers=np.zeros((n_samples, n_frames))),
    }

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_samples, PFD_BATCH_SIZE):
            end = min(start + PFD_BATCH_SIZE, n_samples)
            batch = dataset.data[start:end].to(device)

            targets_latent = batch[:, context_len:context_len + n_frames * num_x, :latent_dim]
            last_frame = batch[:, context_len - num_x:context_len, :latent_dim]
            persistence_latent = last_frame.repeat(1, n_frames, 1)

            out = rollout_frames(model, batch, Config, ctx_frames=context_steps, n_frames=n_frames)
            B, n, nx, ld = out.shape
            preds_latent = out.reshape(B, n * nx, ld)[:, :n_frames * num_x, :]

            targets_v = decode_latents_to_centroid(targets_latent, ae, converter, device)
            persistence_v = decode_latents_to_centroid(persistence_latent, ae, converter, device)
            preds_v = decode_latents_to_centroid(preds_latent, ae, converter, device)

            for f in range(n_frames):
                sl = slice(f * num_x, (f + 1) * num_x)
                diff_m = preds_v[:, sl] - targets_v[:, sl]          # (B, 26, 3)
                diff_p = persistence_v[:, sl] - targets_v[:, sl]

                metrics["mae"]["model"][start:end, f] = torch.mean(
                    torch.abs(diff_m), dim=(1, 2)).cpu().numpy()
                metrics["mae"]["pers"][start:end, f] = torch.mean(
                    torch.abs(diff_p), dim=(1, 2)).cpu().numpy()
                metrics["rmse"]["model"][start:end, f] = torch.sqrt(torch.mean(
                    diff_m ** 2, dim=(1, 2))).cpu().numpy()
                metrics["rmse"]["pers"][start:end, f] = torch.sqrt(torch.mean(
                    diff_p ** 2, dim=(1, 2))).cpu().numpy()
                # L2: per-coordinate Euclidean norm of the (vx, vy, vz) error,
                # averaged over the 26 coords -- the trainer's own
                # `l2_loss = mean(norm(diff, dim=-1))` geometry.
                metrics["l2"]["model"][start:end, f] = torch.mean(
                    torch.linalg.norm(diff_m, dim=2), dim=1).cpu().numpy()
                metrics["l2"]["pers"][start:end, f] = torch.mean(
                    torch.linalg.norm(diff_p, dim=2), dim=1).cpu().numpy()

                for c in range(3):
                    metrics["mae"]["model_c"][start:end, f, c] = torch.mean(
                        torch.abs(diff_m[:, :, c]), dim=1).cpu().numpy()
                    metrics["mae"]["pers_c"][start:end, f, c] = torch.mean(
                        torch.abs(diff_p[:, :, c]), dim=1).cpu().numpy()
                    metrics["rmse"]["model_c"][start:end, f, c] = torch.sqrt(torch.mean(
                        diff_m[:, :, c] ** 2, dim=1)).cpu().numpy()
                    metrics["rmse"]["pers_c"][start:end, f, c] = torch.sqrt(torch.mean(
                        diff_p[:, :, c] ** 2, dim=1)).cpu().numpy()

            print(f"  rolled out sequences [{start}:{end}) of {n_samples}")

    total = time.perf_counter() - t0
    print(f"rollout wall-clock  : {total:.1f}s for {n_samples} sequences "
          f"({total / n_samples:.3f}s/sequence avg)")
    return metrics


def group_mask(group, is_wake, n_samples):
    if group == "all":
        return np.ones(n_samples, dtype=bool)
    if group == "wake":
        return is_wake
    if group == "random":
        return ~is_wake
    raise ValueError(group)


def _add_row(rows, group, n_group, hf, horizon_ms, component, metric, model_vals, pers_vals,
             n_boot, rng):
    diff = pers_vals - model_vals  # positive = model better
    m_lo, m_hi = bootstrap_ci(model_vals, n_boot, 0.95, rng)
    p_lo, p_hi = bootstrap_ci(pers_vals, n_boot, 0.95, rng)
    d_lo, d_hi = bootstrap_ci(diff, n_boot, 0.95, rng)
    rows.append(dict(
        group=group, n_sequences=n_group, horizon_frames=hf, horizon_ms=horizon_ms,
        component=component, metric=metric,
        model_mean=float(model_vals.mean()), model_ci_lo=m_lo, model_ci_hi=m_hi,
        persistence_mean=float(pers_vals.mean()), persistence_ci_lo=p_lo, persistence_ci_hi=p_hi,
        mean_diff=float(diff.mean()), diff_ci_lo=d_lo, diff_ci_hi=d_hi,
        frac_beating_persistence=float((model_vals < pers_vals).mean()) if n_group else float("nan"),
    ))


def build_table(metrics, is_wake, horizon_frames, dt_ms, n_boot, seed):
    """Tidy long-format rows: one per (group, horizon_frame, component, metric).

    `metrics` is the dict returned by `run_rollouts`. "l2" only ever appears
    with component="all" -- it is a cross-component vector norm, so a
    per-component "l2" value does not exist.
    """
    rng = np.random.default_rng(seed)
    n_samples = metrics["mae"]["model"].shape[0]
    rows = []
    for group in GROUPS:
        mask = group_mask(group, is_wake, n_samples)
        n_group = int(mask.sum())
        for hf in horizon_frames:
            f = hf - 1  # frame index 1 -> array column 0
            horizon_ms = round(hf * dt_ms, 3)
            for metric in METRICS:
                m = metrics[metric]["model"][mask, f]
                p = metrics[metric]["pers"][mask, f]
                _add_row(rows, group, n_group, hf, horizon_ms, "all", metric, m, p, n_boot, rng)
                if metric == "l2":
                    continue  # no per-component variant
                for c, name in enumerate(VELOCITY_COMPONENTS):
                    mc = metrics[metric]["model_c"][mask, f, c]
                    pc = metrics[metric]["pers_c"][mask, f, c]
                    _add_row(rows, group, n_group, hf, horizon_ms, name, metric, mc, pc, n_boot, rng)
    return rows


def write_csv(rows, path):
    cols = ["group", "n_sequences", "horizon_frames", "horizon_ms", "component", "metric",
            "model_mean", "model_ci_lo", "model_ci_hi",
            "persistence_mean", "persistence_ci_lo", "persistence_ci_hi",
            "mean_diff", "diff_ci_lo", "diff_ci_hi", "frac_beating_persistence"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e6e6e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_horizon_summary(rows, horizon_frames, ckpt_name, path, metric="mae"):
    """One row per group: left = metric vs horizon (model vs persistence, 95% CI),
    right = fraction of sequences beating persistence."""
    by_key = {(r["group"], r["horizon_frames"], r["component"], r["metric"]): r for r in rows}
    label = METRIC_LABELS[metric]

    fig, axes = plt.subplots(len(GROUPS), 2, figsize=(11, 3.1 * len(GROUPS)))
    for i, group in enumerate(GROUPS):
        ax_metric, ax_frac = axes[i, 0], axes[i, 1]
        gr = [by_key[(group, hf, "all", metric)] for hf in horizon_frames]
        n_group = gr[0]["n_sequences"]
        xs = [r["horizon_ms"] for r in gr]

        model_mean = [r["model_mean"] for r in gr]
        model_lo = [r["model_ci_lo"] for r in gr]
        model_hi = [r["model_ci_hi"] for r in gr]
        pers_mean = [r["persistence_mean"] for r in gr]
        pers_lo = [r["persistence_ci_lo"] for r in gr]
        pers_hi = [r["persistence_ci_hi"] for r in gr]

        ax_metric.fill_between(xs, model_lo, model_hi, color=COLOR_MODEL, alpha=0.15, linewidth=0)
        ax_metric.fill_between(xs, pers_lo, pers_hi, color=COLOR_PERSISTENCE, alpha=0.15, linewidth=0)
        ax_metric.plot(xs, model_mean, color=COLOR_MODEL, linewidth=2, marker="o",
                        markersize=5, label="Model")
        ax_metric.plot(xs, pers_mean, color=COLOR_PERSISTENCE, linewidth=2, marker="o",
                        markersize=5, label="Persistence")
        ax_metric.set_title(f"{GROUP_TITLES[group]} (n={n_group}) -- centroid {label} vs. horizon")
        ax_metric.set_xlabel("horizon (ms past context boundary)")
        ax_metric.set_ylabel(f"{label} (m/s)")
        _style_axis(ax_metric)
        if i == 0:
            ax_metric.legend(frameon=False, loc="upper left")

        frac = [r["frac_beating_persistence"] for r in gr]
        ax_frac.bar([str(int(round(x))) for x in xs], frac, color=COLOR_MODEL, width=0.6, zorder=2)
        ax_frac.axhline(0.5, color=COLOR_REFERENCE, linewidth=1, linestyle="--", zorder=1)
        ax_frac.set_ylim(0, 1)
        ax_frac.set_title(f"{GROUP_TITLES[group]} (n={n_group}) -- fraction beating persistence ({label})")
        ax_frac.set_xlabel("horizon (ms past context boundary)")
        ax_frac.set_ylabel("fraction of sequences")
        _style_axis(ax_frac)

    fig.suptitle(f"{ckpt_name} vs. persistence -- {label}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_components(rows, horizon_frames, ckpt_name, path, metric="mae"):
    """Rows = groups, columns = velocity components. Metric vs horizon, 95% CI.

    L2 has no per-component variant (it is a cross-component vector norm), so
    this figure is only meaningful for metric in {"mae", "rmse"}.
    """
    if metric == "l2":
        raise ValueError("L2 has no per-component breakdown; use metric='mae' or 'rmse'.")
    by_key = {(r["group"], r["horizon_frames"], r["component"], r["metric"]): r for r in rows}
    label = METRIC_LABELS[metric]

    fig, axes = plt.subplots(len(GROUPS), len(VELOCITY_COMPONENTS),
                              figsize=(4.0 * len(VELOCITY_COMPONENTS), 3.1 * len(GROUPS)),
                              sharex=True)
    for i, group in enumerate(GROUPS):
        for j, comp in enumerate(VELOCITY_COMPONENTS):
            ax = axes[i, j]
            gr = [by_key[(group, hf, comp, metric)] for hf in horizon_frames]
            n_group = gr[0]["n_sequences"]
            xs = [r["horizon_ms"] for r in gr]

            model_mean = [r["model_mean"] for r in gr]
            model_lo = [r["model_ci_lo"] for r in gr]
            model_hi = [r["model_ci_hi"] for r in gr]
            pers_mean = [r["persistence_mean"] for r in gr]
            pers_lo = [r["persistence_ci_lo"] for r in gr]
            pers_hi = [r["persistence_ci_hi"] for r in gr]

            ax.fill_between(xs, model_lo, model_hi, color=COLOR_MODEL, alpha=0.15, linewidth=0)
            ax.fill_between(xs, pers_lo, pers_hi, color=COLOR_PERSISTENCE, alpha=0.15, linewidth=0)
            ax.plot(xs, model_mean, color=COLOR_MODEL, linewidth=2, marker="o", markersize=4,
                    label="Model")
            ax.plot(xs, pers_mean, color=COLOR_PERSISTENCE, linewidth=2, marker="o", markersize=4,
                    label="Persistence")
            _style_axis(ax)
            if i == 0:
                ax.set_title(comp)
            if j == 0:
                ax.set_ylabel(f"{GROUP_TITLES[group]}\n(n={n_group})\n{label} (m/s)")
            if i == len(GROUPS) - 1:
                ax.set_xlabel("horizon (ms)")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8)

    fig.suptitle(f"{ckpt_name} vs. persistence -- {label} by velocity component", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_report(rows, horizon_frames, ckpt_name, n_avail, n_samples, params):
    by_key = {(r["group"], r["horizon_frames"], r["component"], r["metric"]): r for r in rows}
    lines = []
    add = lines.append
    add(f"# Formal documentation: {ckpt_name} vs. persistence")
    add("")
    add(f"- Validation pool: {n_avail} sequences available "
        f"(subset_ratio={PFD_SUBSET_RATIO}); {n_samples} evaluated.")
    for group in GROUPS:
        n_group = by_key[(group, horizon_frames[0], "all", "mae")]["n_sequences"]
        add(f"  - {GROUP_TITLES[group]}: {n_group} sequences.")
    add(f"- Context: 12 frames = 100.0 ms. Forecast: 28 frames = 233.3 ms. "
        f"Window: 40 frames = 333.3 ms.")
    add(f"- Horizons reported: {horizon_frames} frames = "
        f"{[round(f * PFD_CONTEXT_MS / 12, 1) for f in horizon_frames]} ms.")
    add(f"- 95% CI: percentile bootstrap, {PFD_N_BOOTSTRAP} resamples over sequences.")
    add(f"- Metrics: MAE, RMSE, and L2 (mean per-coordinate Euclidean norm of the "
        f"velocity error -- the trainer's own `l2_loss` geometry), all from the "
        f"same rollout.")
    add("")

    for group in GROUPS:
        add(f"## {GROUP_TITLES[group]}")
        add("")
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            add(f"### {label}")
            add("")
            add(f"| horizon (ms) | model {label} (95% CI) | persistence {label} (95% CI) | "
                "mean diff (95% CI) | frac beating persistence |")
            add("|---|---|---|---|---|")
            for hf in horizon_frames:
                r = by_key[(group, hf, "all", metric)]
                add(f"| {r['horizon_ms']:.1f} "
                    f"| {r['model_mean']:.6f} [{r['model_ci_lo']:.6f}, {r['model_ci_hi']:.6f}] "
                    f"| {r['persistence_mean']:.6f} [{r['persistence_ci_lo']:.6f}, {r['persistence_ci_hi']:.6f}] "
                    f"| {r['mean_diff']:+.6f} [{r['diff_ci_lo']:+.6f}, {r['diff_ci_hi']:+.6f}] "
                    f"| {r['frac_beating_persistence']:.2%} |")
            frame1 = by_key[(group, horizon_frames[0], "all", metric)]
            last = by_key[(group, horizon_frames[-1], "all", metric)]
            frame1_pct = (frame1["persistence_mean"] - frame1["model_mean"]) / \
                (frame1["persistence_mean"] + 1e-8) * 100
            last_pct = (last["persistence_mean"] - last["model_mean"]) / \
                (last["persistence_mean"] + 1e-8) * 100
            add("")
            add(f"frame1% = {frame1_pct:+.2f}%   last% = {last_pct:+.2f}%   "
                f"gap (accumulation cost) = {frame1_pct - last_pct:+.2f} pts")
            add("")
            if metric == "l2":
                continue  # no per-component breakdown for L2
            add(f"Per velocity component (mean {label}, model / persistence, at each horizon):")
            add("")
            header = "| component | " + " | ".join(
                f"{hf}f={by_key[(group, hf, 'all', metric)]['horizon_ms']:.0f}ms"
                for hf in horizon_frames) + " |"
            add(header)
            add("|" + "---|" * (len(horizon_frames) + 1))
            for comp in VELOCITY_COMPONENTS:
                cells = []
                for hf in horizon_frames:
                    r = by_key[(group, hf, comp, metric)]
                    cells.append(f"{r['model_mean']:.4f} / {r['persistence_mean']:.4f}")
                add(f"| {comp} | " + " | ".join(cells) + " |")
            add("")

    add("## Figures")
    add("")
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        add(f"- `horizon_summary_{metric}.pdf` -- {label} vs. horizon (with 95% CI) and "
            f"fraction beating persistence, one row per group.")
        if metric != "l2":
            add(f"- `velocity_components_{metric}.pdf` -- {label} vs. horizon per velocity "
                f"component, one row per group, one column per component.")
    add("")
    add(f"params: {params:.2f}M")
    return "\n".join(lines)


def main():
    os.makedirs(PFD_OUT_DIR, exist_ok=True)

    model, device, ckpt_name = load_model()
    ae, ae_path, converter, metric_space = load_autoencoder(device)
    print(f"metric space        : {metric_space}")

    val_h5 = os.path.join(HERE, "data/val_40.h5")
    if not os.path.exists(val_h5):
        raise SystemExit("Validation data not found at " + val_h5)
    dataset = TransformerDataset(val_h5, subset_ratio=PFD_SUBSET_RATIO)
    n_avail = len(dataset)
    n_samples = n_avail if PFD_SAMPLES <= 0 else min(PFD_SAMPLES, n_avail)
    print(f"val data            : {val_h5} ({n_avail} sequences available, "
          f"{n_samples} evaluated, subset_ratio={PFD_SUBSET_RATIO})")

    is_wake = label_wake_or_random(dataset, n_samples)
    print(f"groups              : all={n_samples}  wake={int(is_wake.sum())}  "
          f"random={int((~is_wake).sum())}")

    context_steps = getattr(Config, "VAL_CONTEXT_STEPS", 12)
    num_x = getattr(Config, "NUM_X", 26)
    latent_dim = getattr(Config, "LATENT_DIM", 47)
    full_horizon_frames = getattr(Config, "NUM_TIME", 40) - context_steps  # 28
    dt_ms = PFD_CONTEXT_MS / context_steps

    horizon_frames = sorted(set(PFD_HORIZON_FRAMES))
    bad = [f for f in horizon_frames if f < 1 or f > full_horizon_frames]
    if bad:
        raise SystemExit(f"PFD_HORIZON_FRAMES {bad} outside the available "
                          f"1..{full_horizon_frames} frame horizon.")
    n_frames = max(horizon_frames)
    print(f"horizons            : frames {horizon_frames} = "
          f"{[round(f * dt_ms, 1) for f in horizon_frames]} ms "
          f"(dt={dt_ms:.3f} ms/frame, rolling out to frame {n_frames})")

    metrics = run_rollouts(model, device, ae, converter, dataset, n_samples, num_x, latent_dim,
                            context_steps, n_frames)

    rows = build_table(metrics, is_wake, horizon_frames, dt_ms, PFD_N_BOOTSTRAP, PFD_SEED)

    csv_path = os.path.join(PFD_OUT_DIR, "results_by_horizon.csv")
    write_csv(rows, csv_path)
    print(f"wrote {csv_path}")

    for metric in METRICS:
        summary_path = os.path.join(PFD_OUT_DIR, f"horizon_summary_{metric}.pdf")
        plot_horizon_summary(rows, horizon_frames, ckpt_name, summary_path, metric=metric)
        print(f"wrote {summary_path}")

        if metric == "l2":
            continue  # no per-component breakdown for L2
        components_path = os.path.join(PFD_OUT_DIR, f"velocity_components_{metric}.pdf")
        plot_velocity_components(rows, horizon_frames, ckpt_name, components_path, metric=metric)
        print(f"wrote {components_path}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    report = render_report(rows, horizon_frames, ckpt_name, n_avail, n_samples, params_m)
    report_path = os.path.join(PFD_OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"wrote {report_path}")

    print("\n" + report)


if __name__ == "__main__":
    main()
