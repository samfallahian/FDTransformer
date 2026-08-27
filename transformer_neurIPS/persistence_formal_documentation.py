"""
Formal documentation figures: r1_a3b_delta_ar_best.pt vs. persistence.

This script targets exactly ONE checkpoint -- r1_a3b_delta_ar_best.pt (the
best teacher-forced VAL checkpoint of that run, NOT the _rollout_best twin
tests/test_model_vs_baseline.py defaults to). It produces the
documentation-grade evidence behind a PASS/FAIL claim -- run once, remotely,
on the GPU box (it is not a test and is not run in CI).

WHAT "40-frame setup" MEANS HERE
=================================
Every validation sequence is 40 frames: 12 frames of context, 28 predicted.
At the data's native cadence (12 context frames = 100 ms, i.e. dt = 8.3(3) ms
per frame -- see Documentation/persistence_baseline_design.md for the same
100 ms context convention used elsewhere in this project):

    context : 12 frames =  100.0 ms
    forecast: 68 frames =  566.7 ms   (v2.0: was 28 frames / 233.3 ms in v1.0)
    window  : 80 frames =  666.7 ms   (v2.0: was 40 frames / 333.3 ms in v1.0)

Rather than push the forecast horizon further, this script spends the budget
on evaluation rigor instead:

  - ALL available validation sequences (v2.0: val_80.h5 at the default 0.2
    subset ratio -- see prepare_data.py --num-time 80. v1.0 used val_40.h5 at
    the same subset ratio = 165 of 829 sequences.), not a handful.
  - Two disjoint sub-populations, reported SEPARATELY rather than pooled:
    "wake-targeted" sequences (centered on one of the 24 (y, z) coordinates
    prepare_data.py deliberately samples in the vortex wake) and
    "random-location" sequences (sampled uniformly outside the wake set).
    They are NOT a 50/50 split of this val file -- see the printed group
    sizes -- because random-location sequences are dropped more often for
    landing outside the data's valid domain (see prepare_data.py).
  - Eight horizons (v2.0): frames [1, 6, 12, 24, 36, 48, 60, 68] = [8.3, 50,
    100, 200, 300, 400, 500, 566.7] ms past the context boundary. v1.0 used
    six horizons up to 233.3 ms; the shared 1..28-frame prefix is preserved
    so v1.0/v2.0 numbers are directly comparable at every retained horizon.
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
    PFD_KIND             target checkpoint kind          (default "rollout_best";
                                                          v1.0 default was "best")
    PFD_SUBSET_RATIO     val-set fraction to load        (default 0.2)
    PFD_SAMPLES          sequences to use, 0 = all loaded (default 0)
    PFD_HORIZON_FRAMES   comma list of horizon frames    (default v2.0:
                                                          "1,6,12,24,36,48,60,68")
    PFD_CONTEXT_MS       wall-clock ms the context spans (default 100.0)
    PFD_N_BOOTSTRAP      bootstrap resamples for the CI  (default 10000)
    PFD_BATCH_SIZE       sequences per rollout batch     (default: 32 on cuda, 4 on mps/cpu)
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


# --- console color (ANSI) -----------------------------------------------------
# Separate from the matplotlib palette above: this is terminal styling only,
# never written into report.md/results_by_horizon.csv, which stay plain text.
# Off automatically when stdout isn't a tty (piped to a log file) unless
# PFD_FORCE_COLOR is set; off entirely with PFD_NO_COLOR or the NO_COLOR
# convention (https://no-color.org/).
_COLOR_ON = (
    os.environ.get("PFD_NO_COLOR") is None
    and os.environ.get("NO_COLOR") is None
    and (sys.stdout.isatty() or os.environ.get("PFD_FORCE_COLOR") is not None)
)

_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
    "blue": "\033[94m", "magenta": "\033[95m", "cyan": "\033[96m",
}
_RAINBOW = ["red", "yellow", "green", "cyan", "blue", "magenta"]
# Device -> a fitting single accent color when not rainbow-ing it letter by letter.
_DEVICE_COLOR = {"cuda": "green", "mps": "magenta", "cpu": "yellow"}


def _c(text, color):
    if not _COLOR_ON:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


def _bold(text, color=None):
    if not _COLOR_ON:
        return text
    prefix = _ANSI["bold"] + (_ANSI[color] if color else "")
    return f"{prefix}{text}{_ANSI['reset']}"


def _rainbow(text):
    """Cycle non-whitespace characters through the rainbow palette."""
    if not _COLOR_ON:
        return text
    out, i = [], 0
    for ch in text:
        if ch.strip():
            out.append(f"{_ANSI[_RAINBOW[i % len(_RAINBOW)]]}{ch}")
            i += 1
        else:
            out.append(ch)
    out.append(_ANSI["reset"])
    return "".join(out)


def _banner(title):
    line = "=" * (len(title) + 4)
    print(_c(line, "cyan"))
    print(_bold(f"  {title}", "cyan"))
    print(_c(line, "cyan"))


def _kv(label, value, color=None, width=20):
    val = _c(str(value), color) if color else str(value)
    print(f"{label:<{width}}: {val}")


def _ok(msg):
    print(_c(f"[OK] {msg}", "green"))


def _warn(msg):
    print(_c(f"[!!] {msg}", "yellow"))


def _err(msg):
    print(_bold(f"[XX] {msg}", "red"))


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
# v2.0: point at the rollout-best checkpoint by default (matches
# tests/reports/r1_a3b_delta_ar_deep_dive.md's winner). v1.0 default was "best".
PFD_KIND = os.environ.get("PFD_KIND", "rollout_best")
PFD_SUBSET_RATIO = _env_float("PFD_SUBSET_RATIO", 0.2)
PFD_SAMPLES = _env_int("PFD_SAMPLES", 0)  # 0 = all loaded
# v2.0 horizons cover the 68-frame (566.7 ms) forecast; the first three
# entries (1, 6, 12) are shared with v1.0 for direct A/B comparability.
PFD_HORIZON_FRAMES = [int(x) for x in os.environ.get(
    "PFD_HORIZON_FRAMES", "1,6,12,24,36,48,60,68").split(",") if x.strip()]
PFD_CONTEXT_MS = _env_float("PFD_CONTEXT_MS", 100.0)
PFD_N_BOOTSTRAP = _env_int("PFD_N_BOOTSTRAP", 10000)
# None = device-dependent default, resolved in main() once the device is known
# (see resolve_batch_size): 32 on cuda, 4 on mps/cpu. The autoregressive
# rollout grows its sequence length over `n_frames` internal steps, each a
# different shape, and MPS's caching allocator cannot reuse blocks across
# shapes -- so peak memory during ONE rollout call scales with batch size,
# and 32 sequences is enough to blow past a Mac's MPS ceiling on the very
# first batch (see the OOM this default used to hit: 88 GB allocated before
# a 298 MB request tipped it over -- not "many batches accumulating", one
# batch's internal growth).
PFD_BATCH_SIZE = _env_int("PFD_BATCH_SIZE", 0) or None
PFD_SEED = _env_int("PFD_SEED", 0)
PFD_OUT_DIR = os.environ.get("PFD_OUT_DIR",
                              os.path.join(PROJECT_ROOT, "Documentation", "persistence_formal"))

GROUPS = ("all", "wake", "random")
GROUP_TITLES = {"all": "All sequences", "wake": "Wake-targeted", "random": "Random-location"}


def pick_device():
    """cuda > mps > cpu. Local (Mac) runs otherwise silently fall back to CPU
    even with Apple-silicon GPU available, since torch.cuda.is_available() is
    always False there."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_batch_size(device):
    """PFD_BATCH_SIZE overrides; otherwise 32 on cuda, 4 on mps/cpu.

    mps/cpu default is small on purpose: the autoregressive rollout's
    sequence length grows every internal step, and MPS's caching allocator
    cannot reuse a block once a new step needs a different shape, so peak
    memory during a SINGLE rollout call scales with batch size. 32 was
    enough to hit an OOM on the first batch on a Mac.
    """
    if PFD_BATCH_SIZE is not None:
        return PFD_BATCH_SIZE
    return 32 if device == "cuda" else 4


def clear_device_cache(device):
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def device_memory_str(device):
    """Best-effort current/peak allocation, for diagnosing OOMs like this one."""
    if device == "mps":
        cur = torch.mps.current_allocated_memory() / (1024 ** 3)
        drv = torch.mps.driver_allocated_memory() / (1024 ** 3)
        return f"mps allocated={cur:.2f} GiB, driver={drv:.2f} GiB"
    if device == "cuda":
        cur = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return f"cuda allocated={cur:.2f} GiB, peak={peak:.2f} GiB"
    return None


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
    _banner("Loading checkpoint")
    ckpt_name = f"{PFD_RUN}_{PFD_KIND}"
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
    if not os.path.exists(ckpt_path):
        _err(f"{ckpt_path} not found.")
        raise SystemExit(
            f"{ckpt_path} not found. This script targets one specific "
            f"checkpoint on the remote GPU box; set PFD_RUN/PFD_KIND to "
            f"point elsewhere if you meant a different one.")

    device = pick_device()
    _kv("device", _rainbow(device.upper()))
    _kv("checkpoint", ckpt_path)

    ckpt = load_checkpoint(ckpt_path)
    apply_checkpoint_config(ckpt)
    model = get_model(Config)
    state_dict, stripped = normalize_state_dict(ckpt["model_state_dict"])
    if stripped:
        _warn(f"stripped state-dict prefix(es): {stripped}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if k not in BENIGN_MISSING_KEYS]
    if missing:
        _err(f"{ckpt_name}: {len(missing)} missing key(s): {missing[:4]}")
        raise SystemExit(f"{ckpt_name}: {len(missing)} missing key(s), cannot be "
                          f"scored: {missing[:4]}")
    model.eval()
    model.to(device)

    probe = probe_causality(model, Config, device)
    if probe["causal"]:
        _ok(f"causal (past outputs moved {probe['max_change_before_cut']:.3e}, "
            f"tolerance {probe['tolerance']:.3e})")
    else:
        _err(f"CAUSALITY LEAK: past outputs moved {probe['max_change_before_cut']:.3e} "
             f"(tolerance {probe['tolerance']:.3e})")
        raise SystemExit("Architecture leaks the future; every downstream metric "
                          "would be meaningless. Fix the leak before documenting it.")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    _kv("architecture", f"variant={getattr(Config, 'VARIANT', '?')} "
        f"embed={getattr(Config, 'EMBED_SIZE', '?')} layers={getattr(Config, 'N_LAYERS', '?')} "
        f"params={params_m:.2f}M epoch={ckpt.get('epoch')}", color="cyan")
    return model, device, ckpt_name


def run_rollouts(model, device, ae, converter, data, n_samples, num_x, latent_dim,
                  context_steps, n_frames, batch_size):
    """Full-horizon rollout for every sequence, batched for speed.

    `data` must already be resident on `device` (see main(): the whole
    validation subset is a few tens of MB, so it is moved once up front
    rather than per-batch here -- repeated small host<->device transfers,
    and worse, a `.cpu().numpy()` sync inside a per-frame/per-component
    Python loop, were making this look "IO bound" the way disk reads used to
    during training, except the stall was PCIe/Metal round-trips, not disk).

    `batch_size` matters more than it looks: `rollout_frames` grows its
    sequence length one step at a time internally (n_frames sequential
    forward passes), and MPS's caching allocator cannot reuse a block once
    the next step needs a different shape -- so peak memory during a SINGLE
    batch's rollout call scales with batch_size. See resolve_batch_size().

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

    _kv("batch size", batch_size, color="cyan")
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = data[start:end]  # already on `device` -- no per-batch transfer

            targets_latent = batch[:, context_len:context_len + n_frames * num_x, :latent_dim]
            last_frame = batch[:, context_len - num_x:context_len, :latent_dim]
            persistence_latent = last_frame.repeat(1, n_frames, 1)

            out = rollout_frames(model, batch, Config, ctx_frames=context_steps, n_frames=n_frames)
            B, n, nx, ld = out.shape
            preds_latent = out.reshape(B, n * nx, ld)[:, :n_frames * num_x, :]

            targets_v = decode_latents_to_centroid(targets_latent, ae, converter, device)
            persistence_v = decode_latents_to_centroid(persistence_latent, ae, converter, device)
            preds_v = decode_latents_to_centroid(preds_latent, ae, converter, device)

            # Vectorized over ALL frames at once (B, n_frames, num_x, 3) --
            # no Python loop over frames/components, so the GPU/MPS command
            # queue runs the whole batch before the single sync at the end
            # (the .cpu().numpy() calls below), instead of stalling on one
            # every frame x metric x component.
            preds_r = preds_v.reshape(B, n_frames, num_x, 3)
            targets_r = targets_v.reshape(B, n_frames, num_x, 3)
            pers_r = persistence_v.reshape(B, n_frames, num_x, 3)

            diff_m = preds_r - targets_r      # (B, n_frames, num_x, 3)
            diff_p = pers_r - targets_r

            mae_m = diff_m.abs().mean(dim=(2, 3))                       # (B, n_frames)
            mae_p = diff_p.abs().mean(dim=(2, 3))
            rmse_m = torch.sqrt((diff_m ** 2).mean(dim=(2, 3)))
            rmse_p = torch.sqrt((diff_p ** 2).mean(dim=(2, 3)))
            # L2: per-coordinate Euclidean norm of the (vx, vy, vz) error,
            # averaged over the 26 coords -- the trainer's own
            # `l2_loss = mean(norm(diff, dim=-1))` geometry.
            l2_m = torch.linalg.norm(diff_m, dim=3).mean(dim=2)         # (B, n_frames)
            l2_p = torch.linalg.norm(diff_p, dim=3).mean(dim=2)

            mae_c_m = diff_m.abs().mean(dim=2)                          # (B, n_frames, 3)
            mae_c_p = diff_p.abs().mean(dim=2)
            rmse_c_m = torch.sqrt((diff_m ** 2).mean(dim=2))
            rmse_c_p = torch.sqrt((diff_p ** 2).mean(dim=2))

            metrics["mae"]["model"][start:end] = mae_m.cpu().numpy()
            metrics["mae"]["pers"][start:end] = mae_p.cpu().numpy()
            metrics["rmse"]["model"][start:end] = rmse_m.cpu().numpy()
            metrics["rmse"]["pers"][start:end] = rmse_p.cpu().numpy()
            metrics["l2"]["model"][start:end] = l2_m.cpu().numpy()
            metrics["l2"]["pers"][start:end] = l2_p.cpu().numpy()
            metrics["mae"]["model_c"][start:end] = mae_c_m.cpu().numpy()
            metrics["mae"]["pers_c"][start:end] = mae_c_p.cpu().numpy()
            metrics["rmse"]["model_c"][start:end] = rmse_c_m.cpu().numpy()
            metrics["rmse"]["pers_c"][start:end] = rmse_c_p.cpu().numpy()

            # Release the batch's intermediate tensors back to the device
            # allocator's free pool before the next (differently-shaped)
            # batch starts -- otherwise MPS in particular tends to keep
            # growing rather than reusing freed blocks across shapes.
            del out, preds_latent, targets_v, persistence_v, preds_v
            del preds_r, targets_r, pers_r, diff_m, diff_p
            clear_device_cache(device)

            pct = end / n_samples * 100
            elapsed = time.perf_counter() - t0
            mem = device_memory_str(device)
            mem_suffix = f"  [{mem}]" if mem else ""
            print(_c(f"  [{pct:5.1f}%] rolled out sequences [{start}:{end}) of {n_samples} "
                      f"({elapsed:.1f}s elapsed){mem_suffix}", "dim"))

    total = time.perf_counter() - t0
    _ok(f"rollout complete: {total:.1f}s for {n_samples} sequences "
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
    add(f"- Context: 12 frames = 100.0 ms. Forecast: 68 frames = 566.7 ms. "
        f"Window: 80 frames = 666.7 ms. (v2.0 -- v1.0 forecast/window was 28/40 frames.)")
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


def _print_console_summary(rows, horizon_frames):
    """Console-only colored headline: frame1%/last% per group, per metric.
    Never written to report.md -- that file stays plain text."""
    by_key = {(r["group"], r["horizon_frames"], r["component"], r["metric"]): r for r in rows}
    _banner("Headline: model vs. persistence")
    for group in GROUPS:
        print(_bold(f"{GROUP_TITLES[group]}", "cyan"))
        for metric in METRICS:
            frame1 = by_key[(group, horizon_frames[0], "all", metric)]
            last = by_key[(group, horizon_frames[-1], "all", metric)]
            frame1_pct = (frame1["persistence_mean"] - frame1["model_mean"]) / \
                (frame1["persistence_mean"] + 1e-8) * 100
            last_pct = (last["persistence_mean"] - last["model_mean"]) / \
                (last["persistence_mean"] + 1e-8) * 100
            f1c = "green" if frame1_pct > 0 else "red"
            lc = "green" if last_pct > 0 else "red"
            print(f"  {METRIC_LABELS[metric]:<22} frame1% = {_c(f'{frame1_pct:+.2f}%', f1c):<20} "
                  f"last% = {_c(f'{last_pct:+.2f}%', lc)}")
    print()


def main():
    os.makedirs(PFD_OUT_DIR, exist_ok=True)

    model, device, ckpt_name = load_model()

    _banner("Loading autoencoder + validation data")
    ae, ae_path, converter, metric_space = load_autoencoder(device)
    _kv("metric space", metric_space, color="cyan")

    # v2.0 AE smoke assertion (plan Step 3b): before spending any rollout time,
    # confirm the scripted GEN3 decoder round-trips a random (1, 47) latent to
    # a (1, 375) reconstruction. If this ever fails (wrong scripted module, a
    # future rescript that drops `.decode`, dtype/device mismatch), we bail out
    # here rather than emitting nonsense metrics later.
    with torch.no_grad():
        _probe_z = torch.randn(1, 47, device=device)
        _probe_recon = ae.decode(_probe_z) if hasattr(ae, "decode") else ae(_probe_z)
    if not (hasattr(_probe_recon, "shape")
            and tuple(_probe_recon.shape) == (1, 375)):
        raise SystemExit(
            f"[pfd] AE round-trip smoke FAILED: (1,47) -> "
            f"{tuple(_probe_recon.shape) if hasattr(_probe_recon, 'shape') else type(_probe_recon)}, "
            f"expected (1, 375). Refusing to run rollouts.")
    _ok("AE round-trip smoke passed: (1, 47) -> (1, 375)")

    # v2.0: point at the 80-frame val file. If it is missing (e.g. someone is
    # regenerating v1.0 numbers on a checkout that never ran prepare_data.py
    # --num-time 80), fall back to val_40.h5 with a very loud note so the
    # horizon-vs-file mismatch is impossible to miss.
    val_h5 = os.path.join(HERE, "data/val_80.h5")
    if not os.path.exists(val_h5):
        legacy = os.path.join(HERE, "data/val_40.h5")
        if os.path.exists(legacy):
            print(_c(f"[pfd] WARNING: {val_h5} missing; falling back to {legacy}. "
                     f"Horizons > 28 will be rejected against the 40-frame window.", "yellow"))
            val_h5 = legacy
        else:
            raise SystemExit("Validation data not found at " + val_h5)
    dataset = TransformerDataset(val_h5, subset_ratio=PFD_SUBSET_RATIO)
    n_avail = len(dataset)
    n_samples = n_avail if PFD_SAMPLES <= 0 else min(PFD_SAMPLES, n_avail)
    _kv("val data", f"{val_h5} ({n_avail} available, {n_samples} evaluated, "
        f"subset_ratio={PFD_SUBSET_RATIO})")

    is_wake = label_wake_or_random(dataset, n_samples)
    n_wake, n_random = int(is_wake.sum()), int((~is_wake).sum())
    _kv("groups", f"all={_c(str(n_samples), 'cyan')}  wake={_c(str(n_wake), 'magenta')}  "
        f"random={_c(str(n_random), 'yellow')}")

    context_steps = getattr(Config, "VAL_CONTEXT_STEPS", 12)
    num_x = getattr(Config, "NUM_X", 26)
    latent_dim = getattr(Config, "LATENT_DIM", 47)
    # v2.0: NUM_TIME=80 -> full_horizon_frames=68. v1.0 was NUM_TIME=40 -> 28.
    full_horizon_frames = getattr(Config, "NUM_TIME", 80) - context_steps
    dt_ms = PFD_CONTEXT_MS / context_steps

    horizon_frames = sorted(set(PFD_HORIZON_FRAMES))
    bad = [f for f in horizon_frames if f < 1 or f > full_horizon_frames]
    if bad:
        raise SystemExit(f"PFD_HORIZON_FRAMES {bad} outside the available "
                          f"1..{full_horizon_frames} frame horizon.")
    n_frames = max(horizon_frames)
    _kv("horizons", f"frames {horizon_frames} = {[round(f * dt_ms, 1) for f in horizon_frames]} ms "
        f"(dt={dt_ms:.3f} ms/frame, rolling out to frame {n_frames})")

    # The whole evaluated subset is a few tens of MB (n_samples x 40 x 26 x 52
    # float32) -- easily resident on the GPU/MPS device at once. Move it there
    # ONCE here rather than per-batch inside run_rollouts: repeated small
    # host<->device transfers (plus, previously, a sync inside a per-frame
    # Python loop) were the "IO bound" symptom, just PCIe/Metal round-trips
    # instead of disk reads.
    data_mb = dataset.data[:n_samples].numel() * 4 / (1024 * 1024)
    t_xfer0 = time.perf_counter()
    data_on_device = dataset.data[:n_samples].to(device)
    _ok(f"moved {data_mb:.1f} MB to {device} in {time.perf_counter() - t_xfer0:.2f}s "
        f"(resident for the whole rollout, no further host<->device copies)")

    batch_size = resolve_batch_size(device)

    _banner("Running rollouts")
    metrics = run_rollouts(model, device, ae, converter, data_on_device, n_samples, num_x,
                            latent_dim, context_steps, n_frames, batch_size)

    _banner("Building tidy table (bootstrap CIs)")
    rows = build_table(metrics, is_wake, horizon_frames, dt_ms, PFD_N_BOOTSTRAP, PFD_SEED)
    _ok(f"{len(rows)} rows across {len(GROUPS)} groups x {len(horizon_frames)} horizons x "
        f"{len(METRICS)} metrics")

    _banner("Writing outputs")
    csv_path = os.path.join(PFD_OUT_DIR, "results_by_horizon.csv")
    write_csv(rows, csv_path)
    _ok(f"wrote {csv_path}")

    for metric in METRICS:
        summary_path = os.path.join(PFD_OUT_DIR, f"horizon_summary_{metric}.pdf")
        plot_horizon_summary(rows, horizon_frames, ckpt_name, summary_path, metric=metric)
        _ok(f"wrote {summary_path}")

        if metric == "l2":
            continue  # no per-component breakdown for L2
        components_path = os.path.join(PFD_OUT_DIR, f"velocity_components_{metric}.pdf")
        plot_velocity_components(rows, horizon_frames, ckpt_name, components_path, metric=metric)
        _ok(f"wrote {components_path}")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    report = render_report(rows, horizon_frames, ckpt_name, n_avail, n_samples, params_m)
    report_path = os.path.join(PFD_OUT_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    _ok(f"wrote {report_path}")

    _print_console_summary(rows, horizon_frames)
    print(report)


if __name__ == "__main__":
    main()
