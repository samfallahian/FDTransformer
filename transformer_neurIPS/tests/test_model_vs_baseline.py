"""
Leaderboard test: every transformer checkpoint vs. the persistence baseline.

Discovers ALL checkpoints in transformer_neurIPS/saved_models/, evaluates each
one against the same validation samples and the same persistence baseline, and
prints a ranked report at the end (also written to tests/reports/).

Checkpoints written per run, each with an optional `_scripted.pt` twin that is
skipped here:

    <run>_best.pt          <- best teacher-forced VAL metric
    <run>_rollout_best.pt  <- best ROLLOUT MSE  (the one that matters)
    <run>_latest.pt        <- periodic snapshot
    <run>_train_best.pt    <- best TRAIN L2, written by the older trainer only

Every checkpoint carries its own `config` dict, so each is rebuilt with the
architecture it was trained as (VARIANT, TOKENIZATION, EMBED_SIZE, N_LAYERS,
PREDICT_DELTA, ...) rather than with whatever Config happens to hold.

Tunable via environment variables (all optional):
    TX_MODELS           glob for checkpoint names      (default "r1_a3b_delta_ar_rollout_best.pt";
                                                         set to "*.pt" for the full sweep)
    TX_SINGLE_SAMPLES   samples for single-step test   (default 5)
    TX_ROLLOUT_SAMPLES  samples for rollout test       (default 3)
    TX_ROLLOUT_STEPS    rollout horizon in tokens      (default 104; 0 = full)
    TX_COORDS           coords for single-step test    (default "25,24,23")
    TX_SUBSET_RATIO     val-set fraction to load       (default 0.1)
    TX_SKIP_LATEST      1 to skip *_latest.pt          (default 0)
    TX_MAX_CKPTS        max checkpoints before failing (default 64)

NOTE ON RE-EVALUATED OLD CHECKPOINTS
====================================
model_variants.py used to leak the future in two places (the
`nn.MultiheadAttention` + `is_causal` hint call, and ConvBlock's symmetrically
padded Conv1d). Both are fixed, and `CausalSelfAttention` was written to be
parameter-name compatible with the module it replaced, so pre-fix checkpoints
still load -- into the FIXED, non-leaky architecture.

Consequence: a checkpoint's recorded `val_l2` can be much BETTER than the value
this test measures for it now. That delta is not a bug in either place; it is the
size of the leak the checkpoint was trained with. `test_checkpoints_are_causal`
below asserts that what we evaluate today is causal, whatever it was trained as.
"""

import os
import sys
import glob
import time
import fnmatch
import unittest
import importlib.util

import torch
import numpy as np
import h5py

# Add project root to path to allow imports
current_file_path = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(current_file_path)                       # .../transformer_neurIPS/tests
PKG_ROOT = os.path.dirname(TEST_DIR)                                # .../transformer_neurIPS
PROJECT_ROOT = os.path.dirname(PKG_ROOT)                            # .../cgan
project_root = PROJECT_ROOT                                         # kept for back-compat
print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: abspath(__file__) = {current_file_path}")
print(f"DEBUG: project_root = {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Inserted {project_root} to sys.path")

print(f"DEBUG: sys.path = {sys.path}")

# Config/dataset/rollout come from the deep-dive trainer: it is the one that
# carries the new fields (TOKENIZATION, ATTN_IMPL, PREDICT_DELTA, ...) and the
# shared `rollout_frames` helper, so token-level and frame-level checkpoints can
# be scored by the same code path.
from transformer_neurIPS.train_production_transformer_deep_dive import (
    Config, TransformerDataset, mse_loss, l2_loss, rollout_frames, probe_causality)
from transformer_neurIPS.model_variants import get_model
from encoder_neurIPS.models import create_model_variant


# --- FloatConverter resolution ------------------------------------------------
# TransformLatent.py lives at the repo ROOT and is only tracked on the LFM /
# publication_2025 branches.  Remote boxes that were cloned from `main`, or that
# only rsync the *_neurIPS packages, end up with the packages present but the
# root-level module missing -> "ModuleNotFoundError: No module named
# 'TransformLatent'" even though project_root is correctly on sys.path.
#
# Resolve in order: normal import -> explicit load by file path -> inline copy.
class _FallbackFloatConverter:
    """Mirror of TransformLatent.FloatConverter (numpy-only, no pandas)."""
    def __init__(self):
        self.min_value = -0.197745
        self.max_value = 0.263599
        self.scale = 1.0 / (self.max_value - self.min_value)
        self.shift = -self.min_value * self.scale

    def convert(self, value):
        return value * self.scale + self.shift

    def unconvert(self, value):
        return (value - self.shift) / self.scale


def _resolve_float_converter():
    try:
        from TransformLatent import FloatConverter as _FC
        print(f"DEBUG: FloatConverter imported normally from {sys.modules['TransformLatent'].__file__}")
        return _FC
    except ImportError as e:
        print(f"DEBUG: plain `from TransformLatent import FloatConverter` failed: {e}")

    candidates = [
        os.path.join(PROJECT_ROOT, "TransformLatent.py"),
        os.path.join(PKG_ROOT, "TransformLatent.py"),
        os.path.join(TEST_DIR, "TransformLatent.py"),
        os.path.join(os.getcwd(), "TransformLatent.py"),
    ]
    for path in candidates:
        print(f"DEBUG:   candidate {path} -> exists={os.path.exists(path)}")
        if not os.path.exists(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location("TransformLatent", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["TransformLatent"] = mod
            spec.loader.exec_module(mod)
            print(f"DEBUG: FloatConverter loaded by file path from {path}")
            return mod.FloatConverter
        except Exception as e:  # e.g. pandas missing on a slim remote box
            print(f"DEBUG:   failed to exec {path}: {type(e).__name__}: {e}")

    try:
        root_py = sorted(f for f in os.listdir(PROJECT_ROOT) if f.endswith(".py"))
    except OSError as e:
        root_py = [f"<unreadable: {e}>"]
    print(f"DEBUG: cwd = {os.getcwd()}")
    print(f"DEBUG: *.py at project_root ({PROJECT_ROOT}): {root_py}")
    print("WARNING: TransformLatent.py not found on this machine -- using the "
          "inline _FallbackFloatConverter (constants copied from the repo-root "
          "module). Sync TransformLatent.py to the box to silence this.")
    return _FallbackFloatConverter


FloatConverter = _resolve_float_converter()


# --- Run configuration --------------------------------------------------------
def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not an int; using default {default}")
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a float; using default {default}")
        return default


# For now, scoped to the one checkpoint under active investigation
# (r1_a3b_delta_ar_rollout_best.pt). Set TX_MODELS="*.pt" to run the full
# multi-checkpoint sweep again.
CKPT_GLOB = os.environ.get("TX_MODELS", "r1_a3b_delta_ar_rollout_best.pt")
SINGLE_SAMPLES = _env_int("TX_SINGLE_SAMPLES", 5)
ROLLOUT_SAMPLES = _env_int("TX_ROLLOUT_SAMPLES", 3)
# The trainer's Config.VAL_ROLLOUT_STEPS is 26*(40-12) = 728 tokens. That is 728
# SEQUENTIAL forward passes per sample on an ever-growing sequence -- fine for one
# model, far too slow for a sweep over a dozen checkpoints. Cap it by default and
# print the horizon in the report so the numbers are never ambiguous.
ROLLOUT_STEPS_CAP = _env_int("TX_ROLLOUT_STEPS", 104)  # 104 tokens = 4 time steps
COORDS = [int(c) for c in os.environ.get("TX_COORDS", "25,24,23").split(",") if c.strip()]
SUBSET_RATIO = _env_float("TX_SUBSET_RATIO", 0.1)
SKIP_LATEST = _env_int("TX_SKIP_LATEST", 0) == 1
# A 5-arm sweep round writes up to 3 checkpoints per arm, so the old hard-coded
# "< 20" ceiling failed on a perfectly healthy sweep. Tunable, with a ceiling
# high enough for several rounds.
MAX_CKPTS = _env_int("TX_MAX_CKPTS", 64)

CHECKPOINT_DIR = os.path.join(PKG_ROOT, "saved_models")
REPORT_DIR = os.path.join(TEST_DIR, "reports")

# Longest-first so "_rollout_best"/"_train_best" win over the "_best" substring.
KIND_SUFFIXES = ("_rollout_best", "_train_best", "_latest", "_best")

# State-dict keys whose absence is expected rather than a broken checkpoint:
#   causal_mask          legacy no-op buffer, now non-persistent
#   feat_mean/feat_std   NORMALIZE_FEATURES statistics, added after the older
#                        checkpoints were written. Absent means the model runs
#                        with mean 0 / std 1, i.e. exactly the un-normalised
#                        behaviour it was trained with -- correct, not degraded.
# Anything else missing means randomly-initialised layers, and a metric computed
# from those is noise dressed up as a result.
BENIGN_MISSING_KEYS = frozenset({"causal_mask", "feat_mean", "feat_std"})


def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def load_autoencoder(device):
    """Locate + load the AE used to decode latents -> centroid velocities.

    Returns (ae, ae_path, converter, metric_space). `ae` is None (and
    metric_space says so) if no checkpoint is found on any of the known
    round-naming layouts; callers then fall back to raw latent dims 0:3.
    """
    ae_search_paths = [
        "encoder_neurIPS/saved_models/round_production/model_04_best.pt",
        "encoder_neurIPS/saved_models/round_4/model_04_best.pt",
        "encoder_neurIPS/saved_models/simultaneous_training/model_04_best.pt",
        "encoder_neurIPS/saved_models/model_04_best.pt",
    ]
    ae_path = None
    for rel_path in ae_search_paths:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        print(f"DEBUG: AE candidate {full_path} -> exists={os.path.exists(full_path)}")
        if os.path.exists(full_path):
            ae_path = full_path
            break

    converter = FloatConverter()
    if ae_path:
        print(f"Loading AE from: {ae_path}")
        ae = create_model_variant(4)
        ae_ckpt = load_checkpoint(ae_path)
        ae.load_state_dict(ae_ckpt["model_state_dict"])
        ae.eval()
        ae.to(device)
        metric_space = "centroid velocity (m/s, AE-decoded)"
    else:
        print("WARNING: AE model not found. Centroid decoding falls back to "
              "raw latent dims 0:3 -- numbers are still comparable across "
              "models, but they are NOT physical velocities.")
        ae = None
        metric_space = "raw latent dims 0:3 (AE MISSING)"
    return ae, ae_path, converter, metric_space


def decode_latents_to_centroid(latents, ae, converter, device):
    """Decode (..., 47) latents to (..., 3) centroid velocities [vx, vy, vz]."""
    if ae is None:
        return latents[..., :3]

    with torch.no_grad():
        orig_shape = latents.shape
        lat_flat = latents.reshape(-1, orig_shape[-1])
        recon = ae.decode(lat_flat)                           # (N, 375)
        # Centroid sits at neighbor index 62 -> columns 186:189
        centroid_v = recon[:, 186:189].cpu().numpy()
        centroid_v = converter.unconvert(centroid_v)          # back to physical units
        new_shape = list(orig_shape[:-1]) + [3]
        return torch.from_numpy(centroid_v).reshape(new_shape).to(device)


def split_run_kind(stem):
    """'production_conv_E256_L6_rollout_best' -> ('production_conv_E256_L6', 'rollout_best')."""
    for suffix in KIND_SUFFIXES:
        if stem.endswith(suffix):
            return stem[:-len(suffix)], suffix[1:]
    return stem, "unknown"


def discover_checkpoints():
    """All non-scripted .pt files in saved_models/, sorted by (run, kind)."""
    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"DEBUG: checkpoint dir does not exist: {CHECKPOINT_DIR}")
        return []

    found = []
    for path in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))):
        name = os.path.basename(path)
        if name.endswith("_scripted.pt"):
            continue  # TorchScript twin of a checkpoint we already have
        if not fnmatch.fnmatch(name, CKPT_GLOB):
            continue
        run, kind = split_run_kind(name[:-3])
        if SKIP_LATEST and kind == "latest":
            continue
        found.append({"path": path, "name": name[:-3], "run": run, "kind": kind})

    kind_order = {"rollout_best": 0, "best": 1, "train_best": 2, "latest": 3, "unknown": 4}
    found.sort(key=lambda c: (c["run"], kind_order.get(c["kind"], 9)))
    return found


def load_checkpoint(path):
    """torch.load that survives the weights_only=True default of torch >= 2.6.

    Checkpoints embed a plain-python `config` dict, which the restricted
    unpickler refuses, so fall back to the unrestricted loader for our own files.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state_dict):
    """Strip torch.compile / DDP wrapper prefixes from state-dict keys."""
    stripped = []
    for prefix in ("_orig_mod.", "module."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            stripped.append(prefix)
    return state_dict, stripped


# Snapshot of Config as imported, so each checkpoint's config can be applied and
# then rolled back. Necessary because get_model() itself mutates config
# (VARIANT='swiglu' sets config.USE_SWIGLU=True) and checkpoints differ in
# EMBED_SIZE / N_LAYERS / VARIANT -- without a reset, model N inherits leftovers
# from model N-1.
CONFIG_SNAPSHOT = {
    k: getattr(Config, k)
    for k in dir(Config)
    if not k.startswith("_") and not callable(getattr(Config, k))
}


def reset_config():
    for k, v in CONFIG_SNAPSHOT.items():
        setattr(Config, k, v)
    for k in [k for k in dir(Config) if not k.startswith("_") and k not in CONFIG_SNAPSHOT]:
        if not callable(getattr(Config, k, None)):
            try:
                delattr(Config, k)
            except AttributeError:
                pass


def apply_checkpoint_config(ckpt):
    reset_config()
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        return False
    for k, v in cfg.items():
        # Paths inside the checkpoint point at whatever box it was trained on.
        # We resolve data/checkpoint locations ourselves from __file__.
        if k in ("TRAIN_H5", "VAL_H5", "CHECKPOINT_DIR", "DEVICE"):
            continue
        setattr(Config, k, v)
    return True


def fmt(value, spec=".6f", na="  --  "):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return na
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


class TestModelVsBaseline(unittest.TestCase):
    results = []          # one dict per checkpoint
    header_lines = []     # run settings, reprinted in the report

    # ------------------------------------------------------------------ setup
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nDEBUG: checkpoint dir = {CHECKPOINT_DIR}")

        cls.checkpoints = discover_checkpoints()
        if not cls.checkpoints:
            try:
                present = sorted(os.listdir(CHECKPOINT_DIR))
            except OSError as e:
                present = [f"<unreadable: {e}>"]
            print(f"DEBUG: directory contents: {present}")
            raise unittest.SkipTest(
                f"No checkpoints matching {CKPT_GLOB!r} in {CHECKPOINT_DIR}")

        print(f"DEBUG: {len(cls.checkpoints)} checkpoint(s) to evaluate:")
        for c in cls.checkpoints:
            size_mb = os.path.getsize(c["path"]) / (1024 * 1024)
            print(f"DEBUG:   [{c['kind']:>12}] {c['name']}  ({size_mb:.1f} MB)")

        # --- Autoencoder used to decode latents -> centroid velocities --------
        # NOTE: encoder_neurIPS is a SIBLING of transformer_neurIPS, so these
        # must be joined against PROJECT_ROOT. Joining against PKG_ROOT produced
        # .../transformer_neurIPS/encoder_neurIPS/... which never matched, and
        # the centroid decode silently degraded to latents[:, :3].
        cls.ae, cls.ae_path, cls.converter, cls.metric_space = load_autoencoder(cls.device)

        # --- Validation data --------------------------------------------------
        val_h5 = os.path.join(PKG_ROOT, "data/val_40.h5")
        if not os.path.exists(val_h5):
            raise unittest.SkipTest("Validation data not found at " + val_h5)
        cls.dataset = TransformerDataset(val_h5, subset_ratio=SUBSET_RATIO)
        n_avail = len(cls.dataset)
        cls.n_single = min(SINGLE_SAMPLES, n_avail)
        cls.n_rollout = min(ROLLOUT_SAMPLES, n_avail)
        # Fixed, unshuffled samples so every checkpoint is scored on identical data.
        cls.samples = [cls.dataset[i].unsqueeze(0)
                       for i in range(max(cls.n_single, cls.n_rollout))]

        cls._baseline_cache = {}

        cls.header_lines = [
            f"device              : {cls.device}",
            f"checkpoint dir      : {CHECKPOINT_DIR}",
            f"checkpoints matched : {len(cls.checkpoints)} (glob {CKPT_GLOB!r}, *_scripted.pt skipped)",
            f"val data            : {val_h5} (subset_ratio={SUBSET_RATIO}, {n_avail} sequences)",
            f"metric space        : {cls.metric_space}",
            f"single-step samples : {cls.n_single} x coords {COORDS}",
            f"rollout samples     : {cls.n_rollout}",
        ]
        print("\n" + "\n".join(cls.header_lines))

        # --- Evaluate every checkpoint ---------------------------------------
        cls.results = []
        for idx, c in enumerate(cls.checkpoints, start=1):
            print(f"\n{'=' * 78}\n[{idx}/{len(cls.checkpoints)}] {c['name']}\n{'=' * 78}")
            t0 = time.perf_counter()
            try:
                row = cls._evaluate_checkpoint(c)
            except Exception as e:  # never let one bad checkpoint kill the sweep
                import traceback
                traceback.print_exc()
                row = dict(c, status=f"ERROR: {type(e).__name__}: {e}")
            finally:
                reset_config()
                if cls.device == "cuda":
                    torch.cuda.empty_cache()
            row["eval_seconds"] = time.perf_counter() - t0
            
            # Determine PASS/FAIL result based on whether evaluation is ok and beats the baseline
            is_ok = row.get("status") == "ok"
            rollout_imp = row.get("rollout_improvement_pct")
            single_imp = row.get("single_improvement_pct")
            has_rollout_imp = rollout_imp is not None and rollout_imp > 0.0
            has_single_imp = single_imp is not None and single_imp > 0.0
            
            if is_ok and (has_rollout_imp or has_single_imp):
                row["result"] = "PASS"
            else:
                row["result"] = "FAIL"
                
            cls.results.append(row)
            print(f"  -> {row.get('status')} [{row['result']}] in {row['eval_seconds']:.1f}s")

    @classmethod
    def tearDownClass(cls):
        if cls.results:
            cls._print_report()

    # -------------------------------------------------------------- evaluation
    @classmethod
    def _evaluate_checkpoint(cls, c):
        row = dict(c, status="ok")
        ckpt = load_checkpoint(c["path"])

        row["epoch"] = ckpt.get("epoch")
        row["ckpt_train_l2"] = ckpt.get("train_l2")
        row["ckpt_val_l2"] = ckpt.get("val_l2")
        row["ckpt_rollout_mse"] = ckpt.get("rollout_mse")

        had_config = apply_checkpoint_config(ckpt)
        row["variant"] = getattr(Config, "VARIANT", "?")
        row["embed"] = getattr(Config, "EMBED_SIZE", "?")
        row["layers"] = getattr(Config, "N_LAYERS", "?")
        if not had_config:
            row["status"] = "no-config-in-ckpt (used defaults)"
        print(f"  arch: variant={row['variant']} embed={row['embed']} "
              f"layers={row['layers']} epoch={row['epoch']} "
              f"(config in ckpt: {had_config})")

        model = get_model(Config)
        state_dict, stripped = normalize_state_dict(ckpt["model_state_dict"])
        if stripped:
            print(f"  stripped state-dict prefix(es): {stripped}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if k not in BENIGN_MISSING_KEYS]
        unexpected = list(incompatible.unexpected_keys)
        row["missing_keys"] = len(missing)
        row["unexpected_keys"] = len(unexpected)
        if missing:
            # Missing weights mean randomly-initialised layers; any metric
            # computed from this model would be noise dressed up as a result.
            print(f"  MISSING {len(missing)} key(s), e.g. {missing[:4]}")
            row["status"] = f"INCOMPATIBLE ({len(missing)} missing keys)"
            return row
        if unexpected:
            print(f"  {len(unexpected)} unexpected key(s) ignored, e.g. {unexpected[:4]}")

        model.eval()
        model.to(cls.device)
        row["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
        row["tokenization"] = getattr(Config, "TOKENIZATION", "token")

        # Prove the architecture we are about to score cannot see the future.
        # Cheap (one forward pair on random data) and it removes the single
        # failure mode that would make every number below meaningless.
        probe = probe_causality(model, Config, cls.device)
        row["causal"] = probe["causal"]
        row["causal_leak"] = probe["max_change_before_cut"]
        print(f"  causality: {'OK' if probe['causal'] else 'LEAK'} "
              f"(past outputs moved {probe['max_change_before_cut']:.3e} when the "
              f"future was perturbed; tolerance {probe['tolerance']:.3e})")

        with torch.no_grad():
            cls._score_single_step(model, row)
            cls._score_rollout(model, row)

        del model
        return row

    @classmethod
    def _geometry(cls):
        context_steps = getattr(Config, "VAL_CONTEXT_STEPS", 12)
        num_x = getattr(Config, "NUM_X", 26)
        latent_dim = getattr(Config, "LATENT_DIM", 47)
        full_rollout = getattr(Config, "VAL_ROLLOUT_STEPS", 28)
        rollout_steps = full_rollout if ROLLOUT_STEPS_CAP <= 0 else min(full_rollout, ROLLOUT_STEPS_CAP)
        return context_steps, num_x, latent_dim, rollout_steps

    @classmethod
    def decode_to_centroid(cls, latents):
        """Decode (..., 47) latents to (..., 3) centroid velocities [vx, vy, vz]."""
        return decode_latents_to_centroid(latents, cls.ae, cls.converter, cls.device)

    @classmethod
    def _single_step_baseline(cls, sample_idx, x_idx, geom):
        """Target + persistence prediction for one (sample, coord). Model-independent."""
        key = ("single", sample_idx, x_idx, geom)
        if key not in cls._baseline_cache:
            context_steps, num_x, latent_dim, _ = geom
            context_len = num_x * context_steps
            batch = cls.samples[sample_idx].to(cls.device)
            t0 = context_len + x_idx
            target_v = cls.decode_to_centroid(batch[:, t0:t0 + 1, :latent_dim])
            p0 = (context_steps - 1) * num_x + x_idx
            persistence_v = cls.decode_to_centroid(batch[:, p0:p0 + 1, :latent_dim])
            cls._baseline_cache[key] = (target_v, persistence_v)
        return cls._baseline_cache[key]

    @classmethod
    def _rollout_baseline(cls, sample_idx, geom):
        key = ("rollout", sample_idx, geom)
        if key not in cls._baseline_cache:
            context_steps, num_x, latent_dim, rollout_steps = geom
            context_len = num_x * context_steps
            batch = cls.samples[sample_idx].to(cls.device)
            targets_latent = batch[:, context_len:context_len + rollout_steps, :latent_dim]
            targets_v = cls.decode_to_centroid(targets_latent)
            # Persistence: last context frame (26 tokens) repeated over the horizon
            last_frame = batch[:, context_len - num_x:context_len, :latent_dim]
            repeats = (rollout_steps + num_x - 1) // num_x
            persistence_latent = last_frame.repeat(1, repeats, 1)[:, :rollout_steps, :]
            persistence_v = cls.decode_to_centroid(persistence_latent)
            cls._baseline_cache[key] = (targets_latent, targets_v, persistence_latent, persistence_v)
        return cls._baseline_cache[key]

    @classmethod
    def _rollout_latents(cls, model, batch, n_frames):
        """Autoregressive rollout -> (B, n_frames * num_x, latent_dim).

        Delegates to the trainer's `rollout_frames`, which dispatches on
        `model.frame_native`, so a frame-tokenised checkpoint (40 tokens of
        26x47) and a token-level one (1040 tokens of 47) are both driven
        correctly and both come back in token order.
        """
        context_steps = getattr(Config, "VAL_CONTEXT_STEPS", 12)
        out = rollout_frames(model, batch, Config,
                             ctx_frames=context_steps, n_frames=n_frames)
        B, n, num_x, latent_dim = out.shape
        return out.reshape(B, n * num_x, latent_dim)

    @classmethod
    def _score_single_step(cls, model, row):
        geom = cls._geometry()
        context_steps, num_x, latent_dim, _ = geom

        model_err = persistence_err = 0.0
        count = 0
        for i in range(cls.n_single):
            batch = cls.samples[i].to(cls.device)
            # One rollout of a single frame covers EVERY coord: the prediction at
            # offset k is the same no matter how many further steps are taken, so
            # the old per-coord re-rollout was doing the same work len(COORDS)
            # times.
            preds = cls._rollout_latents(model, batch, 1)
            for x_idx in COORDS:
                target_v, persistence_v = cls._single_step_baseline(i, x_idx, geom)
                if x_idx >= preds.shape[1]:
                    continue
                pred_v = cls.decode_to_centroid(preds[:, x_idx:x_idx + 1, :])

                m_err = mae_loss(pred_v, target_v).item()
                p_err = mae_loss(persistence_v, target_v).item()
                model_err += m_err
                persistence_err += p_err
                count += 1
                print(f"    single  sample={i} coord={x_idx}  "
                      f"model MAE={m_err:.8f}  persistence MAE={p_err:.8f}")

        if count:
            row["single_model_mae"] = model_err / count
            row["single_persistence_mae"] = persistence_err / count
            row["single_improvement_pct"] = (
                (row["single_persistence_mae"] - row["single_model_mae"])
                / (row["single_persistence_mae"] + 1e-8) * 100)
            row["single_points"] = count
            print(f"    SINGLE-STEP  model={row['single_model_mae']:.6f}  "
                  f"persistence={row['single_persistence_mae']:.6f}  "
                  f"improvement={row['single_improvement_pct']:+.2f}%")

    @classmethod
    def _score_rollout(cls, model, row):
        geom = cls._geometry()
        context_steps, num_x, latent_dim, rollout_steps = geom
        row["rollout_horizon"] = rollout_steps

        model_err = persistence_err = 0.0
        model_latent_mse = persistence_latent_mse = 0.0
        count = 0
        for i in range(cls.n_rollout):
            batch = cls.samples[i].to(cls.device)
            targets_latent, targets_v, persistence_latent, persistence_v = \
                cls._rollout_baseline(i, geom)

            # Round the token horizon up to whole frames (a frame-native model
            # can only step a frame at a time) and then truncate back, so both
            # tokenizations are scored over exactly `rollout_steps` tokens.
            n_frames = (rollout_steps + num_x - 1) // num_x
            preds_latent = cls._rollout_latents(model, batch, n_frames)[:, :rollout_steps, :]

            if preds_latent.shape[1] != rollout_steps:
                print(f"    rollout sample={i}: only {preds_latent.shape[1]}/{rollout_steps} "
                      f"tokens produced (sequence exhausted); skipped")
                continue

            preds_v = cls.decode_to_centroid(preds_latent)
            m_err = mae_loss(preds_v, targets_v).item()
            p_err = mae_loss(persistence_v, targets_v).item()
            model_err += m_err
            persistence_err += p_err
            # Latent-space MSE: same quantity the trainer logs as `rollout_mse`,
            # so it can be cross-checked against the value stored in the ckpt.
            model_latent_mse += torch.mean((preds_latent - targets_latent) ** 2).item()
            persistence_latent_mse += torch.mean((persistence_latent - targets_latent) ** 2).item()
            count += 1
            print(f"    rollout sample={i}  model MAE={m_err:.8f}  "
                  f"persistence MAE={p_err:.8f}")

        if count:
            row["rollout_model_mae"] = model_err / count
            row["rollout_persistence_mae"] = persistence_err / count
            row["rollout_improvement_pct"] = (
                (row["rollout_persistence_mae"] - row["rollout_model_mae"])
                / (row["rollout_persistence_mae"] + 1e-8) * 100)
            row["rollout_model_latent_mse"] = model_latent_mse / count
            row["rollout_persistence_latent_mse"] = persistence_latent_mse / count
            row["rollout_sequences"] = count
            print(f"    ROLLOUT({rollout_steps} tok)  model={row['rollout_model_mae']:.6f}  "
                  f"persistence={row['rollout_persistence_mae']:.6f}  "
                  f"improvement={row['rollout_improvement_pct']:+.2f}%")

    # ------------------------------------------------------------------ report
    @classmethod
    def _scored(cls, metric):
        return [r for r in cls.results if r.get(metric) is not None]

    @classmethod
    def _best(cls, metric):
        rows = cls._scored(metric)
        return min(rows, key=lambda r: r[metric]) if rows else None

    @classmethod
    def _render_report(cls):
        lines = []
        add = lines.append
        hdr = (f"{'#':>2}  {'checkpoint':<44} {'kind':<12} {'var':<7} {'tok':<6} {'caus':<5} "
               f"{'1step MAE':>11} {'1step %':>9} {'roll MAE':>11} {'roll %':>9} {'result':<8} {'status':<28}")
        add("=" * len(hdr))
        add("MODEL vs PERSISTENCE LEADERBOARD")
        add("=" * len(hdr))
        for h in cls.header_lines:
            add(h)
        horizons = sorted({r["rollout_horizon"] for r in cls.results if r.get("rollout_horizon")})
        if horizons:
            add(f"rollout horizon     : {horizons} tokens "
                f"({[round(h / getattr(Config, 'NUM_X', 26), 1) for h in horizons]} time steps) "
                f"[TX_ROLLOUT_STEPS={ROLLOUT_STEPS_CAP}, 0=full]")
        add("")

        ranked = sorted(
            cls.results,
            key=lambda r: (r.get("rollout_improvement_pct") is None,
                           -(r.get("rollout_improvement_pct") or 0.0)))

        add(hdr)
        add("-" * len(hdr))
        for i, r in enumerate(ranked, start=1):
            caus = {True: "ok", False: "LEAK"}.get(r.get("causal"), "-")
            add(f"{i:>2}  {r['run'][:44]:<44} {r['kind']:<12} {str(r.get('variant', '?'))[:7]:<7} "
                f"{str(r.get('tokenization', '-'))[:6]:<6} {caus:<5} "
                f"{fmt(r.get('single_model_mae')):>11} "
                f"{fmt(r.get('single_improvement_pct'), '+.2f'):>9} "
                f"{fmt(r.get('rollout_model_mae')):>11} "
                f"{fmt(r.get('rollout_improvement_pct'), '+.2f'):>9} "
                f"{r.get('result', 'FAIL'):<8} "
                f"{r.get('status', '?')[:28]:<28}")

        add("")
        add("Persistence baseline (identical for every model):")
        for metric, label in (("single_persistence_mae", "single-step"),
                              ("rollout_persistence_mae", "rollout    ")):
            vals = {round(r[metric], 8) for r in cls._scored(metric)}
            if vals:
                add(f"  {label} MAE = {', '.join(f'{v:.6f}' for v in sorted(vals))}")

        add("")
        add("Checkpoint-recorded metrics (what the trainer saw at save time):")
        sub = (f"    {'checkpoint':<44} {'kind':<12} {'ep':>4} {'train L2':>10} "
               f"{'val L2':>10} {'roll MSE':>10} {'roll MSE (here)':>16} {'params M':>9}")
        add(sub)
        for r in ranked:
            add(f"    {r['run'][:44]:<44} {r['kind']:<12} {str(r.get('epoch', '-')):>4} "
                f"{fmt(r.get('ckpt_train_l2')):>10} {fmt(r.get('ckpt_val_l2')):>10} "
                f"{fmt(r.get('ckpt_rollout_mse')):>10} "
                f"{fmt(r.get('rollout_model_latent_mse')):>16} "
                f"{fmt(r.get('params_m'), '.2f'):>9}")

        # Per-run winner: which of the 2-4 checkpoints a run produced is actually best?
        add("")
        add("Best checkpoint per training run (by rollout improvement):")
        runs = {}
        for r in cls._scored("rollout_improvement_pct"):
            best = runs.get(r["run"])
            if best is None or r["rollout_improvement_pct"] > best["rollout_improvement_pct"]:
                runs[r["run"]] = r
        for run in sorted(runs, key=lambda k: -runs[k]["rollout_improvement_pct"]):
            r = runs[run]
            add(f"    {run:<44} -> {r['kind']:<12} "
                f"({r['rollout_improvement_pct']:+.2f}% vs persistence)")

        add("")
        beat_single = [r for r in cls._scored("single_improvement_pct")
                       if r["single_improvement_pct"] > 0]
        beat_rollout = [r for r in cls._scored("rollout_improvement_pct")
                        if r["rollout_improvement_pct"] > 0]
        add(f"Beat persistence -- single-step: {len(beat_single)}/{len(cls._scored('single_improvement_pct'))}"
            f"   rollout: {len(beat_rollout)}/{len(cls._scored('rollout_improvement_pct'))}")

        best_single = cls._best("single_model_mae")
        best_rollout = cls._best("rollout_model_mae")
        if best_single:
            add(f"BEST single-step: {best_single['name']} "
                f"(MAE {best_single['single_model_mae']:.6f}, "
                f"{best_single['single_improvement_pct']:+.2f}%)")
        if best_rollout:
            add(f"BEST rollout    : {best_rollout['name']} "
                f"(MAE {best_rollout['rollout_model_mae']:.6f}, "
                f"{best_rollout['rollout_improvement_pct']:+.2f}%)")

        skipped = [r for r in cls.results if r.get("status") != "ok"]
        if skipped:
            add("")
            add("Not scored:")
            for r in skipped:
                add(f"    {r['name']:<56} {r['status']}")
        add("=" * len(hdr))
        return "\n".join(lines)

    @classmethod
    def _print_report(cls):
        report = cls._render_report()
        print("\n" + report)

        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            md_path = os.path.join(REPORT_DIR, "model_leaderboard.md")
            with open(md_path, "w") as f:
                f.write("# Model vs Persistence Leaderboard\n\n```\n" + report + "\n```\n")

            csv_path = os.path.join(REPORT_DIR, "model_leaderboard.csv")
            cols = ["name", "run", "kind", "variant", "tokenization", "causal", "causal_leak",
                    "embed", "layers", "epoch", "params_m",
                    "single_model_mae", "single_persistence_mae", "single_improvement_pct",
                    "rollout_horizon", "rollout_model_mae", "rollout_persistence_mae",
                    "rollout_improvement_pct", "rollout_model_latent_mse",
                    "rollout_persistence_latent_mse", "ckpt_train_l2", "ckpt_val_l2",
                    "ckpt_rollout_mse", "eval_seconds", "result", "status"]
            with open(csv_path, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in cls.results:
                    f.write(",".join(str(r.get(c, "")).replace(",", ";") for c in cols) + "\n")
            print(f"Report written to:\n  {md_path}\n  {csv_path}")
        except OSError as e:
            print(f"WARNING: could not write report files to {REPORT_DIR}: {e}")

    # ------------------------------------------------------------------- tests
    def test_checkpoints_are_loadable(self):
        """Every discovered checkpoint rebuilds into its recorded architecture."""
        broken = [r for r in self.results if r.get("status") != "ok"]
        for r in broken:
            print(f"UNSCORED: {r['name']} -> {r['status']}")
        self.assertLess(len(broken), len(self.results),
                        "No checkpoint could be loaded and scored at all.")

    def test_best_beats_persistence_single_step(self):
        """At least one checkpoint beats persistence on single-step centroid MAE."""
        best = self._best("single_model_mae")
        self.assertIsNotNone(best, "No checkpoint produced a single-step score.")
        print(f"\nBest single-step: {best['name']} "
              f"MAE={best['single_model_mae']:.6f} vs "
              f"persistence={best['single_persistence_mae']:.6f} "
              f"({best['single_improvement_pct']:+.2f}%)")
        self.assertLess(best["single_model_mae"], best["single_persistence_mae"],
                        f"No model beat persistence on single-step centroid MAE. "
                        f"Best was {best['name']}.")

    def test_best_beats_persistence_rollout(self):
        """At least one checkpoint beats persistence over the rollout horizon."""
        best = self._best("rollout_model_mae")
        self.assertIsNotNone(best, "No checkpoint produced a rollout score.")
        print(f"\nBest rollout: {best['name']} "
              f"MAE={best['rollout_model_mae']:.6f} vs "
              f"persistence={best['rollout_persistence_mae']:.6f} "
              f"({best['rollout_improvement_pct']:+.2f}%) "
              f"over {best['rollout_horizon']} tokens")
        self.assertLess(best["rollout_model_mae"], best["rollout_persistence_mae"],
                        f"No model beat the persistence baseline in rollout. "
                        f"Best was {best['name']} at "
                        f"{best['rollout_improvement_pct']:+.2f}%.")

    def test_checkpoints_are_causal(self):
        """Every scored checkpoint's architecture must be unable to see the future.

        This is the guard against the two leaks that model_variants.py used to
        have. A checkpoint TRAINED with a leak still fails honestly here on its
        rollout metrics; what this test catches is a leak reintroduced into the
        architecture itself, which would silently flatter every other number in
        the report.
        """
        probed = [r for r in self.results if r.get("causal") is not None]
        self.assertTrue(probed, "No checkpoint got as far as the causality probe.")
        leaky = [r for r in probed if not r["causal"]]
        for r in leaky:
            print(f"LEAKY: {r['name']} -> past outputs moved {r['causal_leak']:.3e}")
        self.assertFalse(
            leaky,
            f"{len(leaky)}/{len(probed)} checkpoint architectures can see the future: "
            f"{[r['name'] for r in leaky]}. Every metric in this report would be "
            f"meaningless until that is fixed.")

    def test_checkpoint_count_limit(self):
        """Keep saved_models/ from growing without bound.

        A 5-arm sweep round writes up to 3 checkpoints per arm, so the old
        hard-coded "< 20" failed on a healthy sweep. Raise TX_MAX_CKPTS if you
        are deliberately keeping several rounds around.
        """
        num_ckpts = len(self.checkpoints)
        print(f"\nDEBUG: Asserting total checkpoint count < {MAX_CKPTS}. Found: {num_ckpts}")
        self.assertLess(num_ckpts, MAX_CKPTS,
                        f"Expected fewer than {MAX_CKPTS} checkpoints, found {num_ckpts}. "
                        f"Prune saved_models/ or raise TX_MAX_CKPTS.")

    def test_individual_checkpoint_status_and_performance(self):
        """Walk all discovered checkpoints, reporting PASS/FAIL status based on metrics and loadability."""
        print("\n" + "=" * 78 + "\nINDIVIDUAL CHECKPOINT EVALUATION REPORT\n" + "=" * 78)
        for r in self.results:
            with self.subTest(checkpoint=r["name"]):
                print(f"Checking {r['name']} (kind: {r['kind']})...")
                self.assertEqual(r.get("status"), "ok", 
                                 f"Checkpoint {r['name']} failed to load or evaluate successfully. Status: {r.get('status')}")
                
                # Assert that "best" and "rollout_best" models must beat the persistence baseline (PASS result)
                if r["kind"] in ("best", "rollout_best"):
                    self.assertEqual(r.get("result"), "PASS",
                                     f"Best checkpoint {r['name']} (kind: {r['kind']}) failed to beat the persistence baseline. "
                                     f"Rollout improvement: {fmt(r.get('rollout_improvement_pct'), '+.2f')}% "
                                     f"Single-step improvement: {fmt(r.get('single_improvement_pct'), '+.2f')}%")


# ==============================================================================
# Deep dive: r1_a3b_delta_ar_rollout_best.pt vs. persistence
# ==============================================================================
#
# The leaderboard above answers "did anything beat persistence." This class
# answers "how, where, and by how much" for exactly one checkpoint:
# r1_a3b_delta_ar_rollout_best.pt -- the Branch-A arm from
# Documentation/deep_dive_decision_tree.md that combines PREDICT_DELTA with an
# autoregressive rollout loss (arm a3b_delta_ar, branch "A -- single-step works,
# the horizon collapses"; see train_production_transformer_deep_dive.py's
# ROUND2_ARMS for the training recipe). A leaderboard run clocked this
# checkpoint at +7.97% vs persistence over the rollout horizon; this test locks
# that claim to a PASS/FAIL gate and breaks the win down by rollout frame
# (time), spatial coordinate and velocity component (space), and raw
# latent/token MSE (the trainer's own objective), plus compute cost.
#
# Deliberately separate from the sweep above: that one caps horizon and sample
# count so a dozen checkpoints stay fast to compare. This targets one
# known-good checkpoint and can afford the full horizon and more sequences.
#
# The checkpoint is trained and kept on the remote GPU box, not committed to
# the repo -- this test SKIPS (does not fail) when it is absent locally.
#
# Tunable via environment variables (all optional):
#   TX_DEEPDIVE_RUN            target run name          (default "r1_a3b_delta_ar")
#   TX_DEEPDIVE_KIND           target checkpoint kind   (default "rollout_best")
#   TX_DEEPDIVE_SAMPLES        val sequences to use      (default 8)
#   TX_DEEPDIVE_ROLLOUT_STEPS  token horizon, 0 = full   (default 0)
#   TX_DEEPDIVE_SUBSET_RATIO   val-set fraction to load  (default 0.2)

DEEPDIVE_RUN = os.environ.get("TX_DEEPDIVE_RUN", "r1_a3b_delta_ar")
DEEPDIVE_KIND = os.environ.get("TX_DEEPDIVE_KIND", "rollout_best")
DEEPDIVE_SAMPLES = _env_int("TX_DEEPDIVE_SAMPLES", 8)
DEEPDIVE_ROLLOUT_STEPS = _env_int("TX_DEEPDIVE_ROLLOUT_STEPS", 0)  # 0 = full horizon
DEEPDIVE_SUBSET_RATIO = _env_float("TX_DEEPDIVE_SUBSET_RATIO", 0.2)

VELOCITY_COMPONENTS = ("vx", "vy", "vz")


def _pct_improvement(persistence, model):
    return (persistence - model) / (persistence + 1e-8) * 100


class TestR1A3bDeltaArDeepDive(unittest.TestCase):
    """Deep dive on r1_a3b_delta_ar_rollout_best.pt: where and how it beats persistence.

    Skips (does not fail) if the checkpoint or validation data is not present
    -- this targets a specific checkpoint trained on the remote GPU box.
    """

    report = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_name = f"{DEEPDIVE_RUN}_{DEEPDIVE_KIND}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_name}.pt")
        print(f"\nDEBUG: deep-dive target checkpoint = {ckpt_path}")
        if not os.path.exists(ckpt_path):
            raise unittest.SkipTest(
                f"{ckpt_path} not found. This deep dive targets one specific "
                f"checkpoint that lives on the remote GPU box, not a repo asset "
                f"-- run it there, or point TX_DEEPDIVE_RUN/TX_DEEPDIVE_KIND at "
                f"a checkpoint you do have locally.")

        val_h5 = os.path.join(PKG_ROOT, "data/val_40.h5")
        if not os.path.exists(val_h5):
            raise unittest.SkipTest("Validation data not found at " + val_h5)

        cls.ae, cls.ae_path, cls.converter, cls.metric_space = load_autoencoder(cls.device)

        cls.dataset = TransformerDataset(val_h5, subset_ratio=DEEPDIVE_SUBSET_RATIO)
        n_avail = len(cls.dataset)
        cls.n_samples = min(DEEPDIVE_SAMPLES, n_avail)
        cls.samples = [cls.dataset[i].unsqueeze(0) for i in range(cls.n_samples)]

        t_load0 = time.perf_counter()
        ckpt = load_checkpoint(ckpt_path)
        load_seconds = time.perf_counter() - t_load0
        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)

        had_config = apply_checkpoint_config(ckpt)
        model = get_model(Config)
        state_dict, stripped = normalize_state_dict(ckpt["model_state_dict"])
        if stripped:
            print(f"  stripped state-dict prefix(es): {stripped}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in incompatible.missing_keys if k not in BENIGN_MISSING_KEYS]
        if missing:
            reset_config()
            raise unittest.SkipTest(
                f"{ckpt_name}: {len(missing)} missing key(s), cannot be scored: "
                f"{missing[:4]}")

        model.eval()
        model.to(cls.device)
        cls.model = model
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        probe = probe_causality(model, Config, cls.device)

        context_steps = getattr(Config, "VAL_CONTEXT_STEPS", 12)
        num_x = getattr(Config, "NUM_X", 26)
        latent_dim = getattr(Config, "LATENT_DIM", 47)
        full_horizon = getattr(Config, "VAL_ROLLOUT_STEPS", 28)
        capped = full_horizon if DEEPDIVE_ROLLOUT_STEPS <= 0 else min(full_horizon, DEEPDIVE_ROLLOUT_STEPS)
        n_frames = max(1, (capped + num_x - 1) // num_x)
        rollout_steps = n_frames * num_x  # whole frames; a frame-native model can't step partial

        cls.geom = dict(context_steps=context_steps, num_x=num_x, latent_dim=latent_dim,
                         n_frames=n_frames, rollout_steps=rollout_steps)

        cls.report = dict(
            run=DEEPDIVE_RUN, kind=DEEPDIVE_KIND, ckpt_name=ckpt_name, ckpt_path=ckpt_path,
            ckpt_size_mb=ckpt_size_mb, load_seconds=load_seconds,
            epoch=ckpt.get("epoch"), ckpt_train_l2=ckpt.get("train_l2"),
            ckpt_val_l2=ckpt.get("val_l2"), ckpt_rollout_mse=ckpt.get("rollout_mse"),
            had_config=had_config, variant=getattr(Config, "VARIANT", "?"),
            embed=getattr(Config, "EMBED_SIZE", "?"), layers=getattr(Config, "N_LAYERS", "?"),
            predict_delta=bool(getattr(Config, "PREDICT_DELTA", False)),
            tokenization=getattr(Config, "TOKENIZATION", "token"),
            params_m=params_m, causal=probe["causal"], causal_leak=probe["max_change_before_cut"],
            causal_tolerance=probe["tolerance"], device=cls.device,
            metric_space=cls.metric_space, n_samples=cls.n_samples, n_avail=n_avail,
            subset_ratio=DEEPDIVE_SUBSET_RATIO, num_x=num_x, latent_dim=latent_dim,
            n_frames=n_frames, rollout_steps=rollout_steps,
        )

        if probe["causal"]:
            cls._compute_breakdowns()
        reset_config()
        if cls.device == "cuda":
            torch.cuda.empty_cache()

    @classmethod
    def decode_to_centroid(cls, latents):
        return decode_latents_to_centroid(latents, cls.ae, cls.converter, cls.device)

    # -------------------------------------------------------------- evaluation
    @classmethod
    def _compute_breakdowns(cls):
        """One full-horizon rollout per sample; every axis below is a slice of it.

        Frame 1 of the rollout covers every spatial coordinate exactly once
        (the same fact `TestModelVsBaseline._score_single_step` exploits), so
        single-step and full-horizon numbers come from one rollout per sample
        rather than two.
        """
        geom = cls.geom
        num_x, latent_dim = geom["num_x"], geom["latent_dim"]
        n_frames, rollout_steps = geom["n_frames"], geom["rollout_steps"]
        context_len = num_x * geom["context_steps"]

        per_frame = [dict(frame=f + 1, model_mae=0.0, pers_mae=0.0,
                           model_lat_mse=0.0, pers_lat_mse=0.0, n=0)
                     for f in range(n_frames)]
        per_coord_frame1 = [dict(coord=x, model_mae=0.0, pers_mae=0.0, n=0) for x in range(num_x)]
        per_coord_last = [dict(coord=x, model_mae=0.0, pers_mae=0.0, n=0) for x in range(num_x)]
        per_component_frame1 = {c: dict(model_mae=0.0, pers_mae=0.0, n=0) for c in VELOCITY_COMPONENTS}
        per_component_last = {c: dict(model_mae=0.0, pers_mae=0.0, n=0) for c in VELOCITY_COMPONENTS}
        wall_times = []

        def _accum_coord(acc, model_v, pers_v, target_v):
            for x in range(num_x):
                acc[x]["model_mae"] += mae_loss(model_v[:, x:x + 1], target_v[:, x:x + 1]).item()
                acc[x]["pers_mae"] += mae_loss(pers_v[:, x:x + 1], target_v[:, x:x + 1]).item()
                acc[x]["n"] += 1

        def _accum_component(acc, model_v, pers_v, target_v):
            for c, name in enumerate(VELOCITY_COMPONENTS):
                d = acc[name]
                d["model_mae"] += torch.mean(torch.abs(model_v[..., c] - target_v[..., c])).item()
                d["pers_mae"] += torch.mean(torch.abs(pers_v[..., c] - target_v[..., c])).item()
                d["n"] += 1

        with torch.no_grad():
            for i in range(cls.n_samples):
                batch = cls.samples[i].to(cls.device)

                targets_latent = batch[:, context_len:context_len + rollout_steps, :latent_dim]
                last_frame = batch[:, context_len - num_x:context_len, :latent_dim]
                persistence_latent = last_frame.repeat(1, n_frames, 1)

                t0 = time.perf_counter()
                out = rollout_frames(cls.model, batch, Config,
                                     ctx_frames=geom["context_steps"], n_frames=n_frames)
                wall_times.append(time.perf_counter() - t0)
                B, n, nx, ld = out.shape
                preds_latent = out.reshape(B, n * nx, ld)[:, :rollout_steps, :]

                targets_v = cls.decode_to_centroid(targets_latent)
                persistence_v = cls.decode_to_centroid(persistence_latent)
                preds_v = cls.decode_to_centroid(preds_latent)

                for f in range(n_frames):
                    sl = slice(f * num_x, (f + 1) * num_x)
                    row = per_frame[f]
                    row["model_mae"] += mae_loss(preds_v[:, sl], targets_v[:, sl]).item()
                    row["pers_mae"] += mae_loss(persistence_v[:, sl], targets_v[:, sl]).item()
                    row["model_lat_mse"] += torch.mean(
                        (preds_latent[:, sl] - targets_latent[:, sl]) ** 2).item()
                    row["pers_lat_mse"] += torch.mean(
                        (persistence_latent[:, sl] - targets_latent[:, sl]) ** 2).item()
                    row["n"] += 1

                frame1_sl = slice(0, num_x)
                last_sl = slice((n_frames - 1) * num_x, n_frames * num_x)
                _accum_coord(per_coord_frame1, preds_v[:, frame1_sl], persistence_v[:, frame1_sl],
                             targets_v[:, frame1_sl])
                _accum_coord(per_coord_last, preds_v[:, last_sl], persistence_v[:, last_sl],
                             targets_v[:, last_sl])
                _accum_component(per_component_frame1, preds_v[:, frame1_sl], persistence_v[:, frame1_sl],
                                 targets_v[:, frame1_sl])
                _accum_component(per_component_last, preds_v[:, last_sl], persistence_v[:, last_sl],
                                 targets_v[:, last_sl])

                print(f"    sample={i}  rollout wall={wall_times[-1]:.3f}s  "
                      f"frame1 MAE={per_frame[0]['model_mae'] / (i + 1):.8f}  "
                      f"frame{n_frames} MAE={per_frame[-1]['model_mae'] / (i + 1):.8f}")

        for row in per_frame:
            row["model_mae"] /= row["n"]
            row["pers_mae"] /= row["n"]
            row["model_lat_mse"] /= row["n"]
            row["pers_lat_mse"] /= row["n"]
            row["improvement_pct"] = _pct_improvement(row["pers_mae"], row["model_mae"])
            row["latent_improvement_pct"] = _pct_improvement(row["pers_lat_mse"], row["model_lat_mse"])
        for acc in (per_coord_frame1, per_coord_last):
            for row in acc:
                row["model_mae"] /= row["n"]
                row["pers_mae"] /= row["n"]
                row["improvement_pct"] = _pct_improvement(row["pers_mae"], row["model_mae"])
        for acc in (per_component_frame1, per_component_last):
            for row in acc.values():
                row["model_mae"] /= row["n"]
                row["pers_mae"] /= row["n"]
                row["improvement_pct"] = _pct_improvement(row["pers_mae"], row["model_mae"])

        overall_model_mae = sum(r["model_mae"] for r in per_frame) / len(per_frame)
        overall_pers_mae = sum(r["pers_mae"] for r in per_frame) / len(per_frame)
        overall_model_lat_mse = sum(r["model_lat_mse"] for r in per_frame) / len(per_frame)
        overall_pers_lat_mse = sum(r["pers_lat_mse"] for r in per_frame) / len(per_frame)

        cls.report.update(dict(
            per_frame=per_frame,
            per_coord_frame1=per_coord_frame1, per_coord_last=per_coord_last,
            per_component_frame1=per_component_frame1, per_component_last=per_component_last,
            overall_model_mae=overall_model_mae, overall_pers_mae=overall_pers_mae,
            overall_improvement_pct=_pct_improvement(overall_pers_mae, overall_model_mae),
            overall_model_lat_mse=overall_model_lat_mse, overall_pers_lat_mse=overall_pers_lat_mse,
            overall_latent_improvement_pct=_pct_improvement(overall_pers_lat_mse, overall_model_lat_mse),
            frame1_improvement_pct=per_frame[0]["improvement_pct"],
            last_improvement_pct=per_frame[-1]["improvement_pct"],
            avg_rollout_wall_seconds=sum(wall_times) / len(wall_times),
            total_wall_seconds=sum(wall_times),
            tokens_per_second=(rollout_steps * len(wall_times)) / sum(wall_times) if sum(wall_times) else None,
        ))

    # ------------------------------------------------------------------ report
    @classmethod
    def _render_report(cls):
        r = cls.report
        lines = []
        add = lines.append
        add("=" * 88)
        add(f"DEEP DIVE: {r['ckpt_name']} vs. PERSISTENCE")
        add("=" * 88)
        add(f"checkpoint          : {r['ckpt_path']}  ({r['ckpt_size_mb']:.1f} MB, "
            f"loaded in {r['load_seconds']:.2f}s)")
        add(f"device              : {r['device']}")
        add(f"metric space        : {r['metric_space']}")
        add(f"architecture        : variant={r['variant']} embed={r['embed']} layers={r['layers']} "
            f"tokenization={r['tokenization']} predict_delta={r['predict_delta']} "
            f"params={r['params_m']:.2f}M epoch={r['epoch']}")
        add(f"causal              : {'OK' if r['causal'] else 'LEAK'} "
            f"(past outputs moved {r['causal_leak']:.3e}, tolerance {r['causal_tolerance']:.3e})")
        add(f"val data            : {r['n_samples']}/{r['n_avail']} sequences "
            f"(subset_ratio={r['subset_ratio']})")
        add(f"rollout horizon     : {r['rollout_steps']} tokens = {r['n_frames']} frames "
            f"(num_x={r['num_x']})")

        if not r["causal"]:
            add("")
            add("*** SKIPPED breakdowns: architecture failed the causality probe. ***")
            add("=" * 88)
            return "\n".join(lines)

        add("")
        add("-- TIME: accuracy over the rollout horizon --------------------------------------")
        add(f"{'frame':>5} {'model MAE':>12} {'persist MAE':>12} {'improve %':>10} "
            f"{'model lat MSE':>14} {'persist lat MSE':>16} {'lat improve %':>14}")
        for row in r["per_frame"]:
            add(f"{row['frame']:>5} {row['model_mae']:>12.6f} {row['pers_mae']:>12.6f} "
                f"{row['improvement_pct']:>+9.2f}% {row['model_lat_mse']:>14.3e} "
                f"{row['pers_lat_mse']:>16.3e} {row['latent_improvement_pct']:>+13.2f}%")
        add(f"frame1% (single-step) = {r['frame1_improvement_pct']:+.2f}%   "
            f"last% (frame {r['n_frames']}) = {r['last_improvement_pct']:+.2f}%   "
            f"gap (accumulation cost) = {r['frame1_improvement_pct'] - r['last_improvement_pct']:+.2f} pts")

        add("")
        add("-- SPACE: accuracy by coordinate (token index within a frame) -------------------")
        add(f"{'coord':>5} {'frame1 model':>13} {'frame1 persist':>15} {'frame1 %':>9}   "
            f"{'last model':>11} {'last persist':>13} {'last %':>9}")
        for c1, cL in zip(r["per_coord_frame1"], r["per_coord_last"]):
            add(f"{c1['coord']:>5} {c1['model_mae']:>13.6f} {c1['pers_mae']:>15.6f} "
                f"{c1['improvement_pct']:>+8.2f}%   {cL['model_mae']:>11.6f} {cL['pers_mae']:>13.6f} "
                f"{cL['improvement_pct']:>+8.2f}%")

        add("")
        add("-- SPACE: accuracy by velocity component -----------------------------------------")
        add(f"{'component':>10} {'frame1 model':>13} {'frame1 persist':>15} {'frame1 %':>9}   "
            f"{'last model':>11} {'last persist':>13} {'last %':>9}")
        for name in VELOCITY_COMPONENTS:
            c1 = r["per_component_frame1"][name]
            cL = r["per_component_last"][name]
            add(f"{name:>10} {c1['model_mae']:>13.6f} {c1['pers_mae']:>15.6f} "
                f"{c1['improvement_pct']:>+8.2f}%   {cL['model_mae']:>11.6f} {cL['pers_mae']:>13.6f} "
                f"{cL['improvement_pct']:>+8.2f}%")

        add("")
        add("-- COMPUTE ------------------------------------------------------------------------")
        add(f"params              : {r['params_m']:.2f}M")
        add(f"checkpoint size     : {r['ckpt_size_mb']:.1f} MB")
        add(f"checkpoint load     : {r['load_seconds']:.2f}s")
        add(f"rollout wall-clock  : {r['avg_rollout_wall_seconds']:.3f}s/sample avg, "
            f"{r['total_wall_seconds']:.2f}s total over {r['n_samples']} samples")
        if r["tokens_per_second"]:
            add(f"throughput          : {r['tokens_per_second']:.1f} tokens/s "
                f"({r['rollout_steps']} tokens x {r['n_samples']} samples)")

        add("")
        add("-- SUMMARY --------------------------------------------------------------------------")
        add(f"OVERALL (horizon-avg) centroid MAE : model={r['overall_model_mae']:.6f}  "
            f"persistence={r['overall_pers_mae']:.6f}  improvement={r['overall_improvement_pct']:+.2f}%")
        add(f"OVERALL (horizon-avg) latent MSE    : model={r['overall_model_lat_mse']:.3e}  "
            f"persistence={r['overall_pers_lat_mse']:.3e}  "
            f"improvement={r['overall_latent_improvement_pct']:+.2f}%")
        verdict = "PASS" if r["overall_model_mae"] < r["overall_pers_mae"] else "FAIL"
        add(f"RESULT              : {verdict} (beats persistence: "
            f"{r['overall_model_mae'] < r['overall_pers_mae']})")
        add("=" * 88)
        return "\n".join(lines)

    @classmethod
    def tearDownClass(cls):
        if not cls.report:
            return
        report = cls._render_report()
        print("\n" + report)

        try:
            os.makedirs(REPORT_DIR, exist_ok=True)
            md_path = os.path.join(REPORT_DIR, "r1_a3b_delta_ar_deep_dive.md")
            with open(md_path, "w") as f:
                f.write(f"# Deep dive: {cls.report['ckpt_name']} vs persistence\n\n"
                        "```\n" + report + "\n```\n")

            if cls.report.get("per_frame"):
                csv_path = os.path.join(REPORT_DIR, "r1_a3b_delta_ar_deep_dive.csv")
                cols = ["frame", "model_mae", "pers_mae", "improvement_pct",
                        "model_lat_mse", "pers_lat_mse", "latent_improvement_pct"]
                with open(csv_path, "w") as f:
                    f.write(",".join(cols) + "\n")
                    for row in cls.report["per_frame"]:
                        f.write(",".join(str(row[c]) for c in cols) + "\n")
                print(f"Report written to:\n  {md_path}\n  {csv_path}")
            else:
                print(f"Report written to:\n  {md_path}")
        except OSError as e:
            print(f"WARNING: could not write deep-dive report to {REPORT_DIR}: {e}")

    # ------------------------------------------------------------------- tests
    def test_checkpoint_is_causal(self):
        """The architecture must be unable to see the future -- otherwise every
        other number in this deep dive is meaningless."""
        r = self.report
        self.assertTrue(r["causal"],
                        f"{r['ckpt_name']}: architecture leaks the future "
                        f"(past outputs moved {r['causal_leak']:.3e}). Every other "
                        f"metric here would be meaningless until this is fixed.")

    def test_single_step_beats_persistence(self):
        """Frame-1 (one step ahead) model MAE must beat persistence."""
        r = self.report
        frame1 = r["per_frame"][0]
        print(f"\nframe1: model MAE={frame1['model_mae']:.6f}  "
              f"persistence MAE={frame1['pers_mae']:.6f}  "
              f"improvement={frame1['improvement_pct']:+.2f}%")
        self.assertLess(frame1["model_mae"], frame1["pers_mae"],
                        f"{r['ckpt_name']} did not beat persistence on the first "
                        f"rollout step ({frame1['improvement_pct']:+.2f}%).")

    def test_full_rollout_beats_persistence(self):
        """Headline gate: horizon-averaged model MAE must beat persistence.

        A prior leaderboard run measured +7.97% here; this assertion does not
        pin that exact number (sample count and horizon are deliberately
        larger in this deep dive), only that the model still wins.
        """
        r = self.report
        print(f"\nOVERALL: model MAE={r['overall_model_mae']:.6f}  "
              f"persistence MAE={r['overall_pers_mae']:.6f}  "
              f"improvement={r['overall_improvement_pct']:+.2f}% "
              f"(reference leaderboard measurement: +7.97%) "
              f"over {r['rollout_steps']} tokens / {r['n_frames']} frames, "
              f"{r['n_samples']} sequences")
        self.assertLess(r["overall_model_mae"], r["overall_pers_mae"],
                        f"{r['ckpt_name']} did not beat persistence over the full "
                        f"rollout horizon ({r['overall_improvement_pct']:+.2f}% vs. "
                        f"the +7.97% previously measured).")


if __name__ == '__main__':
    unittest.main()
