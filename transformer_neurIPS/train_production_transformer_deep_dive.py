"""
Deep-dive transformer trainer for the NeurIPS latent fluid-dynamics sequences.

v2.0 PINNING NOTE (NUM_TIME=80)
===============================
This trainer is intentionally pinned to the 80-frame (v2.0) target. There is
NO 40/80 switch, NO --num-time flag, and NO 40-frame fallback. The v1.0
40-frame model remains fully reproducible from its frozen inputs and outputs:

  * data:       transformer_neurIPS/data/train_40.h5 / val_40.h5 (unchanged)
  * checkpoint: transformer_neurIPS/saved_models/r1_a3b_delta_ar_rollout_best.pt
  * results:    transformer_neurIPS/tests/reports/*.md

v2.0 is a distinct, forward-only pinned target: `train_80.h5` / `val_80.h5`,
positional-embedding capacity `NUM_TIME * NUM_X = 80 * 10 = 800` tokens
(v3.1 restricted NUM_X from 26 to 10; see OVERVIEW.md §12.1).
See transformer_neurIPS/OVERVIEW.md for the full v1.0 -> v2.0 rationale.

This is the ARM-DRIVEN rewrite. One process trains exactly one "arm" (a named
set of config overrides); `sweep_deep_dive.py` runs several arms concurrently on
one box and aggregates the results into a single uploadable report.

    # zero-arg default (Mac/MPS friendly): launches the v1.0 winner arm
    # `a3b_delta_ar` with the default warm-start from
    # `saved_models/r1_a3b_delta_ar_rollout_best.pt` and the auto-detected
    # device regime (rainbow micro-batch on MPS/CPU, H200-defaults on CUDA).
    python train_production_transformer_deep_dive.py

    # explicit overrides (all optional):
    python train_production_transformer_deep_dive.py --arm a0_control --max-steps 6000
    python train_production_transformer_deep_dive.py --diagnostics-only
    python train_production_transformer_deep_dive.py --smoke-test
    python train_production_transformer_deep_dive.py --list-arms

WHAT CHANGED VS THE PREVIOUS VERSION, AND WHY
=============================================
Bugs, all of which capped rollout quality independently of architecture:

1. CAUSALITY. Fixed in model_variants.py (see its docstring). Every run now
   PROVES causality at startup via `probe_causality()` and refuses to train if
   the model can see the future (override with --allow-leak).

2. THE HEADLINE METRIC COMPARED DIFFERENT POPULATIONS. Persistence MSE was
   computed over the whole validation batch (512 sequences) while the model
   rollout was computed over `[:8]`. `persistence_improvement_pct` was therefore
   512 baseline sequences against 8 model sequences. `evaluate()` now scores
   both on the identical, fixed, unshuffled row set.

3. `OneCycleLR` NEVER ANNEALED. With `EPOCHS=100000` and `pct_start=0.1` the
   warmup spanned the first 10,000 epochs, so the LR was still ramping and the
   cosine decay that breaks plateaus was 90,000 epochs away. Replaced with an
   explicit warmup + cosine schedule over the ACTUAL step budget.

4. THE AR AUXILIARY LOSS TRAINED THE WRONG THING. `ar_context_len=128` is
   `4*26 + 24`, i.e. mid-frame, and `AR_ROLLOUT_STEPS=5` stayed inside that one
   time frame -- so it trained spatial continuation, not temporal dynamics. The
   AR loss is now FRAME-ALIGNED: context is a whole number of frames and the
   horizon is a whole number of frames.

5. NO GRADIENT CLIPPING, and fp16+GradScaler on hardware with bf16.

6. UNNORMALIZED INPUT FEATURES. prepare_data.py writes columns 47:52 as raw
   magnitudes (x in [-29,69], y/z in +-80, t in [0,39], param in [5.6,17.8])
   next to latents in ~[0,1], all through one nn.Linear -- a ~200x
   input-variance mismatch. `NORMALIZE_FEATURES` standardises per column using
   statistics measured from the training split and stored in the checkpoint.

7. WALL-CLOCK BUDGETS BIAS A CONCURRENT SWEEP. Arms sharing a GPU slow each
   other down unequally, so a time budget silently gives some arms more
   gradient steps than others. The clock is now OPTIMIZER STEPS, with wall time
   only as a safety net.

Also: `TRAIN_SUBSET_RATIO` was 0.5 (half the data unused) and validation ran
every epoch (~29 optimizer steps at full data), which spent most of the run
inside a 728-step sequential rollout. Both are now explicit and step-based.

FREE DIAGNOSTICS (`--diagnostics-only`)
=======================================
Two questions get answered before any training, because a wrong answer to
either makes the whole sweep meaningless:

* Is attention actually causal, under this exact torch build? Probed, both for
  the fixed implementation and for the old `nn.MultiheadAttention` + is_causal
  hint call, so the historical leak is measured rather than assumed.

* Is there ANY learnable temporal structure beyond persistence? A ridge-fit
  linear frame-to-frame map is rolled out under the identical protocol. If a
  linear map beats persistence by a lot and the transformer does not, the
  transformer is broken. If the linear map also gets ~0%, persistence is simply
  a strong baseline at this sampling rate and the framing needs to change, not
  the architecture.

IF YOU ARE RUNNING THIS ON CUDA
===============================
The trainer detects the device once at startup via `pick_device()` and picks a
`TrainRegime` accordingly. The CUDA branch is the fast-path; MPS/CPU is the
laptop-debug path. What the CUDA branch flips vs. MPS, and why:

  * physical micro-batch bumps from 1 to 32 (or 64 if the device name contains
    "H200") and gradient accumulation collapses to 1. On MPS the caching
    allocator cannot reuse blocks across the changing shapes of an
    autoregressive rollout, so peak memory scales with the batch inside ONE
    rollout call -- a batch of 32 blows past a Mac's MPS ceiling on the very
    first batch. CUDA has no such fragmentation problem, so we run the full
    batch physically and drop the accumulation loop entirely.
  * autocast to bf16 turns on (Ampere+/H200); GradScaler is NOT used because
    bf16 does not underflow the way fp16 does.
  * `torch.compile(model)` is attempted in a try/except (never fatal): a 15-30%
    step-time win on H200, but MPS Inductor is not production-ready.
  * `torch.backends.cudnn.benchmark = True` and
    `torch.set_float32_matmul_precision('high')` are set so the residual fp32
    ops (LayerNorm, softmax reductions) run on TF32 cores.

The banner printed at startup shows a small CUDA-vs-MPS diff table so the
regime is obvious in every log; set `PFD_NO_COLOR=1` (or `NO_COLOR=1`, or pipe
stdout to a file) to strip the ANSI colouring.

H200 PERMUTATIONS (intended, NOT executed by this agent)
========================================================
Three runs are meant to be launched on an H200 host once the repo lands there.
The code is already H200-ready: dropping the tree onto a CUDA box and invoking
this trainer picks up all the CUDA-branch defaults automatically. No code edits
are required on the CUDA host.

  (a) WARM-START FROM v1.0 (promoted path):
        python train_production_transformer_deep_dive.py --arm a3b_delta_ar
      Uses the default `--warm-start saved_models/r1_a3b_delta_ar_rollout_best.pt`
      so 99.57% of the 4.79 M params transfer from the 40-frame winner and only
      `time_embeddings.weight` reinitialises for the new 80-frame horizon.

  (b) COLD-START FROM SCRATCH (comparison run):
        python train_production_transformer_deep_dive.py --arm a3b_delta_ar --no-warm-start
      Same arm, no v1.0 weights. Answers "how much of v2.0 quality is the extra
      horizon vs. the pre-training".

  (c) STAGED 40->80 CURRICULUM (optional):
      Reserved for the case where (a) plateaus. Fine-tune (a) at frozen
      transformer body and only re-train the time embedding + output head for a
      short window, then unfreeze. Not implemented here -- the code path is
      just (a) with a manual LR schedule / param-group override.

PROMOTION GATE (H200)
=====================
An 80-frame checkpoint is promotable only if BOTH of the following hold:

  * `probe_causality(...)['causal'] is True` on the FINAL weights, not just at
    startup. The trainer already refuses to save otherwise.
  * val-loss at the shared 1..28-frame horizon is not WORSE than the frozen
    v1.0 numbers in `tests/reports/r1_a3b_delta_ar_deep_dive.md`
    (single-step centroid MAE 2.47e-4, 28-frame rollout MAE 9.96e-4,
    latent MSE 4.11e-5). This is exactly the shared window `persistence_formal_documentation.py`
    reports; a v2.0 regression at those horizons means the extension to 68 frames
    is being paid for by short-horizon quality, which we do not want.

CHECKPOINT NAMING (compatible with persistence_formal_documentation.py)
=======================================================================
Every saved checkpoint follows the `r{SWEEP_ROUND}_{ARM}_{kind}.pt` pattern
(kinds: `best`, `rollout_best`, `train_best`, `latest`). This is exactly the
pattern `persistence_formal_documentation.py`'s `PFD_RUN` / `PFD_KIND` env-vars
key off, so a promoted H200 checkpoint drops straight into that harness with
no rename step.

SCRIPTED MODEL SAVES
====================
`Config.SAVE_SCRIPTED_MODELS = True` (default) makes every call to
`save_checkpoint(...)` also emit a self-contained TorchScript companion at
`<path>_scripted.pt`. Unlike the plain state-dict `.pt`, the scripted file
does NOT need `model_variants.py` / `Config` on the reload side: the whole
computation graph and every parameter/buffer are baked into the artifact,
so `torch.jit.load(path)` works standalone.

For every kind of checkpoint the trainer produces, the paired files are:

    r{SWEEP_ROUND}_{ARM}_{kind}.pt           # state dict (existing)
    r{SWEEP_ROUND}_{ARM}_{kind}_scripted.pt  # TorchScript companion (new)

Guardrails around the scripted save (see `save_scripted_model()` for the
implementation of each one):

  * `torch.compile(model)` is unwrapped via `getattr(model, "_orig_mod",
    model)` BEFORE scripting -- TorchScript cannot script an
    `OptimizedModule`.
  * `torch.jit.script` is attempted first (preserves control flow such as
    the `PREDICT_DELTA` branch), then `torch.jit.trace` on a
    representative synthetic input as a fallback.
  * `feat_mean` / `feat_std` are asserted to be registered buffers on
    both `BaseTransformer` and `FrameTransformer` before saving, because
    a plain attribute would not ride along in the scripted artifact and
    the eval side would silently see uninitialised statistics.
  * `frame_native` (class attribute) is honoured to size the
    representative example correctly for `FrameTransformer` vs. the
    token-native variants.
  * After writing, the file is reloaded with `torch.jit.load` and one
    forward is executed on a CPU synthetic input; a failed roundtrip is
    logged in yellow but does not abort training -- the state-dict `.pt`
    is authoritative.
  * Every completed write (state-dict AND scripted) is logged with the
    full absolute path in RAINBOW colouring via `_log_write()`, so
    scrollback answers 'where did that go?' without re-deriving
    `Config.CHECKPOINT_DIR`. Honours `PFD_NO_COLOR` / `NO_COLOR` /
    non-tty stdout.

Unit-tested end-to-end in `tests/test_scripted_save.py` (compile-unwrap,
buffer-requirement failure mode, frame-native and token-native variants,
`torch.jit.load` roundtrip on CPU, rainbow-log absolute-path emission).

DECODED-CENTROID TRAINING LOSS
==============================
The training / evaluation error metric is no longer computed in the 47-dim
autoencoder latent space. Latent-L2 is not a physical quantity: its
per-dimension scale is arbitrary, its rotation/basis is set by the encoder
training seed, and L2 in latent space is not comparable across runs,
checkpoints, arms, or encoder retrainings. Instead, both the prediction and
the target are decoded through the frozen scripted GEN3 AttentionSE decoder
(`encoder/autoencoderGEN3/saved_models_production/`
`Model_GEN3_05_AttentionSE_absolute_best_scripted.pt`) and scored on the
central velocity triplet `(vx, vy, vz)` at index 62 of 125 spatial points
(slice `[186:189]` of the 375-dim reconstruction). Gradients flow through
the decoder into the transformer -- the decoder's parameters are frozen
(`requires_grad=False`) so nothing in the encoder is trained by this loss.

The retirement is intentionally auditable: every latent-space error metric
that was replaced carries a comment block beginning with the sentinel
`# LATENT-SPACE ERROR RETIRED`. Grep for that string to find every disabled
call site. The informational latent floor from `null_baselines()` and the
`linear_frame_baseline()` diagnostic still use `l2_loss` / `mse_loss` /
`base_loss` -- they are LATENT-space sanity anchors, not the training
target. See `transformer_neurIPS/OVERVIEW.md` §10.9.7 for the rationale,
centroid-index derivation, and the console/wandb schema changes.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from model_variants import get_model, seq_to_frames, FRAME_META_COLS
except ImportError:
    from transformer_neurIPS.model_variants import get_model, seq_to_frames, FRAME_META_COLS


# --------------------------------------------------------------------------- #
# Console color (ANSI) + device-adaptive training regime
# --------------------------------------------------------------------------- #
# Copied verbatim from persistence_formal_documentation.py to avoid taking a
# hard dependency on that module (it does file I/O and reads env vars at
# import time). Step 6 of the migration plan owns that file; if the helpers
# diverge, sync them back there.
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


def pick_device():
    """Prefer cuda > mps > cpu. Returns a torch.device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_detection_banner(requested=None):
    """Print, as the FIRST thing this script does on any invocation path
    (train / --smoke-test / --diagnostics-only / --list-arms), which compute
    device this run resolved to. `resolve_train_regime()`'s banner (printed
    later, only on the train/smoke paths) covers the detailed micro-batch
    regime; this one line exists so a run's device is never a mystery you
    have to scroll for.
    """
    device = torch.device(requested) if requested else pick_device()
    if device.type == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown GPU"
        print(_bold(f"[DEVICE DETECTED] CUDA — {name}", "green"), flush=True)
    elif device.type == "mps":
        print(_rainbow("[DEVICE DETECTED] MPS (Apple Silicon)"), flush=True)
    else:
        print(_bold("[DEVICE DETECTED] CPU (no CUDA/MPS available)", "yellow"),
              flush=True)
    return device


@dataclass
class TrainRegime:
    device: str
    micro_batch: int
    virtual_batch: int
    eval_micro_batch: int
    aux_micro_batch: int
    disable_ar: bool
    use_amp: bool
    amp_dtype: Any
    compile_model: bool
    cudnn_benchmark: bool
    banner: str


def _regime_banner_mps_cpu(micro_batch):
    header = _rainbow(f"🌈 MICRO-BATCH MODE (micro_batch={micro_batch})")
    why = ("WHY: on MPS/CPU the caching allocator cannot reuse blocks across "
           "the changing shapes of an autoregressive rollout, so peak memory "
           "scales with the batch inside ONE rollout call and blows past the "
           "device ceiling on the very first batch. micro_batch=1 keeps each "
           "forward within budget (same rationale documented in "
           "persistence_formal_documentation.py).")
    what = ("WHAT CHANGES ON CUDA: micro_batch bumps to 32 (64 on H200), "
            "gradient accumulation collapses to 1, AMP bf16 turns on, "
            "torch.compile is attempted, cudnn.benchmark is enabled.")
    return "\n".join([header, why, what])


def _regime_banner_cuda(micro_batch, virtual_batch):
    header = _bold("[CUDA DETECTED — H200 DEFAULTS ACTIVE]", "green")
    diff = [
        ("batch size",         str(micro_batch),  "1"),
        ("eval batch",         str(micro_batch),  "1"),
        ("AR/aux batch",       str(micro_batch),  "1"),
        ("AR aux loss",        "enabled",          "DISABLED"),
        ("AMP dtype",          "bfloat16",         "off"),
        ("torch.compile",      "on",               "off"),
        ("cudnn.benchmark",    "on",               "off"),
        ("grad accumulation",  str(max(1, virtual_batch // micro_batch)), "32"),
    ]
    lines = [header, "  {:<22}  {:<20}  {:<20}".format(
        "setting", _bold("CUDA", "green"), _c("MPS/CPU", "dim"))]
    for label, cuda_val, mps_val in diff:
        lines.append("  {:<22}  {:<20}  {:<20}".format(
            label, _bold(cuda_val, "green"), _c(mps_val, "dim")))
    return "\n".join(lines)


def resolve_train_regime(device):
    """Return the device-adaptive `TrainRegime` and apply the CUDA-only
    matmul-precision / cudnn.benchmark side effects when applicable.
    """
    dev_str = device.type if hasattr(device, "type") else str(device)
    dev_str = dev_str.split(":")[0]

    if dev_str == "cuda":
        try:
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception:
            name = ""
        micro = 64 if "H200" in name.upper() else 32
        virtual = micro                       # accumulation off
        banner = _regime_banner_cuda(micro, virtual)
        # CUDA-only side effects, guarded so this stays safe under a mocked
        # torch.cuda.is_available() in the tests.
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        return TrainRegime(
            device="cuda", micro_batch=micro, virtual_batch=virtual,
            eval_micro_batch=micro, aux_micro_batch=micro,
            disable_ar=False,
            use_amp=True, amp_dtype=torch.bfloat16,
            compile_model=True, cudnn_benchmark=True, banner=banner)

    # MPS / CPU: eval also runs singleton. The AR rollout at NUM_TIME=80 grows
    # the sequence to SEQ_LEN=800 tokens (v3.1: NUM_X cut 26->10, see
    # OVERVIEW.md §12.1); attention scores are (batch * n_heads * L^2 * 4B).
    # The batch=1 clamp below predates that cut and was sized for the
    # original L=2080 worst case -- still the safe conservative choice at
    # L=800, just with more memory headroom than strictly required now.
    banner = _regime_banner_mps_cpu(micro_batch=1)
    # AR / scheduled-sampling losses also clamp to singleton on MPS/CPU: the AR
    # loop is sequential and each intermediate forward keeps its full activation
    # graph for backward, so peak memory scales linearly with AR_SEQS. This
    # clamp predates the v3.1 NUM_X cut (26->10) and was sized against the
    # original L=2080 worst case; it is disabled outright below rather than
    # re-tuned for the smaller L=800, since the OOM this guards against was
    # only ever observed/characterized at the old shape.
    # AR aux loss is DISABLED on MPS/CPU. Even at AR_SEQS=1 the sequential
    # AR loop does `AR_FRAMES * NUM_X` forwards under token tokenization
    # (4*10 = 40 at the default arm's AR_FRAMES=2, post-v3.1 NUM_X=10; was
    # 4*26=104 pre-v3.1), each retaining its full activation graph for
    # backward through `preds` (the `.detach()` only truncates the fed-back
    # token, not the graph of the forward itself). The primary next-token
    # loss still trains the model on MPS; the AR loss is a
    # rollout-stabilization aux and is CUDA-only in v2.0 (kept on CUDA at the
    # arm-specified AR_SEQS).
    return TrainRegime(
        device=dev_str, micro_batch=1, virtual_batch=32,
        eval_micro_batch=1, aux_micro_batch=1,
        disable_ar=True,
        use_amp=False, amp_dtype=None,
        compile_model=False, cudnn_benchmark=False, banner=banner)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Decoded-centroid loss constants
# --------------------------------------------------------------------------- #
# The frozen GEN3 AttentionSE decoder reconstructs a 375-dim vector laid out as
# [vx_0, vy_0, vz_0, vx_1, vy_1, vz_1, ..., vx_124, vy_124, vz_124]. The
# central triplet (vx_62, vy_62, vz_62) at slice [186:189] is the physical
# quantity the training loss is scored on. See OVERVIEW.md §10.9.7.
DECODED_DIM = 375
N_TRIPLETS = 125
CENTROID_TRIPLET_IDX = 62            # 0-based, middle of 125
CENTROID_SLICE = slice(186, 189)     # inclusive/exclusive; gives vx, vy, vz
V_LABELS = ("vx", "vy", "vz")


class Config:
    """Defaults for a single run. Arms (below) override fields on this class.

    NOTE: `Config.LOSS` is now IGNORED at training time -- the training loss is
    fixed to the decoded-centroid L2 via `centroid_velocity_loss`. The field is
    kept because several arms (e.g. `a2_mse`, `d4_delta_mse`) still setattr it
    and their overrides must not raise; those overrides simply have no effect
    on the training objective any more. See OVERVIEW.md §10.9.7.
    """

    # -- data ---------------------------------------------------------------
    # v2.0 pinning: hard-coded 80-frame files. See top-of-file docstring.
    TRAIN_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/train_80.h5")
    VAL_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/val_80.h5")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

    LATENT_DIM = 47      # latent features per (t, x) location
    # v3.1 (prepare_data.py OVERVIEW.md §12.1): x-sweep restricted to
    # |x| <= 20 (10 samples), down from the full 26-sample sensor line.
    # train_80.h5 / val_80.h5 are written at this width -- keep in sync
    # with prepare_data.py's NUM_X/X_COORDS or TransformerDataset's
    # reshape will hard-crash on shape mismatch.
    NUM_X = 10           # x-locations per time frame (v3.1: was 26)
    NUM_TIME = 80        # time frames per sequence (v2.0 pinned; v1.0 was 40)
    SEQ_LEN = NUM_X * NUM_TIME              # 800 tokens (pre-v3.1: 2080; v1.0: 1040)
    # Column layout written by prepare_data.py:
    #   0:47 latents | 47 x | 48 y | 49 z | 50 t_index | 51 param
    INPUT_DIM = 52

    TRAIN_SUBSET_RATIO = 1.0   # was 0.5; half the data was going unused

    # -- architecture -------------------------------------------------------
    EMBED_SIZE = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.01
    BIAS = True
    VARIANT = 'base'           # base | swiglu | mqa | conv
    USE_SWIGLU = False         # set implicitly by VARIANT='swiglu'
    TOKENIZATION = 'token'     # token (NUM_TIME*NUM_X tokens) | frame (NUM_TIME tokens)
    ATTN_IMPL = 'sdpa'         # sdpa (causal by construction) | mha_hint (old, leaky)
    USE_ROPE = False
    PREDICT_DELTA = False
    NORMALIZE_FEATURES = True
    USE_META_COLS = True       # False zeroes the (x, y, z, t, param) input columns

    # -- optimisation -------------------------------------------------------
    DEVICE = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    BATCH_SIZE = 64            # physical micro-batch
    ACCUMULATION_STEPS = 8     # effective batch = 512
    EVAL_BATCH_SIZE = 128      # forward-only; kept modest so 5 arms can share a GPU
    LEARNING_RATE = 1e-3       # peak LR
    WARMUP_FRAC = 0.03         # fraction of the step budget spent warming up
    LR_FINAL_FRAC = 0.02       # cosine floor, as a fraction of peak
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    ADAM_BETAS = (0.9, 0.95)
    LOSS = 'l2norm'            # l2norm | mse | huber
    HUBER_DELTA = 0.01
    MAX_STEPS = 600_000           # OPTIMIZER steps -- the primary clock
    MAX_HOURS = 1200.0           # wall-clock safety net only

    # -- rollout-stability techniques --------------------------------------
    NOISE_STD = 5e-4           # gaussian noise on fed-in latents
    AR_MODE = 'none'           # none | frame_ar | sched
    AR_LOSS_WEIGHT = 0.0
    AR_WEIGHT_WARMUP_FRAC = 0.2   # ramp AR weight 0 -> AR_LOSS_WEIGHT over this fraction
    AR_FRAMES = 2              # horizon in whole TIME FRAMES (10 tokens each, v3.1)
    AR_SEQS = 4                # sequences used for the sequential AR loop
    AR_EVERY_N_STEPS = 4
    AR_DETACH_FEEDBACK = True  # truncate gradient through the fed-back token
    SCHED_SAMPLING_P = 0.25    # replacement probability when AR_MODE='sched'

    # -- evaluation ---------------------------------------------------------
    VAL_CONTEXT_STEPS = 12                      # frames fed as context
    VAL_ROLLOUT_STEPS = NUM_X * (NUM_TIME - VAL_CONTEXT_STEPS)   # 1768 tokens (v1.0: 728)
    VAL_ROLLOUT_SEQS = 64      # fixed row set; model AND persistence both use it
    VAL_EVERY_STEPS = 25
    LOG_EVERY_STEPS = 25
    CHECKPOINT_EVERY_STEPS = 25

    # -- runtime ------------------------------------------------------------
    USE_TF32 = True
    USE_CUDNN_BENCHMARK = True
    AMP = True
    # TorchScript companion saves. When True, every call to `save_checkpoint`
    # writes a self-contained `<name>_scripted.pt` alongside the plain
    # state-dict `<name>.pt`. See `save_scripted_model()` and the SCRIPTED
    # MODEL SAVES section of this module's docstring for the full contract
    # (torch.compile `_orig_mod` unwrap, script->trace fallback, buffer
    # verification, roundtrip check).
    SAVE_SCRIPTED_MODELS = True

    # -- decoded-centroid training loss (see OVERVIEW.md §10.9.7) -----------
    # Frozen scripted GEN3 AttentionSE decoder used to map the transformer's
    # 47-dim latent output to a 375-dim reconstruction, of which the central
    # triplet (vx_62, vy_62, vz_62) is the training target. The env var
    # PFD_DECODER_PATH overrides this at load time (mirrors PFD_NO_COLOR).
    DECODER_SCRIPTED_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "encoder/autoencoderGEN3/saved_models_production",
        "Model_GEN3_05_AttentionSE_absolute_best_scripted.pt")
    CENTROID_WEIGHTS = (1.0, 1.0, 1.0)   # (w_vx, w_vy, w_vz); future emphasis knob
    CENTROID_LOSS = 'l2'                  # 'l2' = mean of vector-L2-norms;
                                          # 'mse' = mean of squared components
    SEED = 1337
    ARM = 'a0_control'
    SWEEP_ROUND = 1
    WANDB_PROJECT = "NI_Review"


# These fields define the v2.0 data/model shape and are deliberately not
# configurable through an arm or the generic --set escape hatch.
PINNED_CONFIG_FIELDS = frozenset({
    "LATENT_DIM", "NUM_X", "NUM_TIME", "SEQ_LEN", "INPUT_DIM",
    "TRAIN_H5", "VAL_H5", "VAL_ROLLOUT_STEPS",
})


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
# Round 1 is a DIAGNOSTIC round, not a performance round. Each arm changes ONE
# thing relative to a0_control so the result pattern identifies the binding
# constraint. Sweeping EMBED_SIZE/N_LAYERS/VARIANT (the old SEARCH_SPACE) varies
# the axis least likely to be binding: at ~4.8M parameters on ~15k sequences the
# model is far more likely to be over- than under-parameterised, and the control
# arm's own train-vs-val gap answers that for free without spending an arm on it.
ROUND1_ARMS = {
    "a0_control": {
        "desc": "All bug fixes, otherwise the original hyperparameters.",
        "hypothesis": "The causality leak was the whole story; nothing else needs to change.",
        "reads_as": "Reference point. Every other arm is judged against this.",
        "overrides": {},
    },
    "a1_nonorm": {
        "desc": "Control with NORMALIZE_FEATURES turned OFF.",
        "hypothesis": "The input-scale mismatch was the binding constraint. Columns "
                      "47:52 (x, y, z, t, param) have std up to ~49 while the latents "
                      "have std ~0.012 -- a ~4000x variance ratio through one "
                      "nn.Linear, which makes the latent signal numerically invisible.",
        "reads_as": "This is the BEFORE picture. control >> a1_nonorm means the scale "
                    "bug was the whole story and the measured gap is its price.",
        "overrides": {"NORMALIZE_FEATURES": False},
    },
    "a2_mse": {
        "desc": "Control with an MSE objective instead of the mean-of-L2-norms, and "
                "LR raised to suit the smaller gradients.",
        "hypothesis": "`l2_loss = mean(norm(diff, dim=-1))` is a different geometry "
                      "from the MSE the model is evaluated on: it weights every token "
                      "by its error MAGNITUDE rather than its square, so a few "
                      "large-error tokens dominate.",
        "reads_as": "Beats control => train the metric you report.",
        "overrides": {"LOSS": "mse", "LEARNING_RATE": 2e-3},
    },
    "a3_delta": {
        "desc": "Control + delta parameterisation (predict the change from the same "
                "x-location one frame earlier; output head zero-initialised).",
        "hypothesis": "The network is burning capacity relearning the identity map. "
                      "Making persistence the zero-output starts the run AT the "
                      "baseline instead of at random.",
        "reads_as": "Beats control by a lot => the problem was parameterisation, and "
                    "every later arm should inherit PREDICT_DELTA.",
        "overrides": {"PREDICT_DELTA": True},
    },
    "a4_frame": {
        "desc": "Frame-level tokenisation: NUM_TIME tokens of 26x47 features instead of "
                "NUM_TIME*NUM_X tokens of 47.",
        "hypothesis": "Token-level flattening makes the objective mostly SPATIAL "
                      "(only every 26th token crosses a time boundary), so capacity "
                      "goes to within-frame continuation rather than the temporal "
                      "dynamics the baseline is scored on.",
        "reads_as": "Beats control => the sequence framing was wrong, which is the "
                    "single largest available win and also makes rollout 26x cheaper.",
        "overrides": {"TOKENIZATION": "frame"},
    },
}

# Round 2 catalogues, one per decision-tree branch. `sweep_deep_dive.py`
# classifies Round 1 and prints the matching set; the reasoning behind each
# branch is in Documentation/deep_dive_decision_tree.md.
ROUND2_ARMS = {
    # ---------------------------------------------------------------- branch S
    "S": {
        "title": "Control already works -- scale and refine",
        "arms": {
            "s1_capacity_xl": {"desc": "E768/L12 at the winning settings.",
                               "overrides": {"EMBED_SIZE": 768, "N_LAYERS": 12, "N_HEADS": 12}},
            "s2_steps_3x": {"desc": "Same model, 3x the step budget, lower cosine floor.",
                            "overrides": {"MAX_STEPS": 18000, "LR_FINAL_FRAC": 0.005}},
            "s3_swiglu": {"desc": "SwiGLU feed-forward at the winning settings.",
                          "overrides": {"VARIANT": "swiglu"}},
            "s4_lr_low": {"desc": "Half the peak LR, longer warmup -- plateau escape by annealing.",
                          "overrides": {"LEARNING_RATE": 5e-4, "WARMUP_FRAC": 0.08}},
            "s5_bigbatch": {"desc": "Effective batch 2048 with LR scaled up.",
                            "overrides": {"ACCUMULATION_STEPS": 32, "LEARNING_RATE": 2e-3}},
        },
    },
    # ---------------------------------------------------------------- branch D
    "D": {
        "title": "Delta parameterisation won -- push on it",
        "arms": {
            "d1_delta_noise": {"desc": "Delta + large input noise.",
                               "overrides": {"PREDICT_DELTA": True, "NOISE_STD": 2e-2}},
            "d2_delta_ar": {"desc": "Delta + frame-aligned AR loss.",
                            "overrides": {"PREDICT_DELTA": True, "AR_MODE": "frame_ar",
                                          "AR_LOSS_WEIGHT": 0.5, "AR_FRAMES": 2}},
            "d3_delta_frame": {"desc": "Delta + frame tokenisation (the two most likely wins together).",
                               "overrides": {"PREDICT_DELTA": True, "TOKENIZATION": "frame"}},
            "d4_delta_mse": {"desc": "Delta with an MSE objective, matching the eval metric.",
                             "overrides": {"PREDICT_DELTA": True, "LOSS": "mse",
                                           "LEARNING_RATE": 2e-3}},
            "d5_delta_big": {"desc": "Delta at E512/L10 -- capacity now buys dynamics, not identity.",
                             "overrides": {"PREDICT_DELTA": True, "EMBED_SIZE": 512,
                                           "N_LAYERS": 10}},
        },
    },
    # ---------------------------------------------------------------- branch E
    "E": {
        "title": "Exposure bias dominates -- attack error accumulation",
        "arms": {
            "e1_noise_big": {"desc": "Noise 1e-1: deliberately past the useful point, to bracket it.",
                             "overrides": {"NOISE_STD": 1e-1}},
            "e2_sched": {"desc": "Scheduled sampling: feed the model its own predictions, 2 forwards, no sequential loop.",
                         "overrides": {"AR_MODE": "sched", "AR_LOSS_WEIGHT": 1.0,
                                       "SCHED_SAMPLING_P": 0.25}},
            "e3_ar_long": {"desc": "AR horizon 8 frames, weight 1.0.",
                           "overrides": {"AR_MODE": "frame_ar", "AR_LOSS_WEIGHT": 1.0,
                                         "AR_FRAMES": 8, "AR_SEQS": 2,
                                         "AR_EVERY_N_STEPS": 8}},
            "e4_noise_plus_ar": {"desc": "Noise and AR loss together.",
                                 "overrides": {"NOISE_STD": 2e-2, "AR_MODE": "frame_ar",
                                               "AR_LOSS_WEIGHT": 0.5, "AR_FRAMES": 4}},
            "e5_delta_noise": {"desc": "Delta parameterisation + noise: shrink the thing that accumulates.",
                               "overrides": {"PREDICT_DELTA": True, "NOISE_STD": 2e-2}},
        },
    },
    # ---------------------------------------------------------------- branch F
    "F": {
        "title": "Frame tokenisation won -- rebuild around NUM_TIME-token sequences",
        "arms": {
            "f1_frame_delta": {"desc": "Frame + delta.",
                               "overrides": {"TOKENIZATION": "frame", "PREDICT_DELTA": True}},
            "f2_frame_deep": {"desc": "Frame + E512/L12 (NUM_TIME tokens is cheap, so spend it on depth).",
                              "overrides": {"TOKENIZATION": "frame", "EMBED_SIZE": 512,
                                            "N_LAYERS": 12}},
            "f3_frame_noise": {"desc": "Frame + noise.",
                               "overrides": {"TOKENIZATION": "frame", "NOISE_STD": 2e-2}},
            "f4_frame_ar": {"desc": "Frame + AR loss over 8 frames (cheap here: 8 forwards).",
                            "overrides": {"TOKENIZATION": "frame", "AR_MODE": "frame_ar",
                                          "AR_LOSS_WEIGHT": 1.0, "AR_FRAMES": 8,
                                          "AR_SEQS": 16}},
            "f5_frame_rope": {"desc": "Frame + RoPE instead of learned absolute time embeddings.",
                              "overrides": {"TOKENIZATION": "frame", "USE_ROPE": True}},
        },
    },
    # ---------------------------------------------------------------- branch A
    "A": {
        "title": "Single-step is good, long horizon collapses -- stabilise the rollout",
        "arms": {
            "a1b_noise_sweep_lo": {"desc": "Noise 5e-3.", "overrides": {"NOISE_STD": 5e-3}},
            "a2b_noise_sweep_hi": {"desc": "Noise 5e-2.", "overrides": {"NOISE_STD": 5e-2}},
            "a3b_delta_ar": {"desc": "Delta + AR loss: shrink per-step error AND train on accumulation.",
                             "overrides": {"PREDICT_DELTA": True, "AR_MODE": "frame_ar",
                                           "AR_LOSS_WEIGHT": 0.5, "AR_FRAMES": 4}},
            "a4b_ar_very_long": {"desc": "AR horizon 14 frames -- half the eval horizon.",
                                 "overrides": {"AR_MODE": "frame_ar", "AR_LOSS_WEIGHT": 1.0,
                                               "AR_FRAMES": 14, "AR_SEQS": 2,
                                               "AR_EVERY_N_STEPS": 16}},
            "a5b_wd_heavy": {"desc": "Weight decay 0.1 + dropout 0.1: damp the amplifying modes.",
                             "overrides": {"WEIGHT_DECAY": 0.1, "DROPOUT": 0.1}},
        },
    },
    # ---------------------------------------------------------------- branch R
    "R": {
        "title": "A linear map beats the transformer -- the model or the framing is broken",
        "arms": {
            "r1_frame": {"desc": "Frame tokenisation: match the linear baseline's own factorisation.",
                         "overrides": {"TOKENIZATION": "frame"}},
            "r2_frame_delta_mse": {"desc": "Frame + delta + MSE -- as close to the linear baseline as a net gets.",
                                   "overrides": {"TOKENIZATION": "frame", "PREDICT_DELTA": True,
                                                 "LOSS": "mse", "LEARNING_RATE": 2e-3}},
            "r3_tiny": {"desc": "Deliberately tiny (E128/L2): if small beats large, this is an optimisation failure.",
                        "overrides": {"EMBED_SIZE": 128, "N_LAYERS": 2, "N_HEADS": 4}},
            "r4_lr_sweep": {"desc": "Peak LR 3e-3 with a long warmup.",
                            "overrides": {"LEARNING_RATE": 3e-3, "WARMUP_FRAC": 0.10}},
            "r5_mse_nonorm": {"desc": "MSE objective, feature normalisation OFF -- isolates the normalisation change.",
                              "overrides": {"LOSS": "mse", "NORMALIZE_FEATURES": False}},
        },
    },
    # ---------------------------------------------------------------- branch T
    "T": {
        "title": "Persistence is near-optimal at this sampling rate -- change the problem",
        "arms": {
            "t1_frame_delta": {"desc": "Frame + delta at maximum sensitivity to small changes.",
                               "overrides": {"TOKENIZATION": "frame", "PREDICT_DELTA": True,
                                             "LOSS": "mse", "LEARNING_RATE": 2e-3}},
            "t2_long_context": {"desc": "Context 4 frames instead of 12: forces a real 36-frame extrapolation "
                                        "and makes persistence a much weaker baseline.",
                                "overrides": {"VAL_CONTEXT_STEPS": 4}},
            "t3_short_context": {"desc": "Context 24 frames: if improvement appears here, the model needs "
                                         "more history than 12 frames.",
                                 "overrides": {"VAL_CONTEXT_STEPS": 24}},
            "t4_mse": {"desc": "Train the exact eval metric.",
                       "overrides": {"LOSS": "mse", "LEARNING_RATE": 2e-3}},
            "t5_huber": {"desc": "Huber: latents may have heavy-tailed outliers that L2 chases.",
                         "overrides": {"LOSS": "huber", "HUBER_DELTA": 0.01}},
        },
    },
    # ---------------------------------------------------------------- branch O
    "O": {
        "title": "No structural arm separated -- grind the optimiser",
        "arms": {
            "o1_lr_low": {"desc": "Peak LR 3e-4.", "overrides": {"LEARNING_RATE": 3e-4}},
            "o2_lr_high": {"desc": "Peak LR 3e-3, long warmup.",
                           "overrides": {"LEARNING_RATE": 3e-3, "WARMUP_FRAC": 0.10}},
            "o3_mse": {"desc": "MSE objective (matches the eval metric; l2norm is a different geometry).",
                       "overrides": {"LOSS": "mse", "LEARNING_RATE": 2e-3}},
            "o4_bigbatch": {"desc": "Effective batch 2048.",
                            "overrides": {"ACCUMULATION_STEPS": 32, "LEARNING_RATE": 2e-3}},
            "o5_long": {"desc": "3x steps at the control settings -- rule out 'not trained long enough'.",
                        "overrides": {"MAX_STEPS": 18000, "LR_FINAL_FRAC": 0.005}},
        },
    },
    # ---------------------------------------------------------------- branch G
    "G": {
        "title": "Overfitting: train loss far below val loss",
        "arms": {
            "g1_dropout": {"desc": "Dropout 0.1.", "overrides": {"DROPOUT": 0.1}},
            "g2_wd": {"desc": "Weight decay 0.1.", "overrides": {"WEIGHT_DECAY": 0.1}},
            "g3_small": {"desc": "E128/L4 -- fewer parameters for ~15k sequences.",
                         "overrides": {"EMBED_SIZE": 128, "N_LAYERS": 4, "N_HEADS": 4}},
            "g4_noise_reg": {"desc": "Noise as a regulariser (2e-2) + dropout 0.05.",
                             "overrides": {"NOISE_STD": 2e-2, "DROPOUT": 0.05}},
            "g5_frame": {"desc": "Frame tokenisation: 26x fewer, harder tokens is itself a capacity cut.",
                         "overrides": {"TOKENIZATION": "frame"}},
        },
    },
    # ---------------------------------------------------------------- branch N
    "N": {
        "title": "Models are far below a trivial baseline -- fix conditioning before anything else",
        "arms": {
            "n1_delta_mse": {"desc": "Delta + MSE + LR 2e-3: the combination that removes both "
                                     "the output-scale and the objective-geometry problems.",
                             "overrides": {"PREDICT_DELTA": True, "LOSS": "mse",
                                           "LEARNING_RATE": 2e-3}},
            "n2_meta_off": {"desc": "Zero the (x, y, z, t, param) input columns entirely -- "
                                    "position still arrives via the embeddings.",
                            "overrides": {"USE_META_COLS": False, "PREDICT_DELTA": True,
                                          "LOSS": "mse"}},
            "n3_lr_low": {"desc": "LR 1e-4: rule out silent divergence at 1e-3.",
                          "overrides": {"LEARNING_RATE": 1e-4, "PREDICT_DELTA": True}},
            "n4_frame_delta": {"desc": "Frame tokenisation + delta + MSE: the smallest, most "
                                       "directly supervised version of the problem.",
                               "overrides": {"TOKENIZATION": "frame", "PREDICT_DELTA": True,
                                             "LOSS": "mse", "LEARNING_RATE": 2e-3}},
            "n5_tiny": {"desc": "E128/L2 + delta + MSE: if a tiny model works and a big one "
                                "does not, this is an optimisation failure, not capacity.",
                        "overrides": {"EMBED_SIZE": 128, "N_LAYERS": 2, "N_HEADS": 4,
                                      "PREDICT_DELTA": True, "LOSS": "mse",
                                      "LEARNING_RATE": 2e-3}},
        },
    },
    # ---------------------------------------------------------------- branch L
    "L": {
        "title": "Causality probe FAILED -- do not interpret any metric until this is fixed",
        "arms": {},
    },
}


def all_arms(round_no):
    if int(round_no) == 1:
        return ROUND1_ARMS
    merged = {}
    for branch in ROUND2_ARMS.values():
        for name, spec in branch["arms"].items():
            merged[name] = spec
    return merged


def resolve_arm(name):
    """Find an arm spec by name across every round/branch."""
    if name in ROUND1_ARMS:
        return ROUND1_ARMS[name]
    for branch in ROUND2_ARMS.values():
        if name in branch["arms"]:
            return branch["arms"][name]
    raise KeyError(name)


def apply_arm(name):
    spec = resolve_arm(name)
    for k, v in spec.get("overrides", {}).items():
        if not hasattr(Config, k):
            raise AttributeError(
                f"arm {name!r} overrides unknown Config field {k!r} -- typo?")
        if k in PINNED_CONFIG_FIELDS:
            raise ValueError(
                f"arm {name!r} cannot override pinned v2.0 Config field {k!r}")
        setattr(Config, k, v)
    Config.ARM = name
    return spec


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def mse_loss(pred, target):
    """LATENT-space mean-squared-error. Retained for `null_baselines()` and
    the linear-baseline diagnostic ONLY -- no longer used as a training loss.
    See OVERVIEW.md §10.9.7."""
    return torch.mean((pred - target) ** 2)


def l2_loss(pred, target):
    """LATENT-space mean vector-L2 norm. Retained for `null_baselines()` and
    the linear-baseline diagnostic ONLY -- no longer used as a training loss.
    See OVERVIEW.md §10.9.7."""
    return torch.mean(torch.norm(pred - target, dim=-1))


def base_loss(pred, target, cfg=Config):
    """LATENT-space error kernel selected by `cfg.LOSS`.

    Retained for informational latent-space baselines only (`null_baselines()`
    reporting and the previous-frame anchor floor in `train()`). The training
    loss is now `centroid_velocity_loss` and does not read `cfg.LOSS`.
    """
    kind = getattr(cfg, 'LOSS', 'l2norm')
    if kind == 'l2norm':
        return l2_loss(pred, target)
    if kind == 'mse':
        return mse_loss(pred, target)
    if kind == 'huber':
        return F.huber_loss(pred, target, delta=getattr(cfg, 'HUBER_DELTA', 0.01))
    raise ValueError(f"Unknown LOSS {kind!r}; expected 'l2norm', 'mse' or 'huber'")


# --------------------------------------------------------------------------- #
# Decoded-centroid training loss
# --------------------------------------------------------------------------- #
# The 47-dim autoencoder latent is not a physical quantity; the training /
# evaluation error is computed in decoded velocity space instead. Both the
# prediction and the target are decoded through the FROZEN scripted GEN3
# AttentionSE decoder, then scored on the central triplet (vx, vy, vz) at
# CENTROID_TRIPLET_IDX=62 of 125 spatial points. See OVERVIEW.md §10.9.7.
_DECODER_CACHE: dict = {}


def _load_decoder(device, cfg=Config, log=print):
    """Load, cache and freeze the scripted GEN3 AttentionSE decoder.

    Returns a callable `decode_fn(z_47) -> velocity_375` that (a) runs on
    `device`, (b) has `requires_grad=False` on all its parameters, and (c) is
    in `eval()` mode. The callable accepts inputs of any leading shape ending
    in `LATENT_DIM=47`, flattens the leading dims for the decoder forward,
    and reshapes the 375-dim reconstruction back to the original leading
    shape. Rainbow-logged with `[start-from:decoder] <abs path>` on the first
    load per (path, device) key so scrollback pinpoints which decoder file
    the run was scored against.

    IMPORTANT: the callable is intentionally NOT wrapped in `torch.no_grad`
    -- the training loss needs gradient flow through the decoder into the
    transformer. Only the decoder's own parameters are frozen.
    """
    path = os.environ.get("PFD_DECODER_PATH") or cfg.DECODER_SCRIPTED_PATH
    key = (path, str(device))
    entry = _DECODER_CACHE.get(key)
    if entry is None:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"scripted decoder not found: {abs_path} "
                f"(set PFD_DECODER_PATH or Config.DECODER_SCRIPTED_PATH)")
        mod = torch.jit.load(abs_path, map_location=device).eval()
        # Freeze: gradients still FLOW through the decoder for backprop into
        # the transformer, but the decoder's own weights never update.
        for p in mod.parameters():
            p.requires_grad_(False)
        try:
            log(_rainbow(f"[start-from:decoder] {abs_path}"))
        except Exception:
            # `_rainbow` should always work, but never let logging block a load.
            print(f"[start-from:decoder] {abs_path}", flush=True)

        def _decode(z):
            # Flatten leading dims so the scripted decoder sees (N, 47), then
            # restore the leading shape with a trailing DECODED_DIM=375.
            orig_shape = z.shape
            z_flat = z.reshape(-1, orig_shape[-1])
            try:
                out = mod.decode(z_flat)
            except AttributeError:
                # Very unlikely: some scripted archives only expose forward().
                # For the GEN3 AttentionSE BaseAE, forward expects the 375-dim
                # input, so this branch is a diagnostic fallback only.
                out = mod(z_flat)
            return out.reshape(*orig_shape[:-1], DECODED_DIM)

        entry = _decode
        _DECODER_CACHE[key] = entry
    return entry


def decode_centroid(latent, cfg=Config):
    """`(..., 47) -> (..., 3)` via the frozen decoder, sliced at CENTROID_SLICE.

    Assumes the decoder is already loadable onto `latent.device`. Preserves
    all leading dimensions (batch, time, etc.) and returns the central
    velocity triplet `(vx, vy, vz)` at index 62 of 125 spatial points.
    """
    dec = _load_decoder(latent.device, cfg)
    v = dec(latent)                                      # (..., 375)
    return v[..., CENTROID_SLICE]                         # (..., 3)


def centroid_velocity_loss(pred_latent, tgt_latent, cfg=Config):
    """Consistent L2-in-velocity-space training loss.

    Decodes both latents through the frozen GEN3 AttentionSE decoder, takes
    the central triplet (vx, vy, vz) at CENTROID_TRIPLET_IDX=62, applies the
    per-dim weights from `cfg.CENTROID_WEIGHTS`, and returns the mean of the
    per-token L2 norm of the weighted 3-vector error. Gradients flow through
    the decoder (whose weights are frozen). `cfg.CENTROID_LOSS` selects
    between `'l2'` (mean of vector-L2-norms, default) and `'mse'` (mean of
    squared components). See OVERVIEW.md §10.9.7.
    """
    pv = decode_centroid(pred_latent, cfg)               # (..., 3)
    tv = decode_centroid(tgt_latent, cfg)                # (..., 3)
    w = torch.tensor(getattr(cfg, 'CENTROID_WEIGHTS', (1.0, 1.0, 1.0)),
                     device=pv.device, dtype=pv.dtype)
    err = (pv - tv) * w                                   # (..., 3)
    if getattr(cfg, 'CENTROID_LOSS', 'l2') == 'mse':
        return err.pow(2).mean()
    return torch.linalg.vector_norm(err, dim=-1).mean()


def centroid_per_dim_errors(pred_latent, tgt_latent, cfg=Config):
    """Per-dim MAE / RMSE and combined L2 on the decoded centroid.

    Returns a plain dict with keys `mae_vx`, `mae_vy`, `mae_vz`, `rmse_vx`,
    `rmse_vy`, `rmse_vz`, `l2_centroid`. All values are Python floats. Used
    for the console breakdown at every LOG_EVERY_STEPS tick and for the
    per-epoch persistence report.
    """
    pv = decode_centroid(pred_latent, cfg)
    tv = decode_centroid(tgt_latent, cfg)
    d = pv - tv                                           # (..., 3)
    out = {}
    for i, lbl in enumerate(V_LABELS):
        di = d[..., i]
        out[f"mae_{lbl}"] = float(di.abs().mean())
        out[f"rmse_{lbl}"] = float(di.pow(2).mean().sqrt())
    out["l2_centroid"] = float(torch.linalg.vector_norm(d, dim=-1).mean())
    return out


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class TransformerDataset(torch.utils.data.Dataset):
    """The whole split, read from HDF5 once and held as one tensor.

    "Small" no longer describes this literally as of the v3.4 wake-atlas
    rebuild: train is ~59.3k x 800 x 52 float32 ~= 9.2 GiB, val ~25.4k x 800 x
    52 ~= 3.9 GiB (~13 GiB combined resident) -- up almost 10x from the v3
    estimate of ~3.7k sequences / ~1.6 GB, because the physics-derived atlas
    produces far more wake seeds than the old 24-tap list did (OVERVIEW.md
    §15), even though NUM_X shrank 26->10 over the same period. Confirm this
    still fits in RAM and (for CUDA) on the GPU before assuming the
    single-resident-tensor design is still free; there is no reason to
    re-open the HDF5 file per worker per epoch as long as it does.
    """

    def __init__(self, h5_path, subset_ratio=1.0):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        with h5py.File(self.h5_path, 'r') as f:
            total_length = f['data'].shape[0]
            self.length = max(1, int(total_length * subset_ratio)) if total_length else 0
            raw = f['data'][:self.length]
        self.total_available = total_length
        self.data = torch.from_numpy(raw).float().reshape(
            self.length, Config.SEQ_LEN, Config.INPUT_DIM)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx]


class InMemoryBatcher:
    """Yield shuffled slices of one resident tensor. No workers, no copies.

    Going back through torch DataLoader for a tensor that is already on the
    training device costs a collate, a pin_memory staging buffer and a H2D copy
    per micro-batch, which measured as ~90% of epoch wall time.
    """

    def __init__(self, data, batch_size, shuffle, generator=None):
        self.data = data
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.n = int(data.shape[0])
        self.generator = generator

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.data.device, generator=self.generator)
        else:
            idx = torch.arange(self.n, device=self.data.device)
        for start in range(0, self.n, self.batch_size):
            yield self.data.index_select(0, idx[start:start + self.batch_size])


def infinite_batches(batcher):
    """Micro-batch stream that never ends, so the step budget is the only clock."""
    while True:
        for b in batcher:
            yield b


def _preload_to_device(dataset, name, device, log=print):
    n_bytes = dataset.data.numel() * dataset.data.element_size()
    log(f"  [data] '{name}': {tuple(dataset.data.shape)} "
        f"({n_bytes / 1e9:.2f} GB float32) -> {device}")
    if device == "cpu":
        return dataset.data
    t0 = time.time()
    try:
        dataset.data = dataset.data.to(device, non_blocking=False)
        if device == "cuda":
            torch.cuda.synchronize()
        log(f"  [data]   resident on {device} after {time.time() - t0:.1f}s")
    except (RuntimeError, MemoryError) as e:
        log(f"  [data]   {device} preload FAILED ({type(e).__name__}: {e}); staying on CPU")
        if device == "cuda":
            torch.cuda.empty_cache()
    return dataset.data


def compute_feature_stats(data, cfg, frame_level, chunk=512, max_seqs=4096):
    """Per-feature mean/std of the training inputs, in float64.

    Measured on the TRAINING split only and stored in the checkpoint, so
    evaluation never has to recompute (or guess) them.

    Accumulated on the CPU in float64: the sums run over millions of rows, the
    x/y/z columns are O(50) while the latents are O(0.01), and MPS has no
    float64 at all. This is a one-time pass, so host-side is free.
    """
    n = min(data.shape[0], max_seqs)
    total = 0
    s1 = s2 = None
    for start in range(0, n, chunk):
        x = data[start:start + chunk]
        if frame_level:
            x = seq_to_frames(x, cfg.NUM_X, cfg.LATENT_DIM)
        x = x.reshape(-1, x.shape[-1]).cpu().double()
        if s1 is None:
            s1 = torch.zeros(x.shape[-1], dtype=torch.float64)
            s2 = torch.zeros_like(s1)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
        total += x.shape[0]
    mean = s1 / total
    var = (s2 / total - mean * mean).clamp_min(0.0)
    return mean.float(), var.sqrt().float()


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_causality(model, cfg, device, tol_rel=1e-4, tol_abs=1e-5):
    """Prove the model cannot see the future.

    Perturb every input position at and after a cut, then check that outputs
    BEFORE the cut are bit-for-bit (up to fp32 noise) unchanged. This is the
    only trustworthy way to establish causality: `is_causal` means different
    things in the module and functional APIs, and one of them is a hint that can
    be silently dropped.
    """
    was_training = model.training
    model.eval()
    frame_native = bool(getattr(model, 'frame_native', False))

    if frame_native:
        n_pos, cut = 12, 6
        width = cfg.NUM_X * cfg.LATENT_DIM + FRAME_META_COLS
    else:
        n_pos, cut = 8 * cfg.NUM_X, 4 * cfg.NUM_X
        width = cfg.INPUT_DIM

    x = torch.randn(2, n_pos, width, device=device)
    a = model(x)
    x2 = x.clone()
    x2[:, cut:] = x2[:, cut:] + 7.0          # perturb ONLY the future
    b = model(x2)

    scale = a.abs().max().item()
    before = (a[:, :cut] - b[:, :cut]).abs().max().item()
    after = (a[:, cut:] - b[:, cut:]).abs().max().item()
    tol = max(tol_abs, tol_rel * scale)

    if was_training:
        model.train()
    return {
        "cut_position": cut,
        "positions": n_pos,
        "output_scale": scale,
        "max_change_before_cut": before,
        "max_change_after_cut": after,
        "tolerance": tol,
        "causal": bool(before <= tol),
        # A model whose future outputs did NOT move is a different failure: the
        # probe itself is not exercising the model.
        "probe_responsive": bool(after > tol),
    }


@torch.no_grad()
def probe_symmetric_conv_leak(device):
    """Quantify the OLD ConvBlock padding bug in isolation.

    `nn.Conv1d(kernel_size=3, padding=1)` centres the kernel, so output t sees
    t+1. Ten lines here turn 'this was a bug' into a logged number.
    """
    c = 4
    x = torch.randn(1, c, 16, device=device)
    x2 = x.clone()
    x2[:, :, 8:] += 5.0
    out = {}
    for label, pad_left, pad_right in (("symmetric_padding_1 (old)", 1, 1),
                                       ("left_padding_2 (fixed)", 2, 0)):
        conv = nn.Conv1d(c, c, kernel_size=3, padding=0, groups=c).to(device)
        a = conv(F.pad(x, (pad_left, pad_right)))
        b = conv(F.pad(x2, (pad_left, pad_right)))
        out[label] = (a[:, :, :8] - b[:, :, :8]).abs().max().item()
    return out


@torch.no_grad()
def probe_legacy_attention(cfg, device):
    """Was `nn.MultiheadAttention(attn_mask=None, is_causal=True)` actually causal?

    Answering this on the box removes the last blind spot: depending on the torch
    build it either raises, silently ignores the hint (leak), or honours it.
    """
    result = {"outcome": None}
    try:
        mha = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True).to(device).train()
        x = torch.randn(1, 8, 32, device=device)
        a, _ = mha(x, x, x, attn_mask=None, is_causal=True, need_weights=False)
        x2 = x.clone()
        x2[:, 5:] += 10.0
        b, _ = mha(x2, x2, x2, attn_mask=None, is_causal=True, need_weights=False)
        delta = (a[:, :5] - b[:, :5]).abs().max().item()
        result.update({
            "outcome": "ran",
            "max_change_before_cut": delta,
            "causal": bool(delta <= 1e-5),
        })
    except Exception as e:
        result.update({"outcome": "raised", "error": f"{type(e).__name__}: {e}"})
    return result


@torch.no_grad()
def null_baselines(data, cfg, frame_level=False, max_seqs=256):
    """Informational LATENT-SPACE floor only.

    The training loss is now the decoded centroid L2 (see
    `centroid_velocity_loss`); these numbers are retained as a sanity check
    on data variance and are NOT the target the model is trained against.
    Kept because the `<-- WORSE THAN PREDICTING ZERO` messaging in `train()`
    and the diagnostics-only `run_diagnostics` reporting both rely on them.

    What trivial predictors score on the TEACHER-FORCED training objective.

    This is the sanity floor, and it is the check whose absence let a badly
    broken run look like a plateau. Measured on the exact quantity the trainer
    minimises, so `train_loss` can be compared against it directly:

        predict all zeros      -- the null model
        predict the mean       -- the best constant
        copy previous TOKEN    -- spatial persistence (tokens are x-minor, so
                                  consecutive tokens are usually x-neighbours)
        copy previous FRAME    -- temporal persistence at the same x; this is
                                  what PREDICT_DELTA makes the network's
                                  zero-output, and what the eval baseline uses

    A train loss ABOVE the zero-predictor means the model has learned nothing
    and the problem is conditioning or optimisation, not capacity, not exposure
    bias and not architecture.
    """
    n = min(data.shape[0], max_seqs)
    x = data[:n]
    if frame_level:
        f = seq_to_frames(x, cfg.NUM_X, cfg.LATENT_DIM)
        width = cfg.NUM_X * cfg.LATENT_DIM
        tgt = f[:, 1:, :width]
        cands = {"zeros": torch.zeros_like(tgt),
                 "mean": f[:, :, :width].mean(dim=(0, 1), keepdim=True).expand_as(tgt),
                 "previous frame": f[:, :-1, :width]}
    else:
        lat = x[..., :cfg.LATENT_DIM]
        tgt = lat[:, 1:, :]
        T = tgt.shape[1]
        k = cfg.NUM_X - 1
        cands = {"zeros": torch.zeros_like(tgt),
                 "mean": lat.mean(dim=(0, 1), keepdim=True).expand_as(tgt),
                 "previous token": lat[:, :-1, :],
                 "previous frame": torch.cat([lat[:, :k, :], lat[:, :T - k, :]], 1)}

    out = {}
    for name, pred in cands.items():
        # All three objectives, so the comparison against `train_loss` is
        # like-for-like whichever LOSS the arm is using.
        out[name] = {
            "l2norm": float(l2_loss(pred, tgt)),
            "mse": float(mse_loss(pred, tgt)),
            "huber": float(F.huber_loss(pred, tgt,
                                        delta=getattr(cfg, 'HUBER_DELTA', 0.01))),
        }
    out["_target_std"] = float(tgt.std())
    out["_sequences"] = n
    return out


@torch.no_grad()
def linear_frame_baseline(train_data, val_data, cfg, device, ridge=1e-3,
                          max_train_seqs=4000, chunk=128, log=print):
    """Ridge-fit frame(t) -> frame(t+1) linear map, rolled out like the model.

    This is the anchor that tells you whether the TASK has learnable temporal
    structure beyond persistence:

      * linear >> persistence but transformer ~= persistence  -> the transformer
        is broken, not the problem.
      * linear ~= persistence too                             -> persistence is
        genuinely strong at this dt; the framing needs to change, not the model.

    Solved on the normal equations in float64. D = NX*LATENT_DIM = 10*47 = 470
    (post-v3.1; was 26*47=1222), so X'X is 471x471 -- trivial regardless of
    how many transitions are accumulated.
    """
    NX, LD, NT = cfg.NUM_X, cfg.LATENT_DIM, cfg.NUM_TIME
    D = NX * LD
    # MPS has no float64 at all, and accumulating a 1223x1223 Gram matrix over
    # ~500k rows in float32 loses enough precision to change the fit. Do the
    # whole least-squares problem in float64 on a device that supports it; it is
    # a one-time diagnostic, so the transfer cost is irrelevant.
    solve_device = 'cpu' if str(device).startswith('mps') else device
    XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=solve_device)
    XtY = torch.zeros(D + 1, D, dtype=torch.float64, device=solve_device)

    n_fit = min(train_data.shape[0], max_train_seqs)
    rows = 0
    for start in range(0, n_fit, chunk):
        b = train_data[start:start + chunk].to(solve_device, non_blocking=True)
        f = b[..., :LD].reshape(b.shape[0], NT, D)
        X = f[:, :-1].reshape(-1, D).double()
        Y = f[:, 1:].reshape(-1, D).double()
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64,
                                      device=solve_device)], 1)
        XtX += X1.T @ X1
        XtY += X1.T @ Y
        rows += X.shape[0]

    reg = torch.eye(D + 1, dtype=torch.float64, device=solve_device) * ridge * (rows / (D + 1))
    reg[-1, -1] = 0.0                          # never penalise the intercept
    A = torch.linalg.solve(XtX + reg, XtY)     # (D+1, D)
    log(f"  [linear] fit on {rows:,} frame transitions from {n_fit:,} sequences")

    ctx = cfg.VAL_CONTEXT_STEPS
    n_frames = NT - ctx
    se_lin = torch.zeros(n_frames, dtype=torch.float64)
    se_pers = torch.zeros(n_frames, dtype=torch.float64)
    count = 0
    for start in range(0, val_data.shape[0], chunk):
        b = val_data[start:start + chunk].to(solve_device, non_blocking=True)
        B = b.shape[0]
        f = b[..., :LD].reshape(B, NT, D).double()
        cur = f[:, ctx - 1]                                   # last context frame
        anchor = cur.clone()
        preds = []
        for _ in range(n_frames):
            cur = torch.cat([cur, torch.ones(B, 1, dtype=torch.float64,
                                             device=solve_device)], 1) @ A
            preds.append(cur.unsqueeze(1))
        preds = torch.cat(preds, 1)                           # (B, n_frames, D)
        true = f[:, ctx:]
        pers = anchor.unsqueeze(1).expand(-1, n_frames, -1)
        se_lin += ((preds - true) ** 2).mean(-1).sum(0).cpu()
        se_pers += ((pers - true) ** 2).mean(-1).sum(0).cpu()
        count += B

    mse_lin = (se_lin / count)
    mse_pers = (se_pers / count)
    imp = ((mse_pers - mse_lin) / (mse_pers + 1e-12) * 100)
    return {
        "fit_transitions": rows,
        "val_sequences": count,
        "linear_mse": float(mse_lin.mean()),
        "persistence_mse": float(mse_pers.mean()),
        "improvement_pct": float((mse_pers.mean() - mse_lin.mean()) / (mse_pers.mean() + 1e-12) * 100),
        "improvement_pct_frame1": float(imp[0]),
        "improvement_pct_per_frame": [round(float(v), 3) for v in imp],
    }


# --------------------------------------------------------------------------- #
# Forward / rollout helpers
# --------------------------------------------------------------------------- #
def _add_latent_noise(x, std, width, generator=None):
    if std <= 0:
        return x
    x = x.clone()
    noise = torch.randn(x[..., :width].shape, device=x.device, dtype=x.dtype,
                        generator=generator) * std
    x[..., :width] = x[..., :width] + noise
    return x


def teacher_forced(model, batch, cfg, noise_std=0.0, generator=None):
    """Returns (prediction, target) for the next-step objective.

    Token model: position t predicts token t+1's latents.
    Frame model: position f predicts frame f+1's flattened latents.
    """
    if getattr(model, 'frame_native', False):
        frames = seq_to_frames(batch, cfg.NUM_X, cfg.LATENT_DIM)
        width = cfg.NUM_X * cfg.LATENT_DIM
        inp = _add_latent_noise(frames[:, :-1, :], noise_std, width, generator)
        return model(inp), frames[:, 1:, :width]
    inp = _add_latent_noise(batch[:, :-1, :], noise_std, cfg.LATENT_DIM, generator)
    return model(inp), batch[:, 1:, :cfg.LATENT_DIM]


def rollout_frames(model, batch, cfg, ctx_frames=None, n_frames=None):
    """Autoregressive rollout, returned as (B, n_frames, NUM_X, LATENT_DIM).

    Both tokenizations produce the same shape so `evaluate()` is shared. The
    non-latent columns of each fed-back step (x, y, z, t, param) are taken from
    ground truth, exactly as at inference time where they are known.
    """
    NX, LD = cfg.NUM_X, cfg.LATENT_DIM
    ctx_frames = cfg.VAL_CONTEXT_STEPS if ctx_frames is None else ctx_frames
    n_frames = (cfg.NUM_TIME - ctx_frames) if n_frames is None else n_frames
    B = batch.shape[0]

    if getattr(model, 'frame_native', False):
        frames = seq_to_frames(batch, NX, LD)
        width = NX * LD
        curr = frames[:, :ctx_frames, :]
        preds = []
        for _ in range(n_frames):
            nxt = model(curr)[:, -1:, :]
            preds.append(nxt)
            gi = curr.shape[1]
            if gi >= frames.shape[1]:
                break
            nf = frames[:, gi:gi + 1, :].clone()
            nf[:, :, :width] = nxt
            curr = torch.cat([curr, nf], dim=1)
        out = torch.cat(preds, dim=1)                       # (B, n, width)
        return out.reshape(B, out.shape[1], NX, LD)

    ctx_len = ctx_frames * NX
    horizon = n_frames * NX
    curr = batch[:, :ctx_len, :]
    preds = []
    for _ in range(horizon):
        nxt = model(curr)[:, -1:, :]
        preds.append(nxt)
        gi = curr.shape[1]
        if gi >= batch.shape[1]:
            break
        tok = batch[:, gi:gi + 1, :].clone()
        tok[:, :, :LD] = nxt
        curr = torch.cat([curr, tok], dim=1)
    out = torch.cat(preds, dim=1)                           # (B, horizon, LD)
    n_done = out.shape[1] // NX
    return out[:, :n_done * NX, :].reshape(B, n_done, NX, LD)


def frame_ar_loss(model, batch, cfg, generator=None):
    """Frame-aligned multi-step autoregressive loss.

    The context is a whole number of frames and the horizon is a whole number of
    frames, so this trains TEMPORAL extrapolation. The previous version used
    `ar_context_len=128` (= 4*26 + 24, mid-frame) with a 5-token horizon, which
    never left the frame it started in.

    The starting frame is drawn at random each call so the model does not learn
    to extrapolate from one fixed anchor. Feedback is detached by default:
    gradients still flow through each individual prediction, but not through the
    whole chain, which keeps activation memory bounded at long horizons.
    """
    NX, LD, NT = cfg.NUM_X, cfg.LATENT_DIM, cfg.NUM_TIME
    n_fr = int(cfg.AR_FRAMES)
    seqs = batch[:int(cfg.AR_SEQS)]
    if seqs.shape[0] == 0 or n_fr < 1:
        return None

    max_ctx = NT - n_fr
    if max_ctx < 2:
        return None
    lo = min(4, max_ctx)
    ctx_frames = int(torch.randint(lo, max_ctx + 1, (1,), generator=generator,
                                   device='cpu').item())
    detach = bool(getattr(cfg, 'AR_DETACH_FEEDBACK', True))

    if getattr(model, 'frame_native', False):
        frames = seq_to_frames(seqs, NX, LD)
        width = NX * LD
        curr = frames[:, :ctx_frames, :]
        preds = []
        for i in range(n_fr):
            nxt = model(curr)[:, -1:, :]
            preds.append(nxt)
            nf = frames[:, ctx_frames + i:ctx_frames + i + 1, :].clone()
            nf[:, :, :width] = nxt.detach() if detach else nxt
            curr = torch.cat([curr, nf], dim=1)
        # LATENT-SPACE ERROR RETIRED -- the 47-dim autoencoder latent is NOT a
        # physical quantity: its per-dimension scale is arbitrary, its
        # rotation/basis is set by the encoder training seed, and L2 in
        # latent space is not comparable across runs, checkpoints, arms, or
        # encoder retrainings. We now decode both the prediction and the
        # target through the frozen GEN3 AttentionSE decoder and score on
        # the central triplet (vx, vy, vz) at index 62 of 125 spatial
        # points. See OVERVIEW.md §10.9.7 for the rationale and centroid-
        # index derivation.
        return centroid_velocity_loss(
            torch.cat(preds, 1),
            frames[:, ctx_frames:ctx_frames + n_fr, :width], cfg)

    ctx_len = ctx_frames * NX
    horizon = n_fr * NX
    curr = seqs[:, :ctx_len, :]
    preds = []
    for i in range(horizon):
        nxt = model(curr)[:, -1:, :]
        preds.append(nxt)
        tok = seqs[:, ctx_len + i:ctx_len + i + 1, :].clone()
        tok[:, :, :LD] = nxt.detach() if detach else nxt
        curr = torch.cat([curr, tok], dim=1)
    # LATENT-SPACE ERROR RETIRED -- the 47-dim autoencoder latent is NOT a
    # physical quantity: its per-dimension scale is arbitrary, its
    # rotation/basis is set by the encoder training seed, and L2 in
    # latent space is not comparable across runs, checkpoints, arms, or
    # encoder retrainings. We now decode both the prediction and the
    # target through the frozen GEN3 AttentionSE decoder and score on
    # the central triplet (vx, vy, vz) at index 62 of 125 spatial
    # points. See OVERVIEW.md §10.9.7 for the rationale and centroid-
    # index derivation.
    return centroid_velocity_loss(
        torch.cat(preds, 1),
        seqs[:, ctx_len:ctx_len + horizon, :LD], cfg)


def sched_sampling_loss(model, batch, cfg, p, generator=None):
    """Scheduled sampling in two parallel forwards -- no sequential loop.

    Pass 1 (no grad) gets the model's own one-step predictions; pass 2 replaces a
    random fraction `p` of input latents with them and takes the normal
    teacher-forced loss. Attacks exposure bias at full batch size, unlike the
    sequential AR loss which can only afford a handful of sequences.
    """
    frame_native = bool(getattr(model, 'frame_native', False))
    if frame_native:
        frames = seq_to_frames(batch, cfg.NUM_X, cfg.LATENT_DIM)
        width = cfg.NUM_X * cfg.LATENT_DIM
        inp, tgt = frames[:, :-1, :], frames[:, 1:, :width]
    else:
        width = cfg.LATENT_DIM
        inp, tgt = batch[:, :-1, :], batch[:, 1:, :width]

    with torch.no_grad():
        own = model(inp)
    # own[t] is the prediction of position t+1, so the replacement for input
    # position t is own[t-1]. Position 0 has no predecessor; keep ground truth.
    repl = torch.cat([inp[:, :1, :width], own[:, :-1, :]], dim=1)
    keep = torch.rand(inp.shape[0], inp.shape[1], 1, device=inp.device,
                      generator=generator) >= p
    inp = inp.clone()
    inp[..., :width] = torch.where(keep, inp[..., :width], repl.to(inp.dtype))
    # LATENT-SPACE ERROR RETIRED -- the 47-dim autoencoder latent is NOT a
    # physical quantity: its per-dimension scale is arbitrary, its
    # rotation/basis is set by the encoder training seed, and L2 in latent
    # space is not comparable across runs, checkpoints, arms, or encoder
    # retrainings. We now decode both the prediction and the target through
    # the frozen GEN3 AttentionSE decoder and score on the central triplet
    # (vx, vy, vz) at index 62 of 125 spatial points. See OVERVIEW.md
    # §10.9.7 for the rationale and centroid-index derivation.
    return centroid_velocity_loss(model(inp), tgt, cfg)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, val_data, cfg, device, amp_dtype=None, chunk=32,
             tf_batch_size=None):
    """Teacher-forced loss over all of val + rollout vs persistence.

    The rollout scores the model AND persistence on the SAME fixed, unshuffled
    first `VAL_ROLLOUT_SEQS` rows. The previous version compared 512 persistence
    sequences against 8 model sequences, which made the headline improvement
    number partly an artefact of sample size.
    """
    model.eval()
    NX, LD = cfg.NUM_X, cfg.LATENT_DIM
    ctx = cfg.VAL_CONTEXT_STEPS
    n_frames = cfg.NUM_TIME - ctx

    def _autocast():
        if amp_dtype is None:
            return torch.autocast(device_type='cpu', enabled=False)
        return torch.autocast(device_type='cuda', dtype=amp_dtype)

    # -- teacher forced, whole validation split ----------------------------
    # On MPS the caller passes tf_batch_size=1 (via regime.eval_micro_batch)
    # because a 128-batch forward at SEQ_LEN-1=799 tokens (pre-v3.1: 2079)
    # costs proportionally large attention-score memory. On CUDA
    # cfg.EVAL_BATCH_SIZE (128) is used as-is.
    tf_bs = int(tf_batch_size) if tf_batch_size else int(cfg.EVAL_BATCH_SIZE)
    tf_loss = tf_mse = 0.0
    tf_n = 0
    for start in range(0, val_data.shape[0], tf_bs):
        b = val_data[start:start + tf_bs]
        if b.device.type != device.split(':')[0]:
            b = b.to(device, non_blocking=True)
        with _autocast():
            pred, tgt = teacher_forced(model, b, cfg)
        pred, tgt = pred.float(), tgt.float()
        tf_loss += base_loss(pred, tgt, cfg).item() * b.shape[0]
        tf_mse += mse_loss(pred, tgt).item() * b.shape[0]
        tf_n += b.shape[0]

    # -- rollout vs persistence, matched populations ------------------------
    # Both are accumulated over the SAME rows and the SAME frames, so the
    # improvement percentage is a like-for-like comparison.
    n_roll = min(int(cfg.VAL_ROLLOUT_SEQS), val_data.shape[0])
    se_model = torch.zeros(n_frames, dtype=torch.float64)
    se_pers = torch.zeros(n_frames, dtype=torch.float64)
    frames_scored = n_frames
    rolled = 0
    t0 = time.time()
    for start in range(0, n_roll, chunk):
        b = val_data[start:min(start + chunk, n_roll)]
        if b.device.type != device.split(':')[0]:
            b = b.to(device, non_blocking=True)
        B = b.shape[0]
        true_f = b[:, ctx * NX:, :LD].reshape(B, n_frames, NX, LD).float()
        pers_f = b[:, (ctx - 1) * NX:ctx * NX, :LD].unsqueeze(1).float() \
                  .expand(-1, n_frames, -1, -1)
        with _autocast():
            pred_f = rollout_frames(model, b, cfg)
        pred_f = pred_f.float()
        k = pred_f.shape[1]
        frames_scored = min(frames_scored, k)
        # .cpu() BEFORE .double(): MPS has no float64, so the cast has to happen
        # on the host or it raises.
        se_model[:k] += ((pred_f - true_f[:, :k]) ** 2).mean(dim=(2, 3)).sum(0).cpu().double()
        se_pers += ((pers_f - true_f) ** 2).mean(dim=(2, 3)).sum(0).cpu().double()
        rolled += B
    roll_secs = time.time() - t0

    # Truncate both to the horizon actually produced, so a short rollout cannot
    # be flattered by comparing against a full-length persistence sum.
    mse_model_pf = (se_model / max(rolled, 1))[:frames_scored]
    mse_pers_pf = (se_pers / max(rolled, 1))[:frames_scored]
    imp_pf = (mse_pers_pf - mse_model_pf) / (mse_pers_pf + 1e-12) * 100
    m_all, p_all = float(mse_model_pf.mean()), float(mse_pers_pf.mean())

    return {
        "val_tf_loss": tf_loss / max(tf_n, 1),
        "val_tf_mse": tf_mse / max(tf_n, 1),
        "val_sequences_tf": tf_n,
        "rollout_sequences": rolled,
        "frames_scored": frames_scored,
        "rollout_mse": m_all,
        "persistence_mse": p_all,
        "improvement_pct": (p_all - m_all) / (p_all + 1e-12) * 100,
        "improvement_pct_frame1": float(imp_pf[0]),
        "improvement_pct_frame_half": float(imp_pf[len(imp_pf) // 2]),
        "improvement_pct_frame_last": float(imp_pf[-1]),
        "improvement_pct_per_frame": [round(float(v), 3) for v in imp_pf],
        "rollout_mse_per_frame": [float(v) for v in mse_model_pf],
        "rollout_seconds": roll_secs,
    }


# --------------------------------------------------------------------------- #
# wandb shim
# --------------------------------------------------------------------------- #
class _Telemetry:
    """wandb that cannot kill a 12-hour unattended run."""

    def __init__(self, enabled, **kwargs):
        self.wandb = None
        self.run = None
        if not enabled:
            return
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(**kwargs)
        except Exception as e:
            print(f"  [wandb] disabled ({type(e).__name__}: {e})", flush=True)
            self.run = None

    def log(self, payload, **kwargs):
        if self.run is None:
            return
        try:
            self.wandb.log(payload, **kwargs)
        except Exception:
            pass

    def set_summary(self, key, value):
        if self.run is None:
            return
        try:
            self.run.summary[key] = value
        except Exception:
            pass

    def finish(self):
        if self.run is None:
            return
        try:
            self.wandb.finish()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #
def make_lr_lambda(cfg):
    """Warmup then cosine decay, over the ACTUAL step budget.

    The old `OneCycleLR(epochs=100000, pct_start=0.1)` put the end of warmup
    10,000 epochs out and the cosine floor 90,000 epochs out, so in practice the
    LR only ever ramped and never annealed.
    """
    total = max(1, int(cfg.MAX_STEPS))
    warmup = max(1, int(total * float(cfg.WARMUP_FRAC)))
    floor = float(cfg.LR_FINAL_FRAC)

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return fn


def config_dict():
    return {k: getattr(Config, k) for k in dir(Config)
            if not k.startswith('_') and not callable(getattr(Config, k))}


def _log_write(path, log=print, kind="checkpoint"):
    """Rainbow-log a completed on-disk write.

    Uses the absolute path so remote logs (wandb / tmux scrollback) are
    unambiguous about where the artifact landed. The `_rainbow` helper
    already honours PFD_NO_COLOR / NO_COLOR / non-tty stdout, so this
    degrades to plain text automatically.
    """
    try:
        abs_path = os.path.abspath(path)
    except Exception:
        abs_path = str(path)
    size_hint = ""
    try:
        if os.path.exists(path):
            size_hint = f" ({os.path.getsize(path) / 1e6:.2f} MB)"
    except Exception:
        pass
    label = f"[write:{kind}]"
    log(_rainbow(f"{label} {abs_path}{size_hint}"))


def save_scripted_model(script_path, model, cfg=Config, device=None, log=print):
    """Save a self-contained TorchScript companion of `model` to `script_path`.

    This is the single guarded implementation used by `save_checkpoint` when
    `Config.SAVE_SCRIPTED_MODELS` is True. It closes every caveat listed in
    the module docstring's SCRIPTED MODEL SAVES section:

      1. `torch.compile` unwrap. If `model` was wrapped by `torch.compile`,
         its parameters are prefixed with `_orig_mod.` and it is not itself a
         plain `nn.Module` from TorchScript's point of view. We unwrap via
         `getattr(model, "_orig_mod", model)` before scripting.

      2. `torch.jit.script` first, `torch.jit.trace` fallback. Scripting
         preserves control flow (e.g. `if self.predict_delta:` branches)
         but requires TorchScript-clean source. Tracing bakes in whichever
         branch the example took but works on almost any eager module. If
         script fails, we fall back to trace on a representative synthetic
         input matching the arm's tokenization.

      3. Buffer sanity. `feat_mean` and `feat_std` MUST be registered as
         real buffers (they are on both `BaseTransformer` and
         `FrameTransformer`) so they ride along in the scripted artifact.
         We refuse to save otherwise -- a scripted module with a plain
         attribute `feat_mean` would evaluate uninitialised on the reload
         side and silently produce garbage.

      4. Frame-native vs. token-native shape. `frame_native` is a class
         attribute on the underlying module (True for `FrameTransformer`,
         False for the token variants); we read it AFTER the compile
         unwrap and size the representative example accordingly. The
         scripted artifact takes the same input shape that `evaluate()`
         and `rollout_frames()` already feed the eager model.

      5. Non-fatal training state preservation. Scripting toggles
         `model.eval()` and can move tensors; we restore `train()` on the
         underlying module before returning so the training loop is
         undisturbed.

      6. Roundtrip verification. We reload the just-written file with
         `torch.jit.load` and run one forward on a CPU-side synthetic
         input, so a broken scripted save is caught HERE, not on the H200
         evaluation box three hours from now. A failed roundtrip is
         logged loudly in yellow but does not abort training (the plain
         state-dict `.pt` is authoritative).

    Returns a small dict describing what happened (method, error strings,
    verification status) for callers that want to log/aggregate this.
    """
    inner = getattr(model, "_orig_mod", model)
    if device is None:
        try:
            device = next(inner.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Remember the eager module's original device BEFORE any scripting /
    # tracing happens. `torch.jit.trace` shares parameter storage with the
    # underlying eager module (unlike `torch.jit.script`, which copies), so
    # a later `scripted.to("cpu")` on a traced artifact silently migrates
    # the LIVE training model to CPU. The next optimizer step then dies
    # with `Expected all tensors to be on the same device, but got mat1 is
    # on cuda:0, different from other tensors on cpu`. We restore the
    # eager module to its original device after saving (whether we went
    # through script or trace) to close that leak. See the "SCRIPTED MODEL
    # SAVES" section of the module docstring for the full story.
    try:
        orig_device = next(inner.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")

    # (3) buffer sanity -- must be REAL buffers so torch.jit picks them up.
    buffer_names = {n for n, _ in inner.named_buffers()}
    for req in ("feat_mean", "feat_std"):
        if req not in buffer_names:
            raise RuntimeError(
                f"[scripted] refusing to save: '{req}' is not a registered "
                f"buffer on {type(inner).__name__}; TorchScript would drop "
                "it and eval would run against an uninitialised normaliser.")

    # (4) shape the representative example for the arm's tokenization.
    frame_native = bool(getattr(inner, "frame_native", False))
    if frame_native:
        seq_len = int(cfg.NUM_TIME)
        width = int(cfg.NUM_X * cfg.LATENT_DIM + FRAME_META_COLS)
    else:
        seq_len = int(cfg.SEQ_LEN)
        width = int(cfg.INPUT_DIM)

    # (5) freeze training-state semantics for the caller.
    was_training = inner.training
    inner.eval()

    result = {"path": script_path, "method": None,
              "script_error": None, "trace_error": None,
              "roundtrip_ok": False}
    scripted = None
    try:
        try:
            scripted = torch.jit.script(inner)
            result["method"] = "script"
        except Exception as e:
            result["script_error"] = f"{type(e).__name__}: {e}"
            # (2) trace fallback on a representative synthetic input.
            ex = torch.zeros(1, seq_len, width, device=device)
            try:
                scripted = torch.jit.trace(
                    inner, ex, strict=False, check_trace=False)
                result["method"] = "trace"
            except Exception as e2:
                result["trace_error"] = f"{type(e2).__name__}: {e2}"
                log(_c(
                    f"  [scripted] BOTH script and trace failed for "
                    f"{type(inner).__name__}: script={result['script_error']}; "
                    f"trace={result['trace_error']}. Skipping "
                    f"{os.path.basename(script_path)}.", "red"))
                return result

        # Convert to CPU before saving so the artifact is portable to a
        # host without a CUDA / MPS device.
        #
        # CRITICAL: `torch.jit.trace` returns a ScriptModule that shares
        # parameter/buffer STORAGE with `inner`. An in-place `.to("cpu")`
        # on such a traced object migrates the eager training model to
        # CPU as a side effect, which then explodes on the next
        # `optimizer.step()` with an addmm device-mismatch. `torch.jit.script`
        # already deep-copies params into a new ScriptModule, so its
        # `.to("cpu")` is independent of `inner`. To keep both branches
        # safe uniformly, we deep-copy the ScriptModule BEFORE moving --
        # cheap on the tiny module sizes we save (~19 MB) and eliminates
        # the storage-sharing footgun entirely. The `finally` block below
        # also unconditionally restores `inner` to `orig_device` as a
        # belt-and-suspenders guard against any future scripting mode
        # (e.g. `torch.jit.freeze`) that might reintroduce the aliasing.
        try:
            scripted_cpu = copy.deepcopy(scripted).to("cpu")
        except Exception:
            # deepcopy of a ScriptModule can fail on some torch versions;
            # if it does, save whichever object we have and rely on the
            # `finally` device-restore below to keep `inner` correct.
            try:
                scripted_cpu = scripted.to("cpu")
            except Exception:
                scripted_cpu = scripted

        tmp = script_path + ".tmp"
        torch.jit.save(scripted_cpu, tmp)
        os.replace(tmp, script_path)   # atomic
        _log_write(script_path, log=log, kind=f"scripted:{result['method']}")

        # (6) roundtrip: reload + one forward on CPU synthetic data.
        try:
            reloaded = torch.jit.load(script_path, map_location="cpu")
            with torch.no_grad():
                ex_cpu = torch.zeros(1, seq_len, width)
                _ = reloaded(ex_cpu)
            result["roundtrip_ok"] = True
        except Exception as e:
            log(_c(
                f"  [scripted] WARNING: roundtrip check FAILED for "
                f"{os.path.basename(script_path)} "
                f"({type(e).__name__}: {e}); state-dict `.pt` is still "
                f"authoritative.", "yellow"))
    finally:
        # Belt-and-suspenders: restore `inner` to the device it was on
        # when we entered, in case some path above (e.g. a future
        # torch.jit.freeze fallback, or a deepcopy that silently
        # didn't) still migrated its parameters.
        #
        # IMPORTANT: only call `.to()` when the device ACTUALLY differs.
        # `nn.Module.to()` unconditionally runs `_apply(...)`, which in
        # recent PyTorch iterates every parameter and evaluates
        # `param_grad = param.grad` (torch/nn/modules/module.py:~974).
        # On a `torch.compile`-wrapped model whose parameters are
        # exposed via a proxy, that grad access trips the "The .grad
        # attribute of a Tensor that is not a leaf Tensor is being
        # accessed" UserWarning even though nothing here needs to move.
        # Gating the call on a real device change avoids the spurious
        # warning without weakening the safety net: if `inner` ever
        # ends up on the wrong device, we still move it back.
        try:
            current_device = next(inner.parameters()).device
        except StopIteration:
            current_device = orig_device
        if current_device != orig_device:
            try:
                inner.to(orig_device)
            except Exception:
                pass
        if was_training:
            inner.train()

    return result


def save_checkpoint(path, model, optimizer, step, extra, scheduler=None,
                    save_scripted=None, cfg=Config, log=print):
    """Write a state-dict checkpoint (atomically) and, if enabled, a
    TorchScript companion at `<path without .pt>_scripted.pt`.

    Every completed on-disk write is rainbow-logged with the FULL absolute
    path via `_log_write` so an operator scrolling through a long log can
    always answer 'where did that best-rollout artifact go?' without
    re-deriving `Config.CHECKPOINT_DIR`.

    `save_scripted` overrides `Config.SAVE_SCRIPTED_MODELS` for callers that
    want to force the behaviour one way or the other (e.g. the unit test);
    default None means 'obey the config flag'.
    """
    payload = {
        'step': step,
        'epoch': step,          # back-compat: the leaderboard test reads 'epoch'
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config_dict(),
    }
    payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)       # atomic: a killed run never leaves a half file
    _log_write(path, log=log, kind="state_dict")

    do_scripted = (bool(cfg.SAVE_SCRIPTED_MODELS)
                   if save_scripted is None else bool(save_scripted))
    if do_scripted:
        if path.endswith(".pt"):
            script_path = path[:-3] + "_scripted.pt"
        else:
            script_path = path + "_scripted.pt"
        try:
            save_scripted_model(script_path, model, cfg=cfg, log=log)
        except Exception as e:
            log(_c(
                f"  [scripted] save failed for {os.path.basename(script_path)}"
                f" ({type(e).__name__}: {e}); state-dict `.pt` was still "
                f"written and is authoritative.", "yellow"))


# --------------------------------------------------------------------------- #
# Warm-start (v2.0)
# --------------------------------------------------------------------------- #
# Default checkpoint fed to `--warm-start`: the v1.0 rollout-best winner from
# tests/reports/r1_a3b_delta_ar_deep_dive.md (4.78 M-param, epoch 2400,
# causal OK). NOT `_best.pt` -- the deep dive nominated `_rollout_best.pt`.
DEFAULT_WARM_START_CKPT = os.path.join(
    Config.CHECKPOINT_DIR, "r1_a3b_delta_ar_rollout_best.pt")

# Keys whose absence is EXPECTED when warm-starting a v1.0 (NUM_TIME=40)
# checkpoint into a v2.0 (NUM_TIME=80) model, and why:
#   time_embeddings.weight  learned positional embedding of shape
#                           (NUM_TIME, EMBED_SIZE): 40->80, length-dependent.
#                           Cannot be transferred; will be freshly initialised
#                           by the 80-frame model and re-learned during
#                           warm-started training.
# Additionally we honour BENIGN_MISSING_KEYS from tests/test_model_vs_baseline.py
# (causal_mask/feat_mean/feat_std) for the same reasons the leaderboard test
# tolerates them: legacy no-op buffers or normalisation stats that the trainer
# repopulates before the first optimiser step (see `set_feature_stats` above).
WARM_START_LENGTH_DEPENDENT_KEYS = frozenset({"time_embeddings.weight"})
WARM_START_BENIGN_MISSING_KEYS = frozenset({"causal_mask", "feat_mean", "feat_std"})


def _wsc(text, color):
    """Console-colouring helper local to the warm-start log.

    Honours NO_COLOR (and its de-facto pair PFD_NO_COLOR, already respected
    by persistence_formal_documentation.py) plus non-tty stdout, so piping
    the trainer to a file never emits stray ANSI. Kept intentionally tiny
    and local so this step does not depend on the shared _ANSI helpers that
    live in persistence_formal_documentation.py -- those move into the
    shared module in Step 4.
    """
    if (os.environ.get("NO_COLOR") or os.environ.get("PFD_NO_COLOR")
            or not sys.stdout.isatty()):
        return text
    codes = {"green": "\033[32m", "red": "\033[31m", "yellow": "\033[33m",
             "cyan": "\033[36m", "bold": "\033[1m", "dim": "\033[2m"}
    return f"{codes.get(color, '')}{text}\033[0m"


def load_warm_start(model, ckpt_path, device, log=print):
    """Warm-start `model` from a v1.0 checkpoint under the v2.0 (NUM_TIME=80)
    shape.

    Semantics:
      * Only length-dependent tensors listed in
        `WARM_START_LENGTH_DEPENDENT_KEYS` may be dropped due to shape
        mismatch; everything else must transfer verbatim.
      * `load_state_dict(..., strict=False)` is used, but `strict=False` is
        NOT relied upon as a silent shape-mismatch shield: shape-mismatched
        keys are sanitised out of the state_dict BEFORE the load call, and
        any mismatch outside the allowlist is a hard failure.
      * `unexpected_keys` is always a hard failure (an unexpected key means
        the checkpoint and the current model disagree about architecture,
        which no amount of warm-start will fix).
      * Transferred vs. reinitialised parameter counts are logged in colour.

    Returns a dict summarising what happened, for W&B / audit logging.
    """
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"[warm-start] checkpoint not found: {ckpt_path}. Pass "
            "--warm-start PATH or --no-warm-start.")

    # Rainbow-log the starting-point checkpoint the same way `_log_write`
    # rainbow-logs a completed on-disk write, so "we're STARTING from THIS
    # file" is as visually unmissable in scrollback as "we JUST WROTE that
    # file". Absolute path so remote logs are unambiguous.
    try:
        abs_ckpt = os.path.abspath(ckpt_path)
    except Exception:
        abs_ckpt = str(ckpt_path)
    log(_rainbow(f"[start-from:warm-start] {abs_ckpt}"))
    log(_wsc(f"[warm-start] loading v1.0 winner: {ckpt_path}", "cyan"))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state_dict = ck["model_state_dict"]
        ck_meta = {k: ck.get(k) for k in ("step", "epoch", "val_l2",
                                          "train_l2", "rollout_mse")}
    elif isinstance(ck, dict) and "state_dict" in ck:
        state_dict = ck["state_dict"]
        ck_meta = {}
    else:
        state_dict = ck
        ck_meta = {}

    model_sd = model.state_dict()

    # Sanitise: drop keys whose shape does not match the target model's
    # parameter shape. Every dropped key must be in the allowlist, else fail.
    dropped_shape_mismatch = {}
    filtered = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape"):
            if tuple(v.shape) != tuple(model_sd[k].shape):
                dropped_shape_mismatch[k] = (tuple(v.shape),
                                             tuple(model_sd[k].shape))
                continue
        filtered[k] = v

    bad_shape = [k for k in dropped_shape_mismatch
                 if k not in WARM_START_LENGTH_DEPENDENT_KEYS]
    if bad_shape:
        detail = "\n  ".join(
            f"{k}: ckpt{dropped_shape_mismatch[k][0]} vs model{dropped_shape_mismatch[k][1]}"
            for k in bad_shape)
        raise SystemExit(
            f"[warm-start] REFUSING to load: shape mismatch outside the "
            f"length-dependent allowlist:\n  {detail}\n"
            f"Allowlist: {sorted(WARM_START_LENGTH_DEPENDENT_KEYS)}")

    incompatible = model.load_state_dict(filtered, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise SystemExit(
            f"[warm-start] REFUSING to load: unexpected keys in checkpoint "
            f"(architecture drift):\n  {unexpected}")

    # `missing_keys` after load = (target params never present in the
    # sanitised state_dict). Allowed = union(dropped-by-shape, BENIGN).
    allowed_missing = (set(dropped_shape_mismatch)
                       | WARM_START_BENIGN_MISSING_KEYS)
    bad_missing = [k for k in incompatible.missing_keys
                   if k not in allowed_missing]
    if bad_missing:
        raise SystemExit(
            f"[warm-start] REFUSING to load: missing keys not in allowlist:\n"
            f"  {bad_missing}\n"
            f"Allowed length-dependent: {sorted(WARM_START_LENGTH_DEPENDENT_KEYS)}\n"
            f"Allowed benign:           {sorted(WARM_START_BENIGN_MISSING_KEYS)}")

    # Parameter-count audit. `transferred` counts only rows/columns that
    # actually landed in the model (i.e. every tensor in `filtered` that
    # names a target parameter, not a buffer with no `.numel()` in
    # model_sd -- but state_dict includes both, so we sum over model_sd
    # keys that were populated).
    populated = [k for k in filtered if k in model_sd]
    n_transferred = sum(int(model_sd[k].numel()) for k in populated)
    reinit_keys = [k for k in model_sd if k not in filtered]
    n_reinit = sum(int(model_sd[k].numel()) for k in reinit_keys)
    n_total = sum(int(v.numel()) for v in model_sd.values())
    pct_transferred = 100.0 * n_transferred / max(1, n_total)

    log(_wsc(
        f"[warm-start] transferred {n_transferred/1e6:.2f} M "
        f"({pct_transferred:.2f}% of {n_total/1e6:.2f} M) "
        f"across {len(populated)} tensors", "green"))
    if reinit_keys:
        log(_wsc(
            f"[warm-start] reinitialised {n_reinit/1e6:.4f} M across "
            f"{len(reinit_keys)} tensors: {reinit_keys}", "yellow"))
    if dropped_shape_mismatch:
        for k, (ck_shape, m_shape) in dropped_shape_mismatch.items():
            log(_wsc(
                f"[warm-start]   dropped length-dependent {k}: "
                f"ckpt{ck_shape} -> model{m_shape} (allowlisted; will be "
                f"reinit-and-learned at NUM_TIME={Config.NUM_TIME})",
                "yellow"))
    if ck_meta:
        meta_str = "  ".join(f"{k}={v}" for k, v in ck_meta.items()
                             if v is not None)
        if meta_str:
            log(_wsc(f"[warm-start] source ckpt meta: {meta_str}", "dim"))

    return {
        "warm_start_path": ckpt_path,
        "transferred_params": n_transferred,
        "reinit_params": n_reinit,
        "total_params": n_total,
        "pct_transferred": pct_transferred,
        "dropped_length_dependent": sorted(dropped_shape_mismatch),
        "reinit_keys": reinit_keys,
        "ckpt_meta": {k: v for k, v in ck_meta.items() if v is not None},
    }


# --------------------------------------------------------------------------- #
# Per-epoch persistence report
# --------------------------------------------------------------------------- #
@torch.no_grad()
def per_epoch_persistence_report(model, val_data, cfg, device, epoch,
                                 optimizer_step, telemetry=None,
                                 n_seqs=32, n_frames=28, chunk=None,
                                 log=print):
    """Roll out `n_frames` on a fixed 32-sequence val subset and compare
    MAE / RMSE / L2 against a persistence baseline (last-context-frame held
    constant). Prints one colored line per metric with the `Δ` in green
    when the model beats persistence and red otherwise; optionally logs
    the numbers to W&B under the ``persistence/*`` namespace.
    """
    was_training = model.training
    model.eval()
    NX, LD = cfg.NUM_X, cfg.LATENT_DIM
    ctx_frames = int(cfg.VAL_CONTEXT_STEPS)
    max_horizon = max(0, cfg.NUM_TIME - ctx_frames)
    n_frames = int(min(n_frames, max_horizon))
    n_seqs = int(min(n_seqs, val_data.shape[0]))
    if n_seqs <= 0 or n_frames <= 0:
        if was_training:
            model.train()
        return {}

    batch = val_data[:n_seqs]
    target_device = torch.device(device) if isinstance(device, str) else device
    if batch.device != target_device:
        batch = batch.to(target_device, non_blocking=True)

    # Chunked rollout: at NUM_TIME=80 the AR loop grows the sequence to
    # SEQ_LEN=800 tokens (pre-v3.1: 2080) and the attention scores are
    # (chunk * n_heads * L^2 * 4B) PER LAYER. On MPS the caller passes
    # chunk=1 (via regime.eval_micro_batch); on CUDA the full n_seqs runs in
    # one shot. Accumulators are scalar sums scaled by n_seqs at the end.
    chunk = int(chunk) if chunk else int(n_seqs)
    chunk = max(1, min(chunk, n_seqs))
    gt_all = batch[:, :, :LD].reshape(n_seqs, cfg.NUM_TIME, NX, LD)
    n_out = None
    sum_abs_m = sum_sq_m = sum_l2_m = 0.0
    sum_abs_p = sum_sq_p = sum_l2_p = 0.0
    n_elems_m = n_elems_p = 0     # element count for MAE/RMSE means
    n_l2_m = n_l2_p = 0           # vector count for L2 means
    for start in range(0, n_seqs, chunk):
        end = min(start + chunk, n_seqs)
        sub = batch[start:end]
        pred_c = rollout_frames(model, sub, cfg,
                                ctx_frames=ctx_frames, n_frames=n_frames)
        k = pred_c.shape[1]
        if n_out is None:
            n_out = k
        else:
            n_out = min(n_out, k)
        gt_c = gt_all[start:end, ctx_frames:ctx_frames + k]
        pers_c = gt_all[start:end, ctx_frames - 1:ctx_frames].expand(
            -1, k, -1, -1).contiguous()
        dm = (pred_c - gt_c).float()
        dp = (pers_c - gt_c).float()
        sum_abs_m += dm.abs().sum().item()
        sum_sq_m += dm.pow(2).sum().item()
        sum_l2_m += torch.linalg.vector_norm(dm, dim=-1).sum().item()
        sum_abs_p += dp.abs().sum().item()
        sum_sq_p += dp.pow(2).sum().item()
        sum_l2_p += torch.linalg.vector_norm(dp, dim=-1).sum().item()
        n_elems_m += dm.numel(); n_elems_p += dp.numel()
        n_l2_m += dm.shape[0] * dm.shape[1] * dm.shape[2]
        n_l2_p += dp.shape[0] * dp.shape[1] * dp.shape[2]

    mae_m = sum_abs_m / max(n_elems_m, 1)
    rmse_m = (sum_sq_m / max(n_elems_m, 1)) ** 0.5
    l2_m = sum_l2_m / max(n_l2_m, 1)
    mae_p = sum_abs_p / max(n_elems_p, 1)
    rmse_p = (sum_sq_p / max(n_elems_p, 1)) ** 0.5
    l2_p = sum_l2_p / max(n_l2_p, 1)

    def _delta(m, p):
        return (p - m) / max(p, 1e-12) * 100.0

    def _line(name, m, p):
        d = _delta(m, p)
        colored_delta = _c(f"Δ={d:+.2f}%", "green" if d > 0 else "red")
        log(f"epoch {epoch:>3}  {name:<5} model={m:.3e}  pers={p:.3e}  {colored_delta}")

    _line("MAE",  mae_m,  mae_p)
    _line("RMSE", rmse_m, rmse_p)
    _line("L2",   l2_m,   l2_p)

    payload = {
        "persistence/mae_model": mae_m,
        "persistence/mae_pers": mae_p,
        "persistence/mae_delta_pct": _delta(mae_m, mae_p),
        "persistence/rmse_model": rmse_m,
        "persistence/rmse_pers": rmse_p,
        "persistence/rmse_delta_pct": _delta(rmse_m, rmse_p),
        "persistence/l2_model": l2_m,
        "persistence/l2_pers": l2_p,
        "persistence/l2_delta_pct": _delta(l2_m, l2_p),
        "persistence/epoch": int(epoch),
        "persistence/horizon_frames": int(n_out),
        "persistence/n_seqs": int(n_seqs),
    }
    if telemetry is not None:
        try:
            telemetry.log(payload, step=int(optimizer_step))
        except TypeError:
            telemetry.log(payload)

    if was_training:
        model.train()
    return payload


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args, log=print):
    t_start = time.time()
    device = Config.DEVICE
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Device-adaptive regime: resolved ONCE at startup. Everything the training
    # loop is allowed to know about the hardware flows through this object.
    regime = resolve_train_regime(device)
    print(regime.banner, flush=True)

    if Config.USE_TF32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if Config.USE_CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # AMP is regime-driven. On CUDA we use bf16 autocast without GradScaler
    # (bf16 has no fp16-style underflow to worry about). On MPS/CPU we do not
    # autocast at all -- the extra dtype juggling costs more than it saves.
    amp_dtype = regime.amp_dtype if regime.use_amp else None
    use_scaler = False

    frame_level = Config.TOKENIZATION == 'frame'

    # -- data ---------------------------------------------------------------
    train_ds = TransformerDataset(Config.TRAIN_H5, subset_ratio=Config.TRAIN_SUBSET_RATIO)
    val_ds = TransformerDataset(Config.VAL_H5, subset_ratio=1.0)
    log(f"  [data] train={len(train_ds):,}/{train_ds.total_available:,} sequences  "
        f"val={len(val_ds):,}")
    if not args.cpu_data:
        _preload_to_device(train_ds, "train", device, log)
        _preload_to_device(val_ds, "val", device, log)

    # Three generators because a torch Generator is bound to a device and these
    # three consumers can legitimately live on different ones (e.g. --cpu-data
    # keeps the dataset in host memory while the model trains on cuda):
    #   data_gen -> randperm over the dataset tensor
    #   dev_gen  -> noise / scheduled-sampling masks, on the COMPUTE device
    #   cpu_gen  -> scalar draws (randint) which are always CPU
    data_gen = torch.Generator(device=train_ds.data.device)
    data_gen.manual_seed(Config.SEED)
    dev_gen = torch.Generator(device=device)
    dev_gen.manual_seed(Config.SEED + 1)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(Config.SEED + 2)

    # Regime-driven physical / effective batch. `Config.BATCH_SIZE` and
    # `Config.ACCUMULATION_STEPS` are no longer read inside the loop.
    micro_batch = int(regime.micro_batch)
    virtual_batch = int(regime.virtual_batch)
    accum_steps = max(1, virtual_batch // micro_batch)
    train_loader = InMemoryBatcher(train_ds.data, micro_batch, shuffle=True,
                                   generator=data_gen)
    stream = infinite_batches(train_loader)
    steps_per_epoch = max(1, len(train_loader) // accum_steps)

    # -- model --------------------------------------------------------------
    model = get_model(Config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  [model] variant={Config.VARIANT} tokenization={Config.TOKENIZATION} "
        f"E={Config.EMBED_SIZE} L={Config.N_LAYERS} H={Config.N_HEADS} "
        f"params={n_params / 1e6:.2f}M attn_impl={Config.ATTN_IMPL} "
        f"delta={Config.PREDICT_DELTA} rope={Config.USE_ROPE}")

    if Config.NORMALIZE_FEATURES:
        mean, std = compute_feature_stats(train_ds.data, Config, frame_level)
        model.set_feature_stats(mean, std)
        log(f"  [model] feature stats installed: mean|max|={mean.abs().max():.3f} "
            f"std range [{std.min():.4g}, {std.max():.4g}]")

    # -- warm-start (v2.0) --------------------------------------------------
    # Loads the v1.0 rollout-best winner and transfers all shape-compatible
    # weights; the length-dependent positional embedding is dropped and
    # reinitialised for NUM_TIME=80. Skipped when --no-warm-start is set,
    # or when a resume checkpoint exists (--fresh is off and latest.pt is
    # on disk), because the resume block below will overwrite these weights
    # anyway and doing both would waste I/O.
    run_name_for_resume = f"r{Config.SWEEP_ROUND}_{Config.ARM}"
    latest_path_check = os.path.join(
        Config.CHECKPOINT_DIR, f"{run_name_for_resume}_latest.pt")
    will_resume = os.path.exists(latest_path_check) and not args.fresh
    warm_start_summary = None
    if getattr(args, "no_warm_start", False):
        log(_wsc("  [warm-start] disabled by --no-warm-start", "yellow"))
    elif will_resume:
        log(_wsc(
            f"  [warm-start] skipped: {latest_path_check} exists and "
            "--fresh is off; resume will supply the weights.", "dim"))
    else:
        warm_start_summary = load_warm_start(
            model, args.warm_start, device, log=lambda s: log("  " + s))

    warm_started = warm_start_summary is not None

    # -- torch.compile (CUDA only, never fatal) ----------------------------
    if regime.compile_model:
        try:
            model = torch.compile(model)
            log("  [compile] torch.compile(model): OK")
        except Exception as e:
            log(f"  [compile] torch.compile(model) failed "
                f"({type(e).__name__}: {e}); continuing eager")

    # -- causality gate -----------------------------------------------------
    probe = probe_causality(model, Config, device)
    log(f"  [causality] before_cut={probe['max_change_before_cut']:.3e} "
        f"after_cut={probe['max_change_after_cut']:.3e} "
        f"tol={probe['tolerance']:.3e} -> causal={probe['causal']}")
    if not probe['causal'] and not args.allow_leak:
        raise RuntimeError(
            f"CAUSALITY PROBE FAILED for arm {Config.ARM}: perturbing inputs at/after "
            f"position {probe['cut_position']} changed earlier outputs by "
            f"{probe['max_change_before_cut']:.3e} (tolerance {probe['tolerance']:.3e}). "
            f"The model can see the future; every metric would be meaningless. "
            f"Pass --allow-leak only if you are deliberately measuring the leak.")
    if not probe['probe_responsive']:
        log("  [causality] WARNING: perturbing the future changed nothing either -- "
            "the probe may not be exercising the model.")

    # -- optimiser ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY, betas=tuple(Config.ADAM_BETAS))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(Config))
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_scaler)

    run_name = f"r{Config.SWEEP_ROUND}_{Config.ARM}"
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_latest.pt")

    step = 0
    best = {"rollout_mse": float('inf'), "improvement_pct": -float('inf'),
            "val_tf_mse": float('inf'), "train_loss": float('inf')}

    if os.path.exists(latest_path) and not args.fresh:
        try:
            ck = torch.load(latest_path, map_location=device, weights_only=False)
            raw_sd = ck['model_state_dict']
            # Sanitize length-dependent tensors just like load_warm_start:
            # a `latest.pt` produced by a v1.0 (NUM_TIME=40) run has
            # `time_embeddings.weight` of shape (40, 256), which cannot be
            # copied into the v2.0 (NUM_TIME=80) model's (80, 256) parameter.
            # `strict=False` alone does NOT skip shape-mismatched present
            # keys -- it still raises. So we drop them explicitly and let
            # them stay at their freshly-initialised (or warm-started) values.
            model_sd = model.state_dict()
            dropped = {}
            filtered = {}
            for k, v in raw_sd.items():
                if k in model_sd and hasattr(v, "shape") and tuple(v.shape) != tuple(model_sd[k].shape):
                    dropped[k] = (tuple(v.shape), tuple(model_sd[k].shape))
                    continue
                filtered[k] = v
            bad = [k for k in dropped if k not in WARM_START_LENGTH_DEPENDENT_KEYS]
            if bad:
                detail = ", ".join(
                    f"{k}: ckpt{dropped[k][0]} vs model{dropped[k][1]}" for k in bad)
                raise RuntimeError(
                    f"resume shape mismatch outside length-dependent allowlist: {detail}")
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            # Cross-version detection: if length-dependent tensors had to be
            # dropped, this is not a real resume -- it is a warm-start out of
            # a v1.0 (NUM_TIME=40) checkpoint that just happens to be sitting
            # in the v2.0 saved_models/ directory. In that case:
            #   * the step counter from v1.0 is meaningless for v2.0 (v1.0
            #     ran to MAX_STEPS=6000; keeping step=6000 makes the
            #     `while step < MAX_STEPS:` loop exit immediately without
            #     training a single step at NUM_TIME=80 -- observed regression);
            #   * the optimizer's per-parameter moment tensors are stale for
            #     any reinitialised parameter;
            #   * the scheduler's `last_epoch` should start at 0 so the v2.0
            #     run gets its full warmup+cosine schedule, not the v1.0
            #     annealed tail;
            #   * `best` metrics from v1.0 (at NUM_TIME=40, VAL_ROLLOUT_STEPS=728)
            #     are not comparable to v2.0 (VAL_ROLLOUT_STEPS=1768).
            # So we reset step/best/scheduler to fresh in that branch. Same
            # optimizer/scheduler skip still applies for the regular
            # (no-dropped) resume path.
            cross_version = bool(dropped)
            if ck.get('optimizer_state_dict') and not cross_version:
                optimizer.load_state_dict(ck['optimizer_state_dict'])
            elif ck.get('optimizer_state_dict') and cross_version:
                log(f"  [resume] skipping optimizer state: length-dependent "
                    f"tensors were reinitialised ({sorted(dropped)})")
            if cross_version:
                log(f"  [resume] cross-version detected (v1.0 -> v2.0): "
                    f"resetting step=0 and best/* -- v1.0 metrics at "
                    f"NUM_TIME=40 are not comparable to v2.0 at NUM_TIME=80. "
                    f"Effectively a warm-start from {latest_path}.")
                step = 0
            else:
                step = int(ck.get('step', 0))
            # Restore scheduler state directly instead of replaying
            # scheduler.step() `step` times: the replay path called
            # scheduler.step() before any optimizer.step() had run in this
            # process, which is exactly the pattern PyTorch warns about
            # ("Detected call of `lr_scheduler.step()` before
            # `optimizer.step()`") and which also silently skips the first
            # scheduled LR value. If the checkpoint carries a scheduler
            # state_dict (v2.0.2+), load it verbatim; else fall back to
            # setting `last_epoch` and rebuilding the LR without calling
            # `.step()` (see PyTorch docs: setting last_epoch and calling
            # get_last_lr is the supported resume-without-warning path).
            sched_sd = ck.get('scheduler_state_dict')
            if sched_sd is not None and not cross_version:
                try:
                    scheduler.load_state_dict(sched_sd)
                except Exception as e:
                    log(f"  [resume] scheduler.load_state_dict failed "
                        f"({type(e).__name__}: {e}); reconstructing from step")
                    sched_sd = None
            elif cross_version:
                # Leave the scheduler at last_epoch=-1 so v2.0 gets its full
                # warmup+cosine schedule; no `.step()` is called pre-optimizer.
                sched_sd = "cross_version_reset"
            if sched_sd is None and step > 0:
                # Reconstruct scheduler position without triggering the
                # step-before-optimizer warning. LambdaLR reads last_epoch
                # and applies the lr_lambda(last_epoch) on the next .step().
                scheduler.last_epoch = step - 1
                for group, base_lr in zip(optimizer.param_groups,
                                          scheduler.base_lrs):
                    group['lr'] = base_lr * scheduler.lr_lambdas[0](step - 1)
                scheduler._last_lr = [g['lr'] for g in optimizer.param_groups]
            if not cross_version:
                best.update({k: v for k, v in ck.get('best', {}).items()})
            # Rainbow-log the resume-from checkpoint the same way
            # `_log_write` rainbow-logs a completed write and
            # `load_warm_start` rainbow-logs its start-from checkpoint,
            # so operators can visually pinpoint "we RESUMED from this
            # exact file at step N" in a long scrollback.
            try:
                abs_resume = os.path.abspath(latest_path)
            except Exception:
                abs_resume = str(latest_path)
            log(_rainbow(f"[start-from:resume] {abs_resume} @ step {step}"))
            log(f"  [resume] {latest_path} at step {step} "
                f"(missing={len(missing)}, unexpected={len(unexpected)}, "
                f"dropped_length_dependent={sorted(dropped)})")
        except Exception as e:
            log(f"  [resume] failed ({type(e).__name__}: {e}); starting fresh")

    # W&B run name embeds arm, NUM_TIME, and whether warm-start was actually
    # applied on this run (a resumed run or --no-warm-start reads as ws=0).
    wandb_run_name = f"r{Config.SWEEP_ROUND}_{Config.ARM}_t{Config.NUM_TIME}_ws{int(warm_started)}"
    wandb_config = dict(config_dict())
    wandb_config.update({
        "regime.device": regime.device,
        "regime.micro_batch": regime.micro_batch,
        "regime.virtual_batch": regime.virtual_batch,
        "regime.use_amp": regime.use_amp,
        "regime.amp_dtype": str(regime.amp_dtype),
        "regime.compile_model": regime.compile_model,
        "regime.cudnn_benchmark": regime.cudnn_benchmark,
        "warm_started": warm_started,
    })
    tel = _Telemetry(
        not args.no_wandb, project=Config.WANDB_PROJECT,
        name=wandb_run_name, id=run_name,
        resume="allow", config=wandb_config)

    # Disable AR aux loss on MPS/CPU. Even at AR_SEQS=1 the sequential AR loop
    # under token tokenization does `AR_FRAMES * NUM_X` sequential forwards
    # (e.g. 4*10 = 40 for the default arm, post-v3.1; was 4*26=104), each retaining its own full
    # activation graph for backward through `preds`. That accumulates past the
    # 88 GB MPS ceiling regardless of AR_SEQS. CUDA branch keeps the AR loss
    # at the arm-specified AR_SEQS. The primary next-token loss is unaffected.
    if regime.disable_ar and Config.AR_MODE != 'none':
        log(f"  [regime] disabling AR aux loss ({Config.AR_MODE} -> none) "
            f"on device={regime.device}: sequential rollout retains "
            f"AR_FRAMES*NUM_X={int(Config.AR_FRAMES)*Config.NUM_X} forward "
            f"activation graphs, OOMs on MPS even at AR_SEQS=1. "
            f"CUDA path is untouched.")
        Config.AR_MODE = 'none'
        Config.AR_LOSS_WEIGHT = 0.0
    elif int(Config.AR_SEQS) > int(regime.aux_micro_batch):
        log(f"  [regime] clamping AR_SEQS {Config.AR_SEQS} -> {regime.aux_micro_batch} "
            f"on device={regime.device} (keeps CUDA defaults untouched)")
        Config.AR_SEQS = int(regime.aux_micro_batch)
    ar_mode = Config.AR_MODE
    ar_target_w = float(Config.AR_LOSS_WEIGHT)
    ar_warm = max(1, int(Config.MAX_STEPS * float(Config.AR_WEIGHT_WARMUP_FRAC)))

    # -- memory expectation printout ---------------------------------------
    # Rough peak-attention-score estimate per code path, so an operator can
    # see at a glance what the resolved regime is spending memory on and
    # spot a regression immediately if a future edit re-inflates a path.
    # Model: peak = B * n_heads * L^2 * 4 bytes, times N_LAYERS for the
    # simultaneously-live layer activations, times a small forward-count
    # factor for the sequential AR loop (retained-graph across forwards).
    def _fmt_bytes(nbytes):
        for unit in ("B", "KiB", "MiB", "GiB"):
            if nbytes < 1024.0 or unit == "GiB":
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024.0

    def _attn_bytes(B, L, layers=Config.N_LAYERS, heads=Config.N_HEADS,
                    forwards=1):
        return int(B) * heads * (int(L) ** 2) * 4 * layers * forwards

    tf_L = Config.SEQ_LEN - 1
    train_peak = _attn_bytes(micro_batch, tf_L)
    eval_tf_peak = _attn_bytes(regime.eval_micro_batch, tf_L)
    rollout_L = Config.SEQ_LEN
    eval_rollout_peak = _attn_bytes(regime.eval_micro_batch, rollout_L)
    if regime.disable_ar:
        ar_peak_str = _c("DISABLED (MPS/CPU)", "yellow")
    else:
        # Approximate AR peak: activation graphs for AR_FRAMES*NUM_X forwards
        # under token tokenization, at the growing sequence length. Use the
        # final (largest) forward's L as the ceiling estimate.
        n_ar_fwd = int(Config.AR_FRAMES) * (1 if frame_level else Config.NUM_X)
        ar_peak = _attn_bytes(int(Config.AR_SEQS), rollout_L, forwards=n_ar_fwd)
        ar_peak_str = _fmt_bytes(ar_peak) + f" ({n_ar_fwd} retained forwards)"
    log(_bold("  [memory] expected peak attention-score bytes per path "
              "(B*H*L^2*4B * layers):", "cyan"))
    log(f"  [memory]   train forward         B={micro_batch:<3} L={tf_L:<5} "
        f"-> {_fmt_bytes(train_peak)}")
    log(f"  [memory]   eval TF forward       B={regime.eval_micro_batch:<3} "
        f"L={tf_L:<5} -> {_fmt_bytes(eval_tf_peak)}")
    log(f"  [memory]   eval rollout / persistence report  B={regime.eval_micro_batch:<3} "
        f"L={rollout_L:<5} -> {_fmt_bytes(eval_rollout_peak)}")
    log(f"  [memory]   AR aux loss           {ar_peak_str}")
    if regime.device == "mps":
        log(_c("  [memory]   MPS ceiling on this box is ~88 GB "
               "(PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to override)", "dim"))
    latent_width = (Config.NUM_X * Config.LATENT_DIM) if frame_level else Config.LATENT_DIM

    # The sanity floor, logged before the first step so the whole run can be read
    # against it instead of in a vacuum.
    nulls = null_baselines(val_ds.data, Config, frame_level=frame_level)
    # Floor = the best CONSTANT predictor, not specifically zeros. Which of the
    # two is lower depends on how far the latents are offset from zero, and using
    # only "zeros" would hand an easy pass to any data with a mean offset.
    floor = min(nulls["zeros"][Config.LOSS], nulls["mean"][Config.LOSS])
    anchor = nulls["previous frame"][Config.LOSS]
    log(f"  [floor] trivial predictors on this objective ({Config.LOSS}): "
        + "  ".join(f"{k}={v[Config.LOSS]:.6g}"
                    for k, v in nulls.items() if not k.startswith("_")))
    log(f"  [floor] must beat {floor:.6g} (best constant) to have learned anything, "
        f"and {anchor:.6g} (previous-frame anchor) to have a chance against persistence")

    curves = []
    last_metrics = {}
    log(f"  [train] budget={Config.MAX_STEPS} optimizer steps  "
        f"micro_batch={micro_batch} x accum={accum_steps} "
        f"= effective {micro_batch * accum_steps}  "
        f"steps_per_epoch~{steps_per_epoch}  "
        f"amp={amp_dtype}  loss={Config.LOSS}  device={regime.device}")

    stop_reason = "completed"
    # Rate is measured from where this PROCESS started, not from step 0, so an
    # ETA after resuming a checkpoint is not diluted by the earlier run's steps.
    start_step_for_rate = step
    first_vbatch_micro_count_logged = False
    while step < Config.MAX_STEPS:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc = base_acc = ar_acc = 0.0
        ar_w = ar_target_w * min(1.0, (step + 1) / ar_warm) if ar_mode != 'none' else 0.0
        run_ar = (ar_mode != 'none' and ar_w > 0
                  and step % max(1, int(Config.AR_EVERY_N_STEPS)) == 0)

        micro_count = 0
        for micro in range(accum_steps):
            batch = next(stream)
            if batch.device.type != device.split(':')[0]:
                batch = batch.to(device, non_blocking=True)

            if regime.use_amp and regime.device == "cuda" and amp_dtype is not None:
                amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)
            else:
                amp_ctx = torch.autocast(device_type='cpu', enabled=False)
            with amp_ctx:
                pred, tgt = teacher_forced(model, batch, Config,
                                           noise_std=Config.NOISE_STD, generator=dev_gen)
                loss = base_loss(pred, tgt, Config)
                base_acc += loss.item()
                # The auxiliary loss is sequential and by far the most expensive
                # part of a step, so it runs on the first micro-batch only.
                if run_ar and micro == 0:
                    aux = (frame_ar_loss(model, batch, Config, generator=cpu_gen)
                           if ar_mode == 'frame_ar'
                           else sched_sampling_loss(model, batch, Config,
                                                    Config.SCHED_SAMPLING_P,
                                                    generator=dev_gen))
                    if aux is not None:
                        ar_acc = aux.item()
                        loss = loss + ar_w * aux
                loss_acc += loss.item()
                loss = loss / accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_count += 1

        # First-vbatch accumulation observability: catch an early regression
        # where the loop runs a different number of micro-batches than the
        # regime asked for. Assertion is loud but non-fatal in prod (log only).
        if not first_vbatch_micro_count_logged:
            log(f"  [train] first virtual batch: observed {micro_count} "
                f"micro-batches (expected {accum_steps})")
            assert micro_count == accum_steps, (
                f"accumulation mismatch: observed {micro_count} vs expected {accum_steps}")
            first_vbatch_micro_count_logged = True

        if Config.GRAD_CLIP and Config.GRAD_CLIP > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        else:
            grad_norm = torch.tensor(float('nan'))

        # optimizer.step() + zero_grad() only when accumulated == accum_steps,
        # which is enforced by the accum_steps-length inner for-loop above.
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        step += 1

        train_loss = base_acc / accum_steps
        prev_train_best = best["train_loss"]
        best["train_loss"] = min(best["train_loss"], train_loss)
        # Save a train-best checkpoint on any real improvement so a run that
        # crosses the previous-frame anchor early (e.g. step 25) has a durable
        # artifact on disk, not just an in-memory `best["train_loss"]`. Rate is
        # bounded by LOG_EVERY_STEPS via the gate below so we never write on
        # every single optimizer step. The checkpoint is also gated on beating
        # the anchor floor: below that the "improvement" is still worse than
        # the previous-frame baseline and not worth persisting as best.
        train_improved = train_loss < prev_train_best and train_loss < anchor
        crossed_anchor = prev_train_best >= anchor and train_loss < anchor

        # Log the first few steps unconditionally. These runs are unattended and
        # often remote, and a log that shows nothing until step LOG_EVERY_STEPS is
        # indistinguishable from a hang -- the first three lines confirm the loop
        # is alive and give an immediate per-step cost to extrapolate from.
        if step <= 3 or step % Config.LOG_EVERY_STEPS == 0:
            lr = scheduler.get_last_lr()[0]
            payload = {"step": step, "train_loss": train_loss, "lr": lr,
                       "grad_norm": float(grad_norm), "ar_weight": ar_w}
            if run_ar:
                payload["ar_loss"] = ar_acc
            if torch.cuda.is_available():
                payload["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            tel.log(payload)
            flag = ""
            if train_loss > floor:
                flag = _c(f"  <-- WORSE THAN PREDICTING ZERO ({floor:.6f})", "red")
            elif train_loss > anchor:
                flag = _c(
                    f"  <-- worse than the previous-frame anchor ({anchor:.6f})",
                    "red")
            elif crossed_anchor:
                flag = _c(
                    f"  <-- beats previous-frame anchor ({anchor:.6f}); "
                    f"saving _train_best.pt", "green")
            elapsed = time.time() - t_start
            eta = ""
            if step >= 3:
                per_step = elapsed / max(1, step - start_step_for_rate)
                eta = f"  eta~{per_step * (Config.MAX_STEPS - step) / 3600:.1f}h"
            log(f"  step {step:>6}/{Config.MAX_STEPS}  train={train_loss:.6f}  "
                f"lr={lr:.2e}  gnorm={float(grad_norm):.3f}  "
                f"{elapsed / 60:.1f}m{eta}{flag}")
            # Persist train-best on new minima (post-anchor). Placed inside the
            # log cadence so I/O is bounded to at most ~1 write per
            # LOG_EVERY_STEPS optimizer steps, not per step.
            if train_improved:
                save_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_train_best.pt"),
                    model, optimizer, step,
                    {'train_l2': train_loss, 'best': dict(best)},
                    scheduler=scheduler)
                tel.set_summary("best_train_loss", best["train_loss"])

        hit_budget = step >= Config.MAX_STEPS

        # Per-epoch persistence report on a fixed 32-sequence val subset.
        # Distinct from the VAL_EVERY_STEPS evaluation above -- this fires
        # once per epoch and is cheap enough (28-frame rollout on 32 seqs)
        # to be affordable every epoch, including on MPS.
        if step % steps_per_epoch == 0 or hit_budget:
            epoch = step // steps_per_epoch
            try:
                per_epoch_persistence_report(
                    model, val_ds.data, Config, device,
                    epoch=epoch, optimizer_step=step, telemetry=tel,
                    n_seqs=32, n_frames=28,
                    chunk=regime.eval_micro_batch, log=log)
            except Exception as e:
                log(f"  [persistence] report failed "
                    f"({type(e).__name__}: {e}); continuing")

        if step % Config.VAL_EVERY_STEPS == 0 or hit_budget:
            m = evaluate(model, val_ds.data, Config, device,
                         amp_dtype=amp_dtype,
                         chunk=regime.eval_micro_batch,
                         tf_batch_size=regime.eval_micro_batch)
            m["step"] = step
            m["train_loss"] = train_loss
            m["lr"] = scheduler.get_last_lr()[0]
            m["wall_seconds"] = time.time() - t_start
            last_metrics = m
            # train/val gap on the SAME quantity, so an overfitting verdict does
            # not depend on comparing two differently-scaled losses.
            m["train_val_gap"] = m["val_tf_loss"] - train_loss
            curves.append({k: m[k] for k in (
                "step", "train_loss", "val_tf_loss", "val_tf_mse", "rollout_mse",
                "persistence_mse", "improvement_pct", "improvement_pct_frame1",
                "improvement_pct_frame_last", "lr", "wall_seconds")})
            tel.log({k: v for k, v in m.items() if not isinstance(v, list)})

            log(f"  [eval] step {step}: val_tf={m['val_tf_loss']:.6f} "
                f"rollout_mse={m['rollout_mse']:.6f} pers_mse={m['persistence_mse']:.6f} "
                f"IMPROVEMENT={m['improvement_pct']:+.2f}%  "
                f"(frame1 {m['improvement_pct_frame1']:+.2f}%, "
                f"last {m['improvement_pct_frame_last']:+.2f}%)  "
                f"[{m['rollout_sequences']} seqs, {m['rollout_seconds']:.1f}s]")

            if m["improvement_pct"] > best["improvement_pct"]:
                best["improvement_pct"] = m["improvement_pct"]
            if m["rollout_mse"] < best["rollout_mse"]:
                best["rollout_mse"] = m["rollout_mse"]
                save_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_rollout_best.pt"),
                    model, optimizer, step,
                    {'rollout_mse': m['rollout_mse'], 'val_l2': m['val_tf_loss'],
                     'improvement': m['improvement_pct'], 'train_l2': train_loss,
                     'best': dict(best)},
                    scheduler=scheduler)
                log(f"  --> new best rollout ({m['rollout_mse']:.6f}, "
                    f"{m['improvement_pct']:+.2f}% vs persistence)")
            if m["val_tf_mse"] < best["val_tf_mse"]:
                best["val_tf_mse"] = m["val_tf_mse"]
                save_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pt"),
                    model, optimizer, step,
                    {'val_l2': m['val_tf_loss'], 'rollout_mse': m['rollout_mse'],
                     'improvement': m['improvement_pct'], 'train_l2': train_loss,
                     'best': dict(best)},
                    scheduler=scheduler)
            tel.set_summary("best_rollout_mse", best["rollout_mse"])
            tel.set_summary("best_improvement_pct", best["improvement_pct"])

        if step % Config.CHECKPOINT_EVERY_STEPS == 0 or hit_budget:
            save_checkpoint(latest_path, model, optimizer, step,
                            {'train_l2': train_loss, 'best': dict(best),
                             'val_l2': best['val_tf_mse'],
                             'rollout_mse': best['rollout_mse'],
                             'improvement': best['improvement_pct']},
                            scheduler=scheduler)

        if (time.time() - t_start) / 3600.0 > Config.MAX_HOURS:
            stop_reason = f"wall-clock limit ({Config.MAX_HOURS}h)"
            log(f"  [train] stopping: {stop_reason}")
            break

    final = curves[-1] if curves else {}
    result = {
        "arm": Config.ARM,
        "round": Config.SWEEP_ROUND,
        "run_name": run_name,
        "stop_reason": stop_reason,
        "steps_completed": step,
        "wall_seconds": time.time() - t_start,
        "params_m": n_params / 1e6,
        "causality_probe": probe,
        "null_baselines": nulls,
        "constant_floor": floor,
        "anchor_floor": anchor,
        "beat_constant_predictor": bool(best["train_loss"] < floor),
        "beat_frame_anchor": bool(best["train_loss"] < anchor),
        "config": {k: v for k, v in config_dict().items()
                   if isinstance(v, (int, float, str, bool, tuple, list))},
        "arm_spec": {k: v for k, v in resolve_arm(Config.ARM).items() if k != "overrides"},
        "best": best,
        "final": final,
        "curves": curves,
        # Per-frame breakdown of the last evaluation. This is what separates
        # "one-step prediction is fine but error accumulates" (good frame 1,
        # collapsing tail) from "the model never learned anything" (flat ~0%).
        "final_per_frame_improvement_pct": last_metrics.get("improvement_pct_per_frame", []),
        "final_improvement_pct": last_metrics.get("improvement_pct"),
        "final_train_val_gap": last_metrics.get("train_val_gap"),
        "final_val_tf_loss": last_metrics.get("val_tf_loss"),
    }

    out_json = os.path.join(args.out_dir, f"{Config.ARM}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"  [done] arm={Config.ARM} steps={step} "
        f"best_improvement={best['improvement_pct']:+.2f}% -> {out_json}")

    tel.finish()
    return result


# --------------------------------------------------------------------------- #
# Diagnostics entry point
# --------------------------------------------------------------------------- #
def run_diagnostics(args, log=print):
    device = Config.DEVICE
    out = {"device": device, "torch_version": torch.__version__}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_count"] = torch.cuda.device_count()
        out["bf16_supported"] = torch.cuda.is_bf16_supported()
    log(f"[diag] torch={torch.__version__} device={device} "
        f"gpu={out.get('gpu', 'n/a')} x{out.get('gpu_count', 0)}")

    log("\n[diag] --- causality of the OLD nn.MultiheadAttention is_causal call ---")
    out["legacy_mha_probe"] = probe_legacy_attention(Config, device)
    log(f"  {out['legacy_mha_probe']}")

    log("\n[diag] --- old symmetric vs fixed causal Conv1d padding ---")
    out["conv_padding_probe"] = probe_symmetric_conv_leak(device)
    for k, v in out["conv_padding_probe"].items():
        log(f"  {k:<28} max change in PAST outputs from a FUTURE perturbation = {v:.3e}")

    log("\n[diag] --- causality of each model configuration we intend to train ---")
    out["model_probes"] = {}
    for arm_name, spec in ROUND1_ARMS.items():
        saved = {k: getattr(Config, k) for k in spec.get("overrides", {})}
        try:
            for k, v in spec.get("overrides", {}).items():
                setattr(Config, k, v)
            m = get_model(Config).to(device)
            p = probe_causality(m, Config, device)
            out["model_probes"][arm_name] = p
            log(f"  {arm_name:<14} causal={p['causal']!s:<5} "
                f"before={p['max_change_before_cut']:.3e} "
                f"after={p['max_change_after_cut']:.3e}")
            del m
        except Exception as e:
            out["model_probes"][arm_name] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  {arm_name:<14} ERROR {type(e).__name__}: {e}")
        finally:
            for k, v in saved.items():
                setattr(Config, k, v)
            Config.USE_SWIGLU = False
            if device == "cuda":
                torch.cuda.empty_cache()

    train_ds = TransformerDataset(Config.TRAIN_H5, subset_ratio=Config.TRAIN_SUBSET_RATIO)
    val_ds = TransformerDataset(Config.VAL_H5, subset_ratio=1.0)
    out["train_sequences"] = len(train_ds)
    out["val_sequences"] = len(val_ds)

    log("\n[diag] --- SANITY FLOOR: what trivial predictors score on the training objective ---")
    out["null_baselines"] = {}
    for tok, frame_level in (("token", False), ("frame", True)):
        nb = null_baselines(val_ds.data, Config, frame_level=frame_level)
        out["null_baselines"][tok] = nb
        log(f"  tokenization={tok}  (target std = {nb['_target_std']:.6f})")
        for name in [k for k in nb if not k.startswith("_")]:
            log(f"    {name:<16} l2norm={nb[name]['l2norm']:.6f}  mse={nb[name]['mse']:.3e}")
    log("  Compare a run's `train_loss` against these. A train loss ABOVE the "
        "zero-predictor means the model has learned nothing at all, and the cause is "
        "conditioning or optimisation -- not capacity, not exposure bias, not "
        "architecture.")

    log("\n[diag] --- is there learnable temporal structure beyond persistence? ---")
    lin = linear_frame_baseline(train_ds.data, val_ds.data, Config, device, log=log)
    out["linear_baseline"] = lin
    log(f"  persistence MSE   = {lin['persistence_mse']:.8f}")
    log(f"  linear-map MSE    = {lin['linear_mse']:.8f}")
    log(f"  linear IMPROVEMENT over persistence = {lin['improvement_pct']:+.2f}% "
        f"(1 frame ahead: {lin['improvement_pct_frame1']:+.2f}%)")
    log("  Read this as the floor a competent model must clear. If the sweep's best "
        "arm cannot beat a ridge regression, the model is the problem; if the linear "
        "map is also ~0%, persistence is genuinely strong at this dt and the framing "
        "must change.")

    # Statistics that decide the over- vs under-fitting question for free.
    mean, std = compute_feature_stats(train_ds.data, Config, frame_level=False)
    out["feature_stats"] = {
        "latent_std_mean": float(std[:Config.LATENT_DIM].mean()),
        "latent_std_max": float(std[:Config.LATENT_DIM].max()),
        "meta_cols_mean": [round(float(v), 4) for v in mean[Config.LATENT_DIM:]],
        "meta_cols_std": [round(float(v), 4) for v in std[Config.LATENT_DIM:]],
    }
    log(f"\n[diag] latent std (mean over 47 dims) = {out['feature_stats']['latent_std_mean']:.4f}")
    log(f"[diag] columns 47:52 mean = {out['feature_stats']['meta_cols_mean']}")
    log(f"[diag] columns 47:52 std  = {out['feature_stats']['meta_cols_std']}")
    log("  (Those columns went into the same nn.Linear as latents with std "
        f"~{out['feature_stats']['latent_std_mean']:.3f}; NORMALIZE_FEATURES fixes the mismatch.)")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "diagnostics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\n[diag] written to {path}")
    return out


# --------------------------------------------------------------------------- #
# v2.0 pinning smoke path
# --------------------------------------------------------------------------- #
def run_smoke_test(args):
    """Build the 80-frame model and prove the pinning is coherent.

    This runs BEFORE any real training loop and touches NO HDF5 data. It:

      1. Builds `get_model(Config)` at the pinned `NUM_TIME=80` /
         `SEQ_LEN=800` shape (v3.1: NUM_X restricted 26->10).
      2. Calls `probe_causality(model, Config, device)` and asserts
         `causal is True`. If the probe fails, this function raises
         SystemExit BEFORE the first optimizer step -- which is the whole
         point of the gate.
      3. Only then runs a handful (`--smoke-steps`, default 3) of
         `micro_batch=1` forward + backward + step cycles on synthetic
         (800-token) tensors, to prove the shapes flow end-to-end at the
         new sequence length.

    Intentionally minimal: no data loader, no checkpoint I/O, no W&B. This
    is the shape/causality gate the pinning step needs -- the device-adaptive
    regime and warm-start live in later steps.
    """
    device = Config.DEVICE if args.device is None else args.device
    log = print
    # Print the device-adaptive regime banner even in the smoke path so the
    # colored banner is exercised end-to-end during CPU validation.
    regime = resolve_train_regime(device)
    print(regime.banner, flush=True)
    log(f"[smoke] regime: device={regime.device} micro_batch={regime.micro_batch} "
        f"virtual_batch={regime.virtual_batch} use_amp={regime.use_amp} "
        f"compile={regime.compile_model}")
    log(f"[smoke] pinned v3.1: NUM_TIME={Config.NUM_TIME}, "
        f"NUM_X={Config.NUM_X}, SEQ_LEN={Config.SEQ_LEN} "
        f"(expected 800), device={device}")
    if (Config.NUM_TIME, Config.NUM_X, Config.SEQ_LEN) != (80, 10, 800):
        raise SystemExit(f"[smoke] pinning invariant violated: "
                         f"NUM_TIME={Config.NUM_TIME}, NUM_X={Config.NUM_X}, "
                         f"SEQ_LEN={Config.SEQ_LEN}")

    torch.manual_seed(Config.SEED)
    model = get_model(Config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[smoke] built {type(model).__name__} "
        f"({n_params/1e6:.2f} M params, frame_native={getattr(model, 'frame_native', False)})")

    # -- causality gate ----------------------------------------------------
    probe = probe_causality(model, Config, device)
    log(f"[smoke] probe_causality: before_cut={probe['max_change_before_cut']:.3e} "
        f"after_cut={probe['max_change_after_cut']:.3e} "
        f"tol={probe['tolerance']:.3e} causal={probe['causal']}")
    if not probe["causal"]:
        raise SystemExit(
            f"[smoke] causality gate FAILED at NUM_TIME={Config.NUM_TIME}: "
            f"outputs before the cut moved {probe['max_change_before_cut']:.3e} "
            f"(tolerance {probe['tolerance']:.3e}). Refusing to run optimizer "
            "steps on a leaky model."
        )
    if not probe["probe_responsive"]:
        raise SystemExit("[smoke] probe non-responsive: post-cut outputs did not move; "
                         "the probe is not exercising the model.")
    log("[smoke] causality gate PASSED. Running "
        f"{args.smoke_steps} x micro_batch=1 forward+backward on synthetic data.")

    # -- micro_batch=1 shakedown ------------------------------------------
    frame_native = bool(getattr(model, 'frame_native', False))
    if frame_native:
        # (B, NUM_TIME, frame_dim + FRAME_META_COLS)
        width = Config.NUM_X * Config.LATENT_DIM + FRAME_META_COLS
        seq_len = Config.NUM_TIME
    else:
        width = Config.INPUT_DIM
        seq_len = Config.SEQ_LEN

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for step in range(int(args.smoke_steps)):
        x = torch.randn(1, seq_len, width, device=device)
        y = torch.randn_like(x[..., :model.output_head.out_features])
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        log(f"[smoke]   step {step+1}/{args.smoke_steps}: loss={loss.item():.4e}")

    log("[smoke] OK: 80-frame model builds, is causal, and trains one micro-batch at a time.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default=None, help="arm name (see --list-arms)")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--list-arms", action="store_true")
    p.add_argument("--diagnostics-only", action="store_true",
                   help="run the causality probes and linear baseline, then exit")
    p.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sweep_logs", "manual"))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=None)
    p.add_argument("--subset-ratio", type=float, default=None)
    p.add_argument("--rollout-seqs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--cpu-data", action="store_true",
                   help="keep the dataset in host memory instead of on the GPU")
    p.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint")
    p.add_argument("--allow-leak", action="store_true",
                   help="do not abort when the causality probe fails")
    p.add_argument("--warm-start", default=DEFAULT_WARM_START_CKPT,
                   metavar="CKPT",
                   help="v1.0 checkpoint to warm-start the 80-frame model from "
                        "(default: r1_a3b_delta_ar_rollout_best.pt). Loaded with "
                        "strict=False after sanitising length-dependent tensors; "
                        "any missing/unexpected keys outside the allowlist are a "
                        "hard failure. Ignored when a resume checkpoint exists.")
    p.add_argument("--no-warm-start", action="store_true",
                   help="do NOT warm-start from any v1.0 checkpoint (cold start)")
    p.add_argument("--smoke-test", action="store_true",
                   help="v2.0 pinning smoke path: build the 80-frame model, prove "
                        "probe_causality(...)['causal'] is True, then run a handful of "
                        "micro_batch=1 forward+backward steps on synthetic data. NO "
                        "optimizer step runs before the causality gate passes; no HDF5 "
                        "data is touched. Exits nonzero on causality failure.")
    p.add_argument("--smoke-steps", type=int, default=3,
                   help="number of micro_batch=1 forward+backward passes in --smoke-test")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="ad-hoc Config overrides, applied after the arm")
    return p


def _coerce(raw):
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    return raw


def main(argv=None):
    args = build_parser().parse_args(argv)
    print_device_detection_banner(args.device)

    if args.list_arms:
        for rnd in (1, 2):
            print(f"\n=== ROUND {rnd} ===")
            if rnd == 1:
                for name, spec in ROUND1_ARMS.items():
                    print(f"  {name:<16} {spec['desc']}")
            else:
                for branch, blob in ROUND2_ARMS.items():
                    if not blob["arms"]:
                        continue
                    print(f"  -- branch {branch}: {blob['title']}")
                    for name, spec in blob["arms"].items():
                        print(f"     {name:<20} {spec['desc']}")
        return 0

    Config.SWEEP_ROUND = args.round
    # No-arg default (plan Step 2 "No-arg default on Mac"): running
    #   python train_production_transformer_deep_dive.py
    # with zero CLI flags launches the v1.0 winner arm (`a3b_delta_ar`) --
    # its default `--warm-start` target (`r1_a3b_delta_ar_rollout_best.pt`)
    # is already on disk, so this pairs a coherent arm with a coherent
    # warm-start on the same line. `--arm NAME` / `--diagnostics-only` /
    # `--smoke-test` / `--list-arms` remain explicit overrides.
    if not args.arm and not (args.list_arms or args.diagnostics_only or args.smoke_test):
        args.arm = "a3b_delta_ar"
    if args.arm:
        apply_arm(args.arm)

    # CLI beats the arm, so a launcher can impose one shared budget on every arm.
    for attr, value in (("MAX_STEPS", args.max_steps), ("MAX_HOURS", args.max_hours),
                        ("VAL_EVERY_STEPS", args.val_every), ("SEED", args.seed),
                        ("BATCH_SIZE", args.batch_size),
                        ("ACCUMULATION_STEPS", args.accum),
                        ("TRAIN_SUBSET_RATIO", args.subset_ratio),
                        ("VAL_ROLLOUT_SEQS", args.rollout_seqs),
                        ("DEVICE", args.device)):
        if value is not None:
            setattr(Config, attr, value)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if not hasattr(Config, k):
            raise SystemExit(f"--set: unknown Config field {k!r}")
        if k in PINNED_CONFIG_FIELDS:
            raise SystemExit(
                f"--set cannot override pinned v2.0 Config field {k!r}")
        setattr(Config, k, _coerce(v))

    # Derived fields, recomputed after every override so the `config` dict stored
    # in each checkpoint is self-consistent. VAL_ROLLOUT_STEPS in particular is
    # read back by tests/test_model_vs_baseline.py to size its horizon, and a
    # stale 728 against an overridden VAL_CONTEXT_STEPS would silently mis-scope
    # the evaluation.
    Config.SEQ_LEN = Config.NUM_X * Config.NUM_TIME
    Config.VAL_ROLLOUT_STEPS = Config.NUM_X * (Config.NUM_TIME - Config.VAL_CONTEXT_STEPS)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.diagnostics_only:
        run_diagnostics(args)
        return 0

    if args.smoke_test:
        return run_smoke_test(args)

    if not args.arm:
        # Unreachable under the no-arg default above; kept as a defensive
        # guard so any future code path that clears args.arm still fails
        # loudly rather than launching an unconfigured run.
        raise SystemExit("need --arm NAME (or --diagnostics-only / --list-arms / --smoke-test)")
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
