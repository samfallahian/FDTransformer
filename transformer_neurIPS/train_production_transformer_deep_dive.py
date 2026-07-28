"""
Deep-dive transformer trainer for the NeurIPS latent fluid-dynamics sequences.

This is the ARM-DRIVEN rewrite. One process trains exactly one "arm" (a named
set of config overrides); `sweep_deep_dive.py` runs several arms concurrently on
one box and aggregates the results into a single uploadable report.

    # answer the diagnostic questions once, cheaply, before spending GPU-hours
    python train_production_transformer_deep_dive.py --diagnostics-only

    # train one arm
    python train_production_transformer_deep_dive.py --arm a0_control --max-steps 6000

    # list what exists
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
"""

import argparse
import json
import math
import os
import sys
import time

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
# Configuration
# --------------------------------------------------------------------------- #
class Config:
    """Defaults for a single run. Arms (below) override fields on this class."""

    # -- data ---------------------------------------------------------------
    TRAIN_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/train_40.h5")
    VAL_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/val_40.h5")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

    LATENT_DIM = 47      # latent features per (t, x) location
    NUM_X = 26           # x-locations per time frame
    NUM_TIME = 40        # time frames per sequence
    SEQ_LEN = NUM_X * NUM_TIME              # 1040 tokens
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
    TOKENIZATION = 'token'     # token (1040 tokens) | frame (40 tokens)
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
    MAX_STEPS = 6000           # OPTIMIZER steps -- the primary clock
    MAX_HOURS = 12.0           # wall-clock safety net only

    # -- rollout-stability techniques --------------------------------------
    NOISE_STD = 5e-4           # gaussian noise on fed-in latents
    AR_MODE = 'none'           # none | frame_ar | sched
    AR_LOSS_WEIGHT = 0.0
    AR_WEIGHT_WARMUP_FRAC = 0.2   # ramp AR weight 0 -> AR_LOSS_WEIGHT over this fraction
    AR_FRAMES = 2              # horizon in whole TIME FRAMES (26 tokens each)
    AR_SEQS = 4                # sequences used for the sequential AR loop
    AR_EVERY_N_STEPS = 4
    AR_DETACH_FEEDBACK = True  # truncate gradient through the fed-back token
    SCHED_SAMPLING_P = 0.25    # replacement probability when AR_MODE='sched'

    # -- evaluation ---------------------------------------------------------
    VAL_CONTEXT_STEPS = 12                      # frames fed as context
    VAL_ROLLOUT_STEPS = NUM_X * (NUM_TIME - VAL_CONTEXT_STEPS)   # 728 tokens
    VAL_ROLLOUT_SEQS = 64      # fixed row set; model AND persistence both use it
    VAL_EVERY_STEPS = 400
    LOG_EVERY_STEPS = 25
    CHECKPOINT_EVERY_STEPS = 400

    # -- runtime ------------------------------------------------------------
    USE_TF32 = True
    USE_CUDNN_BENCHMARK = True
    AMP = True
    SAVE_SCRIPTED_MODELS = False
    SEED = 1337
    ARM = 'a0_control'
    SWEEP_ROUND = 1
    WANDB_PROJECT = "runpod_b300_deepdive"


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
        "desc": "Frame-level tokenisation: 40 tokens of 26x47 features instead of "
                "1040 tokens of 47.",
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
        "title": "Frame tokenisation won -- rebuild around 40-token sequences",
        "arms": {
            "f1_frame_delta": {"desc": "Frame + delta.",
                               "overrides": {"TOKENIZATION": "frame", "PREDICT_DELTA": True}},
            "f2_frame_deep": {"desc": "Frame + E512/L12 (40 tokens is cheap, so spend it on depth).",
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
        setattr(Config, k, v)
    Config.ARM = name
    return spec


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def l2_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))


def base_loss(pred, target, cfg=Config):
    kind = getattr(cfg, 'LOSS', 'l2norm')
    if kind == 'l2norm':
        return l2_loss(pred, target)
    if kind == 'mse':
        return mse_loss(pred, target)
    if kind == 'huber':
        return F.huber_loss(pred, target, delta=getattr(cfg, 'HUBER_DELTA', 0.01))
    raise ValueError(f"Unknown LOSS {kind!r}; expected 'l2norm', 'mse' or 'huber'")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class TransformerDataset(torch.utils.data.Dataset):
    """The whole (small) split, read from HDF5 once and held as one tensor.

    Full train is ~15k x 1040 x 52 float32 ~= 3.2 GB, which fits in RAM and on
    the GPU, so there is no reason to re-open the HDF5 file per worker per epoch.
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
    """What trivial predictors score on the TEACHER-FORCED training objective.

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

    Solved on the normal equations in float64. D = 26*47 = 1222, so X'X is
    1223x1223 -- trivial regardless of how many transitions are accumulated.
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
        return base_loss(torch.cat(preds, 1),
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
    return base_loss(torch.cat(preds, 1),
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
    return base_loss(model(inp), tgt, cfg)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, val_data, cfg, device, amp_dtype=None, chunk=32):
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
    tf_loss = tf_mse = 0.0
    tf_n = 0
    for start in range(0, val_data.shape[0], cfg.EVAL_BATCH_SIZE):
        b = val_data[start:start + cfg.EVAL_BATCH_SIZE]
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


def save_checkpoint(path, model, optimizer, step, extra):
    payload = {
        'step': step,
        'epoch': step,          # back-compat: the leaderboard test reads 'epoch'
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'config': config_dict(),
    }
    payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)       # atomic: a killed run never leaves a half file


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args, log=print):
    t_start = time.time()
    device = Config.DEVICE
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    if Config.USE_TF32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if Config.USE_CUDNN_BENCHMARK and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    amp_dtype = None
    use_scaler = False
    if Config.AMP and device == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16          # no GradScaler needed, no overflow
        else:
            amp_dtype = torch.float16
            use_scaler = True

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

    train_loader = InMemoryBatcher(train_ds.data, Config.BATCH_SIZE, shuffle=True,
                                   generator=data_gen)
    stream = infinite_batches(train_loader)

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
            missing, unexpected = model.load_state_dict(ck['model_state_dict'], strict=False)
            if ck.get('optimizer_state_dict'):
                optimizer.load_state_dict(ck['optimizer_state_dict'])
            step = int(ck.get('step', 0))
            for _ in range(step):
                scheduler.step()
            best.update({k: v for k, v in ck.get('best', {}).items()})
            log(f"  [resume] {latest_path} at step {step} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            log(f"  [resume] failed ({type(e).__name__}: {e}); starting fresh")

    tel = _Telemetry(
        not args.no_wandb, project=Config.WANDB_PROJECT, name=run_name, id=run_name,
        resume="allow", config=config_dict())

    ar_mode = Config.AR_MODE
    ar_target_w = float(Config.AR_LOSS_WEIGHT)
    ar_warm = max(1, int(Config.MAX_STEPS * float(Config.AR_WEIGHT_WARMUP_FRAC)))
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
        f"micro_batch={Config.BATCH_SIZE} x accum={Config.ACCUMULATION_STEPS} "
        f"= effective {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}  "
        f"amp={amp_dtype}  loss={Config.LOSS}")

    stop_reason = "completed"
    # Rate is measured from where this PROCESS started, not from step 0, so an
    # ETA after resuming a checkpoint is not diluted by the earlier run's steps.
    start_step_for_rate = step
    while step < Config.MAX_STEPS:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc = base_acc = ar_acc = 0.0
        ar_w = ar_target_w * min(1.0, (step + 1) / ar_warm) if ar_mode != 'none' else 0.0
        run_ar = (ar_mode != 'none' and ar_w > 0
                  and step % max(1, int(Config.AR_EVERY_N_STEPS)) == 0)

        for micro in range(Config.ACCUMULATION_STEPS):
            batch = next(stream)
            if batch.device.type != device.split(':')[0]:
                batch = batch.to(device, non_blocking=True)

            amp_ctx = (torch.autocast(device_type='cuda', dtype=amp_dtype)
                       if amp_dtype is not None
                       else torch.autocast(device_type='cpu', enabled=False))
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
                loss = loss / Config.ACCUMULATION_STEPS

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if Config.GRAD_CLIP and Config.GRAD_CLIP > 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        else:
            grad_norm = torch.tensor(float('nan'))

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        step += 1

        train_loss = base_acc / Config.ACCUMULATION_STEPS
        best["train_loss"] = min(best["train_loss"], train_loss)

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
                flag = f"  <-- WORSE THAN PREDICTING ZERO ({floor:.6f})"
            elif train_loss > anchor:
                flag = f"  <-- worse than the previous-frame anchor ({anchor:.6f})"
            elapsed = time.time() - t_start
            eta = ""
            if step >= 3:
                per_step = elapsed / max(1, step - start_step_for_rate)
                eta = f"  eta~{per_step * (Config.MAX_STEPS - step) / 3600:.1f}h"
            log(f"  step {step:>6}/{Config.MAX_STEPS}  train={train_loss:.6f}  "
                f"lr={lr:.2e}  gnorm={float(grad_norm):.3f}  "
                f"{elapsed / 60:.1f}m{eta}{flag}")

        hit_budget = step >= Config.MAX_STEPS
        if step % Config.VAL_EVERY_STEPS == 0 or hit_budget:
            m = evaluate(model, val_ds.data, Config, device, amp_dtype=amp_dtype)
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
                     'best': dict(best)})
                log(f"  --> new best rollout ({m['rollout_mse']:.6f}, "
                    f"{m['improvement_pct']:+.2f}% vs persistence)")
            if m["val_tf_mse"] < best["val_tf_mse"]:
                best["val_tf_mse"] = m["val_tf_mse"]
                save_checkpoint(
                    os.path.join(Config.CHECKPOINT_DIR, f"{run_name}_best.pt"),
                    model, optimizer, step,
                    {'val_l2': m['val_tf_loss'], 'rollout_mse': m['rollout_mse'],
                     'improvement': m['improvement_pct'], 'train_l2': train_loss,
                     'best': dict(best)})
            tel.set_summary("best_rollout_mse", best["rollout_mse"])
            tel.set_summary("best_improvement_pct", best["improvement_pct"])

        if step % Config.CHECKPOINT_EVERY_STEPS == 0 or hit_budget:
            save_checkpoint(latest_path, model, optimizer, step,
                            {'train_l2': train_loss, 'best': dict(best),
                             'val_l2': best['val_tf_mse'],
                             'rollout_mse': best['rollout_mse'],
                             'improvement': best['improvement_pct']})

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

    if not args.arm:
        raise SystemExit("need --arm NAME (or --diagnostics-only / --list-arms)")
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
