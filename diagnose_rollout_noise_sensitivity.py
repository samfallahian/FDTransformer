#!/usr/bin/env python3
"""
Stage 0 diagnostic: is the catastrophic rollout divergence a butterfly-effect
instability, or a systematic drift?

Background
==========
`train_production_transformer_deep_dive.py`'s autoregressive rollout eval
(`evaluate()`, `per_epoch_persistence_report()`) feeds each step's prediction
back in as the next input. Across a real training run, that rollout has been
measured at ~95-97 (m/s)^2 centroid-velocity MSE against a persistence
baseline of ~0.0028 -- roughly 3 orders of magnitude worse, and getting worse
with horizon, not flat. Two very different root causes produce that same
symptom, and they call for different fixes:

  1. CHAOTIC INSTABILITY -- the model's repeated self-application is a
     numerically unstable operator: even a tiny perturbation to what's fed
     back grows explosively with horizon (sensitive dependence on initial
     conditions). If this is the story, noise-robustness training
     (NOISE_STD, AR_FEEDBACK_NOISE_STD) and regularization (weight decay,
     dropout) are the right lever -- they directly train against sensitivity.

  2. SYSTEMATIC DRIFT -- the model has a consistent directional bias (e.g.
     it under- or over-shoots the same way every step), and that bias
     accumulates linearly/superlinearly regardless of noise. If this is the
     story, noise training won't help much -- the fix is more likely a
     longer AR training horizon (so the model sees and corrects its own
     accumulated bias during training) or a bias-correction mechanism.

This script tells you which one you're looking at, cheaply, with NO training
-- it just perturbs the eval-time rollout at varying noise levels and looks
at how error grows with horizon and with noise level, and whether the error
is dominated by its mean (bias) or its spread (variance).

Usage
=====
    python transformer_neurIPS/diagnose_rollout_noise_sensitivity.py
    python transformer_neurIPS/diagnose_rollout_noise_sensitivity.py \\
        --checkpoint saved_models/r1_a3b_delta_ar_latest.pt --n-seqs 32

Reads ONLY -- never writes a checkpoint, never trains. Runs happily on
MPS/CPU (no_grad throughout, no retained-graph AR loop).
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import train_production_transformer_deep_dive as T  # noqa: E402

Config = T.Config


def _c(text, color):
    colors = {"red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
              "cyan": "\033[96m", "bold": "\033[1m", "reset": "\033[0m"}
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"{colors.get(color, '')}{text}{colors['reset']}"


def _sparkline(values):
    if not values:
        return ""
    blocks = "._-=+*#%@"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return blocks[len(blocks) // 2] * len(values)
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / (hi - lo) * (len(blocks) - 1)))]
                   for v in values)


def _strip_compile_prefix(state_dict):
    """Strip torch.compile / DDP wrapper prefixes from state-dict keys.

    `train()` may run under `torch.compile(model)` before `save_checkpoint`
    calls `model.state_dict()`, which prefixes every key with `_orig_mod.`.
    A freshly-built (uncompiled) model here would otherwise silently show
    100% missing keys. Mirrors `tests/test_model_vs_baseline.py`'s
    `normalize_state_dict` (not importable from a test module cleanly, so
    duplicated -- it's four lines).
    """
    for prefix in ("_orig_mod.", "module."):
        if state_dict and all(k.startswith(prefix) for k in state_dict):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(ckpt_path, device):
    """Rebuild the model at the checkpoint's OWN saved config (architecture
    may differ from whatever Config currently holds), load weights, return
    (model, cfg_used). Self-contained -- does not import from tests/.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = ck.get("config", {})
    for k, v in saved_cfg.items():
        if k in ("TRAIN_H5", "VAL_H5", "CHECKPOINT_DIR", "DEVICE"):
            continue
        if hasattr(Config, k):
            setattr(Config, k, v)
    model = T.get_model(Config).to(device)
    sd = _strip_compile_prefix(ck["model_state_dict"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    benign = {"causal_mask", "feat_mean", "feat_std"}
    bad_missing = [k for k in missing if k not in benign]
    if bad_missing:
        print(_c(f"  WARNING: {len(bad_missing)} non-benign missing key(s): "
                 f"{bad_missing[:4]}", "yellow"))
    model.eval()
    return model, Config


@torch.no_grad()
def noisy_rollout(model, batch, cfg, ctx_frames, n_frames, noise_std,
                  generator=None):
    """Copy of the token-tokenization branch of `rollout_frames()`, with
    Gaussian noise of std `noise_std` added to the fed-back prediction at
    every step (diagnostic only -- production `rollout_frames()` is
    untouched by this script). Frame-native models are not covered; the
    arm this was written to investigate (`a3b_delta_ar`) is token-native.
    """
    NX, LD = cfg.NUM_X, cfg.LATENT_DIM
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
        fed = nxt
        if noise_std > 0:
            fed = fed + noise_std * torch.randn(
                fed.shape, device=fed.device, dtype=fed.dtype, generator=generator)
        tok[:, :, :LD] = fed
        curr = torch.cat([curr, tok], dim=1)
    out = torch.cat(preds, dim=1)
    n_done = out.shape[1] // NX
    B = batch.shape[0]
    return out[:, :n_done * NX, :].reshape(B, n_done, NX, LD)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="default: saved_models/{r1_a3b_delta_ar}_latest.pt")
    p.add_argument("--val-h5", default=None, help="default: data/val_80.h5")
    p.add_argument("--n-seqs", type=int, default=16)
    p.add_argument("--noise-levels", default="0,1e-5,1e-4,1e-3,1e-2,1e-1")
    p.add_argument("--context-steps", type=int, default=None,
                   help="default: Config.VAL_CONTEXT_STEPS")
    p.add_argument("--device", default=None, help="default: auto-detect")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else T.pick_device()
    ckpt_path = args.checkpoint or os.path.join(
        Config.CHECKPOINT_DIR, "r1_a3b_delta_ar_latest.pt")
    val_h5 = args.val_h5 or os.path.join(_HERE, "data", "val_80.h5")

    print(_c(f"[diag] device={device}  checkpoint={ckpt_path}", "cyan"))
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    if not os.path.exists(val_h5):
        raise SystemExit(f"val data not found: {val_h5}")

    model, cfg = load_model_from_checkpoint(ckpt_path, device)
    ctx_frames = args.context_steps if args.context_steps is not None \
        else int(cfg.VAL_CONTEXT_STEPS)
    n_frames = int(cfg.NUM_TIME) - ctx_frames
    print(f"[diag] arch: NUM_TIME={cfg.NUM_TIME} NUM_X={cfg.NUM_X} "
          f"ctx_frames={ctx_frames} rollout_frames={n_frames} "
          f"tokenization={getattr(cfg, 'TOKENIZATION', 'token')}")

    with h5py.File(val_h5, "r") as f:
        raw = f["data"][:args.n_seqs]
    n_seqs, T_frames, NX, feat = raw.shape
    batch = torch.from_numpy(raw).float().reshape(n_seqs, T_frames * NX, feat).to(device)
    gt = batch[:, ctx_frames * NX:, :cfg.LATENT_DIM].reshape(
        n_seqs, T_frames - ctx_frames, NX, cfg.LATENT_DIM)[:, :n_frames]
    gt_v = T.decode_centroid(gt, cfg)   # (n_seqs, n_frames, NX, 3)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    print()
    print(_c(f"{'noise_std':>10} {'rmse_f1':>10} {'rmse_mid':>10} {'rmse_last':>10} "
             f"{'|bias|/rmse':>12}  per-frame RMSE trend", "bold"))
    print("-" * 100)

    results = []
    for noise_std in noise_levels:
        pred_v = T.decode_centroid(
            noisy_rollout(model, batch, cfg, ctx_frames, n_frames, noise_std,
                          generator=generator), cfg)
        err = (pred_v - gt_v).cpu().numpy()   # (n_seqs, k, NX, 3)
        k = err.shape[1]
        per_frame_rmse = np.sqrt((err ** 2).mean(axis=(0, 2, 3)))
        per_frame_bias = err.mean(axis=(0, 2, 3))   # signed mean, can cancel
        overall_rmse = float(np.sqrt((err ** 2).mean()))
        overall_bias = float(np.abs(err.mean()))
        bias_ratio = overall_bias / overall_rmse if overall_rmse > 0 else 0.0
        results.append((noise_std, per_frame_rmse, per_frame_bias, bias_ratio))

        mid = k // 2
        print(f"{noise_std:>10.1e} {per_frame_rmse[0]:>10.4g} "
              f"{per_frame_rmse[mid]:>10.4g} {per_frame_rmse[-1]:>10.4g} "
              f"{bias_ratio:>12.3f}  |{_sparkline(list(per_frame_rmse))}|")

    print()
    base_rmse = results[0][1][-1] if results else float("nan")
    max_rmse = max(r[1][-1] for r in results)
    growth = max_rmse / base_rmse if base_rmse > 0 else float("inf")
    avg_bias_ratio = float(np.mean([r[3] for r in results]))

    print(_c("[verdict]", "bold"))
    print(f"  last-frame RMSE grows {growth:.2f}x from noise_std="
          f"{results[0][0]:.1e} to {results[-1][0]:.1e}")
    print(f"  mean |bias|/RMSE across noise levels: {avg_bias_ratio:.3f} "
          f"(1.0 = pure systematic bias, 0.0 = pure symmetric variance)")
    print()
    if growth > 3.0 and avg_bias_ratio < 0.3:
        print(_c("  -> CHAOTIC/SENSITIVITY-DOMINATED: error grows sharply with "
                 "injected noise and is NOT mostly a constant bias.", "green"))
        print("     Noise-robustness training (NOISE_STD, AR_FEEDBACK_NOISE_STD) "
              "and regularization (weight decay/dropout) are well-targeted here.")
    elif avg_bias_ratio > 0.6:
        print(_c("  -> BIAS-DOMINATED: error is mostly a consistent directional "
                 "drift, largely independent of injected noise.", "yellow"))
        print("     Noise training is unlikely to fix this alone. A longer AR "
              "training horizon (so the model corrects its own accumulated bias "
              "during training) is the better-targeted lever -- see a4b_ar_very_long.")
    else:
        print(_c("  -> MIXED: neither signature dominates cleanly.", "cyan"))
        print("     Worth trying both families (a5b_wd_heavy / e6_sched_noise for "
              "the noise hypothesis, a4b_ar_very_long / e3_ar_long for the horizon "
              "hypothesis) rather than betting on one.")


if __name__ == "__main__":
    main()
