"""
Training Pipeline Diagnostics for the NeurIPS Transformer
=========================================================

Run this over SSH ON THE B300 BOX to find out *exactly* where training time is
being spent. It does NOT modify any weights or checkpoints; it just times each
phase of one training step and prints a table.

Usage (on the remote box):
    cd /path/to/repo
    python -m transformer_neurIPS.run_diagnostics                 # default variant 0
    python -m transformer_neurIPS.run_diagnostics --variant 5     # try a specific one
    python -m transformer_neurIPS.run_diagnostics --batch-size 1024 --num-workers 16
    python -m transformer_neurIPS.run_diagnostics --no-compile    # skip torch.compile
    python -m transformer_neurIPS.run_diagnostics --profile       # dump a chrome trace

What it measures (per-phase, in ms, averaged over N warm iters):
    - dataloader wait     : time the GPU is idle waiting for a CPU batch
    - h2d copy            : host->device transfer (should be tiny w/ pin_memory)
    - fwd (teacher-force) : the main model forward
    - bwd + opt           : autograd + optimizer.step
    - AR-rollout loss     : the sequential auxiliary loop (should be optional)
    - val fwd (no-grad)   : one eval-mode forward
    - jit.trace + to(cpu) : the "robust checkpoint" trace stall
    - one full step total : end-to-end
It also prints:
    - Which SDPA backend PyTorch is actually picking (Flash / mem-efficient / math)
    - Peak VRAM used and % of card capacity
    - Achieved tokens/sec vs. a rough B300 roofline
    - Verdict + recommendation
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import time
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow both `python -m transformer_neurIPS.run_diagnostics` and direct execution.
try:
    from transformer_neurIPS.train_production_transformer import (
        Config, TransformerDataset, l2_loss,
    )
    from transformer_neurIPS.model_variants import get_model
except ImportError:  # direct execution: `python run_diagnostics.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformer_neurIPS.train_production_transformer import (
        Config, TransformerDataset, l2_loss,
    )
    from transformer_neurIPS.model_variants import get_model


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_ms(fn: Callable, iters: int = 5, warmup: int = 2) -> Dict[str, float]:
    """Time a zero-arg callable. Returns mean/median/min/max ms."""
    # Warmup (compilation, autotune, allocator caching)
    for _ in range(warmup):
        fn()
    cuda_sync()
    samples: List[float] = []
    for _ in range(iters):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "n": iters,
    }


def fmt_row(name: str, ms: Dict[str, float], total_ms: float | None = None) -> str:
    frac = f"{100.0 * ms['mean'] / total_ms:5.1f}%" if total_ms else "     "
    return (f"  {name:<28} {ms['mean']:9.2f} ms   "
            f"(median {ms['median']:8.2f}, min {ms['min']:8.2f}, "
            f"max {ms['max']:8.2f})   {frac}")


# ---------------------------------------------------------------------------
# SDPA / FlashAttention detection
# ---------------------------------------------------------------------------

def detect_sdpa_backend(embed_size: int, n_heads: int) -> str:
    """Query which SDPA backend PyTorch would pick for this shape on this device."""
    if not torch.cuda.is_available():
        return "cpu (no CUDA)"
    device = "cuda"
    B, H, T, D = 4, n_heads, 512, embed_size // n_heads
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return "unknown (old torch)"

    backends = []
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            backends.append(name)
        except Exception:
            pass
    if not backends:
        return "NONE (kernel failed)"
    # PyTorch's dispatcher picks the earliest-listed available backend, so
    # showing all supported ones is what matters.
    return " > ".join(backends)


# ---------------------------------------------------------------------------
# Main diagnostic run
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=0,
                    help="Index into Config.SEARCH_SPACE (default 0).")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--no-compile", action="store_true",
                    help="Skip torch.compile (useful to isolate its cost).")
    ap.add_argument("--no-ar", action="store_true",
                    help="Skip the AR-rollout timing (already known to be slow).")
    ap.add_argument("--profile", action="store_true",
                    help="Also dump a PyTorch profiler chrome trace to ./diag_trace.json.")
    args = ap.parse_args()

    # Apply variant + overrides
    v = Config.SEARCH_SPACE[args.variant]
    Config.EMBED_SIZE = v["EMBED_SIZE"]
    Config.N_HEADS = v["N_HEADS"]
    Config.N_LAYERS = v["N_LAYERS"]
    Config.VARIANT = v["VARIANT"]
    if "USE_SWIGLU" in v:
        Config.USE_SWIGLU = v["USE_SWIGLU"]
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        Config.NUM_WORKERS = args.num_workers

    device = Config.DEVICE
    print(f"\n=== NeurIPS Transformer Diagnostics ===")
    print(f"Device         : {device}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        total_gb = p.total_memory / 1e9
        print(f"GPU            : {p.name}  ({total_gb:.1f} GB, cc={p.major}.{p.minor}, SMs={p.multi_processor_count})")
    print(f"Torch          : {torch.__version__}")
    print(f"Variant        : #{args.variant}  {Config.VARIANT}  "
          f"E={Config.EMBED_SIZE} L={Config.N_LAYERS} H={Config.N_HEADS}")
    print(f"Batch size     : {Config.BATCH_SIZE}   seq_len={Config.SEQ_LEN}   "
          f"tokens/batch={Config.BATCH_SIZE * (Config.SEQ_LEN - 1):,}")
    print(f"Num workers    : {Config.NUM_WORKERS}   pin_memory={Config.PIN_MEMORY}   "
          f"persistent_workers={getattr(Config, 'PERSISTENT_WORKERS', False)}")

    # Enable perf-friendly knobs so the numbers reflect production speed.
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # --- SDPA backend probe -------------------------------------------------
    print(f"\n[SDPA backends available for this shape]")
    print(f"  {detect_sdpa_backend(Config.EMBED_SIZE, Config.N_HEADS)}")
    if torch.cuda.is_available():
        try:
            from torch.backends.cuda import (
                flash_sdp_enabled, mem_efficient_sdp_enabled, math_sdp_enabled,
            )
            print(f"  flash={flash_sdp_enabled()}  "
                  f"mem_eff={mem_efficient_sdp_enabled()}  "
                  f"math={math_sdp_enabled()}")
        except Exception:
            pass

    # --- Model + optimizer --------------------------------------------------
    model = get_model(Config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params   : {n_params/1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = torch.amp.GradScaler(device="cuda", enabled=("cuda" in device))

    compile_ms = None
    if not args.no_compile and hasattr(torch, "compile") and device == "cuda":
        t0 = time.perf_counter()
        model = torch.compile(model)
        # Trigger compilation on a dummy tensor.
        dummy = torch.zeros(2, Config.SEQ_LEN - 1, Config.INPUT_DIM, device=device)
        with torch.amp.autocast(device_type="cuda"):
            _ = model(dummy)
        cuda_sync()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        print(f"torch.compile  : first-call compile time = {compile_ms:.0f} ms")

    # --- Data loading -------------------------------------------------------
    print(f"\n[Preparing dataset]")
    train_dataset = TransformerDataset(Config.TRAIN_H5, subset_ratio=1.0)
    print(f"  train samples = {len(train_dataset):,}")
    print(f"  ~batches/epoch = {len(train_dataset) // Config.BATCH_SIZE:,}")
    _persistent = (getattr(Config, "PERSISTENT_WORKERS", False)
                   and Config.NUM_WORKERS > 0)
    loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=_persistent,
        prefetch_factor=getattr(Config, "PREFETCH_FACTOR", 2) if Config.NUM_WORKERS > 0 else None,
    )
    it = iter(loader)

    # Prime the loader (first batch spawns workers + opens HDF5, don't count it).
    print("  priming DataLoader (spawning workers, opening HDF5)...")
    t0 = time.perf_counter()
    first_batch = next(it).to(device, non_blocking=True)
    prime_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  first-batch cold latency = {prime_ms:.0f} ms  "
          f"(one-off, workers stay alive if PERSISTENT_WORKERS=True)")

    # --- Prefetch a handful of batches to steady-state ---------------------
    batches = [first_batch]
    for _ in range(max(args.iters + 2, 6)):
        try:
            batches.append(next(it).to(device, non_blocking=True))
        except StopIteration:
            break
    cuda_sync()

    inputs_gpu = batches[0][:, :-1, :]
    targets_gpu = batches[0][:, 1:, :Config.LATENT_DIM]

    autocast_ctx = (torch.amp.autocast(device_type="cuda") if "cuda" in device
                    else contextlib.nullcontext())

    # ------------------------------------------------------------------ phases
    def phase_fwd():
        with autocast_ctx:
            out = model(inputs_gpu)
            loss = l2_loss(out, targets_gpu)
        return loss

    def phase_fwd_bwd():
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            out = model(inputs_gpu)
            loss = l2_loss(out, targets_gpu)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def phase_val_fwd():
        model.eval()
        try:
            with torch.no_grad(), autocast_ctx:
                _ = model(inputs_gpu)
        finally:
            model.train()

    def phase_dataload():
        # Time to pull the next CPU batch out of the DataLoader and copy to GPU.
        # If workers are keeping up, this should be ~0.
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        b.to(device, non_blocking=True)

    def phase_h2d():
        cpu_batch = torch.empty_like(batches[0], device="cpu").pin_memory()
        cpu_batch.copy_(batches[0].cpu())  # ensure a valid CPU tensor of same size
        cpu_batch.to(device, non_blocking=True)

    def phase_ar_rollout():
        # Reproduce the exact AR loop used in training (same shapes / weight).
        ar_batch_size = min(inputs_gpu.shape[0], 16)
        ar_context_len = min(inputs_gpu.shape[1], 128)
        ar_inputs = inputs_gpu[:ar_batch_size, :ar_context_len, :].clone()
        curr = ar_inputs
        with autocast_ctx:
            for _ in range(Config.AR_ROLLOUT_STEPS):
                out = model(curr)
                next_lat = out[:, -1:, :]
                next_idx = curr.shape[1]
                if next_idx >= inputs_gpu.shape[1]:
                    break
                next_tok = inputs_gpu[:ar_batch_size, next_idx:next_idx + 1, :].clone()
                next_tok[:, :, :Config.LATENT_DIM] = next_lat
                curr = torch.cat([curr, next_tok], dim=1)

    def phase_jit_trace_stall():
        # This is the "robust checkpoint" code path that the training script
        # used to hit every 60 seconds. We include it here so you can SEE
        # exactly how long that block stalls the GPU.
        m = model._orig_mod if hasattr(model, "_orig_mod") else model
        m.to("cpu")
        m.eval()
        dummy = torch.zeros(1, Config.SEQ_LEN - 1, Config.INPUT_DIM)
        try:
            with torch.no_grad():
                traced = torch.jit.trace(m, dummy, check_trace=False)
            del traced
        finally:
            m.to(device)
            m.train()

    # ----------------------------------------------------------- run the timings
    print(f"\n[Timing phases   iters={args.iters}   (means over post-warmup runs)]")
    results: Dict[str, Dict[str, float]] = {}

    results["dataloader wait (next batch)"] = time_ms(phase_dataload, iters=args.iters)
    results["h2d copy (pinned)"] = time_ms(phase_h2d, iters=args.iters)
    results["fwd (teacher-force, no bwd)"] = time_ms(phase_fwd, iters=args.iters)
    results["fwd + bwd + optimizer.step"] = time_ms(phase_fwd_bwd, iters=args.iters)
    results["val fwd (eval, no_grad)"] = time_ms(phase_val_fwd, iters=args.iters)
    if not args.no_ar and Config.AR_ROLLOUT_STEPS > 0:
        results[f"AR rollout aux ({Config.AR_ROLLOUT_STEPS} steps)"] = time_ms(
            phase_ar_rollout, iters=args.iters
        )
    # Only run the trace-stall probe a couple of times, it is expensive.
    results["jit.trace + move-to-CPU stall"] = time_ms(
        phase_jit_trace_stall, iters=max(2, args.iters // 3), warmup=1
    )

    # Sort by mean time for the printout, but keep totals meaningful.
    total_step_ms = (results["fwd + bwd + optimizer.step"]["mean"]
                     + max(0.0, results["dataloader wait (next batch)"]["mean"]))
    print(f"\n  {'phase':<28} {'mean':>12}      {'(distribution)':<38}   %step")
    print(f"  {'-'*28} {'-'*12}      {'-'*38}   -----")
    for name, ms in results.items():
        print(fmt_row(name, ms, total_step_ms))

    # ----------------------------------------------------- memory + throughput
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct = 100.0 * peak_gb / total_gb
        print(f"\n[VRAM]  peak allocated = {peak_gb:5.2f} GB / {total_gb:5.1f} GB "
              f"({pct:4.1f}%)")

    step_ms = results["fwd + bwd + optimizer.step"]["mean"]
    tokens_per_step = Config.BATCH_SIZE * (Config.SEQ_LEN - 1)
    tok_per_sec = tokens_per_step / (step_ms / 1000.0) if step_ms > 0 else 0
    print(f"[Throughput]  {tok_per_sec/1e6:6.2f} M tokens/sec  "
          f"({tok_per_sec:,.0f} tok/s) at step_ms={step_ms:.1f}")

    # ----------------------------------------------------- optional profiler
    if args.profile and torch.cuda.is_available():
        print("\n[Chrome trace] capturing 3 steps to ./diag_trace.json ...")
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True) as prof:
            for _ in range(3):
                phase_fwd_bwd()
        prof.export_chrome_trace("diag_trace.json")
        print("  wrote diag_trace.json (open in chrome://tracing or Perfetto)")

    # ----------------------------------------------------- verdict / advice
    print("\n=== Verdict ===")
    step = results["fwd + bwd + optimizer.step"]["mean"]
    fwd = results["fwd (teacher-force, no bwd)"]["mean"]
    dl = results["dataloader wait (next batch)"]["mean"]
    ar = results.get(f"AR rollout aux ({Config.AR_ROLLOUT_STEPS} steps)",
                     {"mean": 0.0})["mean"]
    trace = results["jit.trace + move-to-CPU stall"]["mean"]

    lines = []
    if dl > 0.25 * step:
        lines.append(f"  * DataLoader is the bottleneck ({dl:.1f} ms vs step {step:.1f} ms). "
                     f"Increase NUM_WORKERS, PREFETCH_FACTOR, or preload data to GPU.")
    else:
        lines.append(f"  * DataLoader is fine ({dl:.1f} ms vs step {step:.1f} ms). "
                     f"Adding more workers will NOT help.")

    if ar > 0.5 * step:
        lines.append(f"  * AR-rollout aux loss dominates ({ar:.1f} ms/step). "
                     f"Raise AR_EVERY_N_STEPS or lower AR_ROLLOUT_STEPS.")
    elif ar > 0:
        lines.append(f"  * AR-rollout aux loss adds {ar:.1f} ms (~{100*ar/step:.0f}% of a step) "
                     f"when it runs.")

    if trace > 3 * step:
        lines.append(f"  * The 'robust checkpoint' jit.trace stall costs {trace:.0f} ms "
                     f"(vs step {step:.1f} ms). Keep SAVE_SCRIPTED_MODELS=False during training.")

    if 2 * fwd < step * 0.6:
        lines.append(f"  * Backward+optimizer costs {step - fwd:.1f} ms vs fwd {fwd:.1f} ms; "
                     f"backward is dominating -- try enabling gradient checkpointing OFF and "
                     f"increasing BATCH_SIZE to amortize it.")

    # Rough B300 dense-fp16 roofline: ~3.5 PFLOP/s. This is only a sanity check.
    if torch.cuda.is_available():
        # crude FLOPs: 6 * N * L * D^2 per token for dense transformer forward+bwd
        approx_flops_per_tok = 6 * n_params
        achieved = approx_flops_per_tok * tok_per_sec
        lines.append(f"  * Approx sustained compute: {achieved/1e12:.1f} TFLOP/s "
                     f"(B300 dense-fp16 roofline ~= 3500 TFLOP/s). "
                     f"MFU ~= {100*achieved/3.5e15:.1f}%.")

    if not lines:
        lines.append("  * Nothing obviously pathological. If it still feels slow, "
                     "profile with --profile and look at diag_trace.json.")
    print("\n".join(lines))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
