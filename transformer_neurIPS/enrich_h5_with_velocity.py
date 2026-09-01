#!/usr/bin/env python3
"""
Enrich transformer_neurIPS/data/{train,val}_80.h5 with pre-decoded centroid velocity.

WHAT THIS SCRIPT DOES
=====================
Reads the existing 80-frame training / validation HDF5 files (shape
``(N_sequences, SEQ_LEN=2080, INPUT_DIM=52)``; columns 0:47 = 47-dim GEN3
autoencoder latent, columns 47:52 = ``(x, y, z, t_index, param)``), decodes
every token's latent through the FROZEN GEN3 AttentionSE scripted autoencoder
at::

    encoder/autoencoderGEN3/saved_models_production/
        Model_GEN3_05_AttentionSE_absolute_best_scripted.pt

to a 375-value velocity field, extracts the CENTRAL triplet of the
125-triplet layout (triplet index 62, 0-based -> array slice ``[186:189]``,
giving one ``(vx, vy, vz)`` per token), and writes new companion files next
to the originals::

    train_80.h5  ->  train_80_enriched.h5
    val_80.h5    ->  val_80_enriched.h5

The enriched file preserves the original ``'data'`` dataset verbatim and
adds a new ``'centroid_velocity'`` dataset of shape
``(N_sequences, SEQ_LEN, 3)``. Both datasets are written with gzip
compression when the source is uncompressed (matches the user's ask: "if we
don't have compression enabled, then do that please").

WHY NOT DECODE INSIDE THE TRAINER?
==================================
The training loss (see the DECODED-CENTROID TRAINING LOSS section in
``train_production_transformer_deep_dive.py``) is planned to score in
velocity space at the central triplet. Pre-decoding the TARGET side once
eliminates one decoder forward per training step -- ~half the added compute
cost of switching to velocity-space error. The trainer still needs the
scripted decoder file on the remote box to decode LIVE model predictions
each forward; that file must be present at the path above regardless.

The ``'data'`` dataset is copied verbatim (not stripped to the latent
columns) so this file remains a drop-in replacement for the original: any
consumer that reads ``f['data']`` continues to work; consumers that want the
new velocity read ``f['centroid_velocity']``.

CENTROID INDEX DERIVATION
=========================
The GEN3 AttentionSE autoencoder decodes a 47-dim latent into 375 = 125 * 3
scalars, laid out as
``[vx_0, vy_0, vz_0, vx_1, vy_1, vz_1, ..., vx_124, vy_124, vz_124]``. Those
125 triplets are the vertices of a 5 x 5 x 5 cube of spatial sample points
around one (t, x) location. The user asked for "only the centroid" of that
cube, which is the middle vertex -- triplet index 62 (0-based, since
``125 // 2 == 62``). Slice math: ``62 * 3 = 186``, so
``vel_375[186:189] == (vx_62, vy_62, vz_62)``. The constants
``CENTROID_TRIPLET_IDX`` and ``CENTROID_SLICE`` at the top of this file are
the single source of truth for that arithmetic; every downstream consumer
should read them from here (or from the future trainer copy) rather than
hard-code 186 / 189.

PARALLELISM
===========
The heavy lifting is ``dec.decode(z)``, a stack of ``nn.Linear`` +
``LayerNorm`` + attention ops on a ``(N, 47)`` tensor. PyTorch already
parallelises those internally (MKL / OpenMP on CPU, CUDA / MPS on
accelerators), so adding Python-level threads on top of the compute path
does not help and can hurt due to GIL contention. What DOES help is:

  1. Batching the decode call as large as memory allows so kernel-launch /
     Python overhead is amortised across many tokens.
  2. Overlapping H5 reads / writes with decode via a small
     ``ThreadPoolExecutor`` (``--workers``, default 4) -- ``h5py`` releases
     the GIL around chunk I/O, so the reader / writer threads run in
     parallel with the main decode thread. This is where "many threads"
     buys you time; the decode itself stays single-threaded from Python's
     point of view.

Set ``--device auto`` (default) to pick cuda > mps > cpu.

USAGE
=====
::

    # both files, defaults (device auto-detected, gzip on, batch 4096 tokens)
    python enrich_h5_with_velocity.py

    # only training file
    python enrich_h5_with_velocity.py --train-only

    # explicit knobs
    python enrich_h5_with_velocity.py \\
        --batch 8192 --device cpu --workers 8 --compression gzip

    # rebuild even if _enriched.h5 already exists
    python enrich_h5_with_velocity.py --overwrite
"""

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import h5py
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Constants -- single source of truth for the centroid geometry
# --------------------------------------------------------------------------- #
LATENT_DIM = 47                       # matches Config.LATENT_DIM in trainer
DECODED_DIM = 375                     # decoder output width
N_TRIPLETS = 125                      # 5 x 5 x 5 cube of spatial points
CENTROID_TRIPLET_IDX = 62             # middle of 125, 0-based (125 // 2 == 62)
CENTROID_SLICE = slice(
    CENTROID_TRIPLET_IDX * 3,
    CENTROID_TRIPLET_IDX * 3 + 3,
)                                     # slice(186, 189) -> (vx, vy, vz)
V_LABELS = ("vx", "vy", "vz")

# On-disk default location of the frozen scripted decoder. Overridable via
# ``--decoder`` or the ``PFD_DECODER_PATH`` env var (matches the PFD_*
# convention already in the trainer).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_DECODER_PATH = os.path.join(
    _REPO_ROOT, "encoder", "autoencoderGEN3", "saved_models_production",
    "Model_GEN3_05_AttentionSE_absolute_best_scripted.pt",
)
DEFAULT_TRAIN_H5 = os.path.join(_THIS_DIR, "data", "train_80.h5")
DEFAULT_VAL_H5 = os.path.join(_THIS_DIR, "data", "val_80.h5")


# --------------------------------------------------------------------------- #
# Console colour (light, self-contained; no dependency on the trainer)
# --------------------------------------------------------------------------- #
_COLOR_ON = (
    os.environ.get("PFD_NO_COLOR") is None
    and os.environ.get("NO_COLOR") is None
    and sys.stdout.isatty()
)
_ANSI = {"reset": "\033[0m", "bold": "\033[1m",
         "red": "\033[91m", "green": "\033[92m", "yellow": "\033[93m",
         "cyan": "\033[96m", "magenta": "\033[95m", "blue": "\033[94m"}
_RAINBOW = ["red", "yellow", "green", "cyan", "blue", "magenta"]


def _c(text, color):
    return f"{_ANSI[color]}{text}{_ANSI['reset']}" if _COLOR_ON else text


def _rainbow(text):
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


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Decoder loading
# --------------------------------------------------------------------------- #
def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_decoder(path, device):
    """Load the scripted GEN3 AttentionSE decoder, freeze it, expose a
    ``decode(z_47) -> vel_375`` callable that runs on ``device``.

    The scripted archive contains the full autoencoder (encode + decode +
    forward). We expose the decode path only.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"[enrich] scripted decoder not found: {path}\n"
            f"Pass --decoder PATH or set PFD_DECODER_PATH."
        )
    log(_rainbow(f"[start-from:decoder] {os.path.abspath(path)}"))
    mod = torch.jit.load(path, map_location=device).eval()
    # Freeze -- these weights must not receive gradients when the trainer
    # eventually differentiates through them.
    for p in mod.parameters():
        p.requires_grad_(False)

    def decode_fn(z: torch.Tensor) -> torch.Tensor:
        """(N, 47) -> (N, 375). No autograd needed here (enrichment is one-shot)."""
        # ScriptModule with the BaseAE contract exposes `decode`. Fall back to
        # `forward` only if scripting flattened the class to forward-only.
        try:
            return mod.decode(z)
        except (AttributeError, RuntimeError):
            # Some torch.jit archives don't surface `decode` as a
            # top-level ScriptModule method; unwrap manually.
            return mod.forward(z)  # last-ditch, expected to raise below if wrong shape

    return decode_fn, mod


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #
# Default gzip level for the enriched output. 6 is HDF5/zlib's sweet spot
# for float32 scientific data: ~15-25% smaller than level 4 at roughly
# 2x the CPU cost, still ~3x faster than level 9 which buys almost
# nothing more. Overridable via --compression-opts.
_DEFAULT_GZIP_LEVEL = 6


def _dataset_kwargs(src_ds: h5py.Dataset, compression: Optional[str],
                    compression_opts: Optional[int],
                    shuffle: bool = True):
    """Choose write kwargs for the enriched copy of a source dataset.

    Policy:
      * If the source has no compression, force gzip at
        ``_DEFAULT_GZIP_LEVEL`` (or ``compression_opts`` when explicit).
      * If the user forced a codec via ``--compression``, use it.
      * Otherwise preserve the source's compression settings verbatim.

    Additionally enables the HDF5 byte-shuffle filter (``shuffle=True``)
    whenever the output codec is gzip or lzf. Shuffle reorders bytes
    within each chunk so that same-position bytes across float32 values
    are contiguous; on smooth scientific data (like decoded velocity
    fields) this typically yields a 1.5-3x improvement in gzip ratio at
    a tiny CPU cost -- it's the single biggest 'free' compression win
    HDF5 offers for float arrays. Pass ``shuffle=False`` to disable.
    """
    def _with_shuffle(kw):
        # `shuffle` is only meaningful for byte-level compressors.
        if shuffle and kw.get("compression") in ("gzip", "lzf"):
            kw["shuffle"] = True
        return kw

    if src_ds.compression is None and compression is None:
        return _with_shuffle({
            "compression": "gzip",
            "compression_opts": compression_opts
                if compression_opts is not None else _DEFAULT_GZIP_LEVEL,
            "chunks": src_ds.chunks or True,
        })
    if compression is not None:
        kw = {"compression": compression, "chunks": src_ds.chunks or True}
        if compression == "gzip":
            kw["compression_opts"] = compression_opts \
                if compression_opts is not None else _DEFAULT_GZIP_LEVEL
        return _with_shuffle(kw)
    # Source already compressed -- keep it, but layer shuffle on top if it
    # wasn't already enabled (shuffle is orthogonal to codec choice).
    kw = {"compression": src_ds.compression,
          "chunks": src_ds.chunks or True}
    if src_ds.compression_opts is not None:
        kw["compression_opts"] = src_ds.compression_opts
    return _with_shuffle(kw)


def _copy_extra_datasets(src: h5py.File, dst: h5py.File,
                         compression: Optional[str],
                         compression_opts: Optional[int],
                         shuffle: bool = True):
    """Copy every top-level dataset except 'data' verbatim (matching or
    upgrading compression + shuffle). 'data' is streamed separately
    (below) so we can share the same chunk loop with the decode pass.
    """
    for name in src.keys():
        if name == "data":
            continue
        obj = src[name]
        if isinstance(obj, h5py.Dataset):
            kw = _dataset_kwargs(obj, compression, compression_opts, shuffle)
            dst.create_dataset(name, data=obj[...], **kw)
        # Groups -- copy recursively via h5py's built-in helper.
        else:
            src.copy(name, dst)
    # File-level attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def enrich_one_file(src_path, dst_path, decode_fn, device,
                    batch_tokens=4096, compression: Optional[str] = None,
                    compression_opts: Optional[int] = None,
                    workers=8, overwrite=False, shuffle=True,
                    prefetch=4):
    """Read one H5, decode centroids in batched chunks, write enriched twin."""
    if not os.path.exists(src_path):
        raise SystemExit(f"[enrich] source file not found: {src_path}")
    if os.path.exists(dst_path) and not overwrite:
        raise SystemExit(
            f"[enrich] refusing to overwrite existing {dst_path}. "
            "Pass --overwrite to force.")

    t0 = time.time()
    log(_c(f"[enrich] {src_path}", "cyan"))
    log(_c(f"      -> {dst_path}", "cyan"))

    tmp_path = dst_path + ".tmp"
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        data = src["data"]
        # The training H5 has been observed in two on-disk layouts:
        #
        #   (a) 3-D pre-flattened:   (N_sequences, SEQ_LEN, INPUT_DIM)
        #                            e.g. (N, 2080, 52)
        #   (b) 4-D frame-shaped:    (N_sequences, NUM_TIME, NUM_X, INPUT_DIM)
        #                            e.g. (N, 80, 26, 52)
        #
        # Both are semantically identical -- (b) is just (a) with the token
        # axis un-flattened back into (t, x). The trainer consumes (a); the
        # ``prepare_data.py`` writer emits (b) in some codepaths. We accept
        # either, preserve the source shape verbatim in the enriched file
        # (so downstream consumers that rely on the exact layout keep
        # working), and only flatten INTERNALLY for the decode pass.
        raw_shape = data.shape
        if data.ndim == 3:
            N, L, W = raw_shape
            vel_shape = (N, L, 3)
            vel_chunks = (1, L, 3)
        elif data.ndim == 4:
            N, NT, NX, W = raw_shape
            L = NT * NX
            vel_shape = (N, NT, NX, 3)
            vel_chunks = (1, NT, NX, 3)
        else:
            raise SystemExit(
                f"[enrich] {src_path}: 'data' has ndim={data.ndim}; expected "
                f"3 (N, L, W) or 4 (N, NT, NX, W). shape={raw_shape}")
        if W < LATENT_DIM:
            raise SystemExit(
                f"[enrich] {src_path}: 'data' has width {W} < LATENT_DIM={LATENT_DIM}; "
                "not a recognisable transformer_neurIPS dataset.")
        log(f"[enrich] source 'data' shape={raw_shape} dtype={data.dtype} "
            f"compression={data.compression!r} chunks={data.chunks}")
        log(f"[enrich] interpreted as N={N} sequences x L={L} tokens x W={W} features "
            f"(ndim={data.ndim}); velocity dataset will be {vel_shape}")

        # 1) Copy 'data' verbatim. h5py's group copy respects compression but
        #    doesn't let us upgrade None -> gzip on the fly, so we do it manually.
        data_kw = _dataset_kwargs(data, compression, compression_opts, shuffle)
        dst_data = dst.create_dataset(
            "data", shape=raw_shape, dtype=data.dtype, **data_kw)
        # 2) Allocate the new centroid_velocity dataset, matching the source's
        #    (N, ..., 3) layout, float32. Mirror the 'data' compression
        #    settings so both datasets share the same codec / level /
        #    shuffle filter (avoids one being much bigger than the other
        #    for no reason).
        vel_kw = {"compression": data_kw.get("compression", "gzip"),
                  "chunks": vel_chunks}
        if "compression_opts" in data_kw:
            vel_kw["compression_opts"] = data_kw["compression_opts"]
        elif vel_kw["compression"] == "gzip":
            vel_kw["compression_opts"] = _DEFAULT_GZIP_LEVEL
        if shuffle and vel_kw["compression"] in ("gzip", "lzf"):
            vel_kw["shuffle"] = True
        dst_vel = dst.create_dataset(
            "centroid_velocity", shape=vel_shape, dtype="float32", **vel_kw)
        log(f"[enrich] output codec: 'data' -> {data_kw.get('compression')!r} "
            f"(opts={data_kw.get('compression_opts')!r}, "
            f"shuffle={data_kw.get('shuffle', False)}); "
            f"'centroid_velocity' -> {vel_kw['compression']!r} "
            f"(opts={vel_kw.get('compression_opts')!r}, "
            f"shuffle={vel_kw.get('shuffle', False)})")
        dst_vel.attrs["centroid_triplet_idx"] = CENTROID_TRIPLET_IDX
        dst_vel.attrs["slice_start"] = CENTROID_SLICE.start
        dst_vel.attrs["slice_stop"] = CENTROID_SLICE.stop
        dst_vel.attrs["labels"] = np.array(V_LABELS, dtype="S")
        dst_vel.attrs["decoder"] = os.path.basename(DEFAULT_DECODER_PATH)
        dst_vel.attrs["description"] = (
            "Per-token decoded velocity at the central triplet of the "
            "125-triplet GEN3 AttentionSE decoder output, index 62 of 125 "
            "(slice 186:189). Order along last axis: (vx, vy, vz).")

        # 3) Copy any other top-level datasets / attrs verbatim.
        _copy_extra_datasets(src, dst, compression, compression_opts, shuffle)

        # 4) Streamed decode: pick a chunk of sequences whose flattened token
        #    count is close to `batch_tokens`. For SEQ_LEN=2080, batch_tokens
        #    4096 means ~2 sequences per chunk -- fine on CPU / MPS. On CUDA
        #    users can push --batch to 65536+.
        seqs_per_chunk = max(1, batch_tokens // L)
        n_chunks = (N + seqs_per_chunk - 1) // seqs_per_chunk

        # Deep I/O pipeline for high-IO Macs.
        #
        # h5py releases the GIL around chunk I/O and decode-on-device also
        # releases it, so a small ThreadPool overlaps reads / writes with
        # compute. Two knobs matter:
        #
        #   workers  = concurrency of the pool (default 8; enough to keep
        #              gzip + decompress + shuffle-decode all in flight
        #              on a Mac SSD that sustains multi-GB/s reads).
        #   prefetch = how many READ chunks are allowed to sit ahead of
        #              the decode. With shuffle+gzip decompression added
        #              to the source-read path, one-chunk pre-fetch is
        #              often not enough to hide latency; queuing 4
        #              chunks ahead lets the pool run several
        #              decompressors in parallel while the accelerator
        #              is busy on the current chunk. Bounded so peak
        #              memory is `prefetch * seqs_per_chunk` sequences.
        prefetch = max(1, int(prefetch))
        io_pool = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                     thread_name_prefix="h5-io")

        def read_chunk(i):
            start = i * seqs_per_chunk
            end = min(start + seqs_per_chunk, N)
            # Slice along axis 0 only; keep all trailing axes untouched so
            # both (N, L, W) and (N, NT, NX, W) sources work with the same
            # indexer. h5py returns a numpy array of shape (b, ...trailing).
            return start, end, data[start:end]

        def write_chunk(start, end, block, vel_block):
            # Both `block` and `vel_block` preserve the source's trailing
            # axes so the assignment matches the destination dataset shape.
            dst_data[start:end] = block
            dst_vel[start:end] = vel_block

        log(_c(f"[enrich] decode: {N} sequences x {L} tokens "
               f"({seqs_per_chunk} seqs/chunk, {n_chunks} chunks, "
               f"device={device}, workers={workers}, prefetch={prefetch})",
               "cyan"))

        # Seed the pipeline with `prefetch` reads so the decode never
        # has to wait on disk on a warm system.
        read_queue = deque()
        for j in range(min(prefetch, n_chunks)):
            read_queue.append(io_pool.submit(read_chunk, j))
        next_to_submit = len(read_queue)
        pending_writes = deque()                                    # bound to `prefetch`
        report_every = max(1, n_chunks // 20)                       # ~20 lines total

        for i in range(n_chunks):
            start, end, block = read_queue.popleft().result()
            # Keep the pipeline full: submit the next unqueued read.
            if next_to_submit < n_chunks:
                read_queue.append(io_pool.submit(read_chunk, next_to_submit))
                next_to_submit += 1

            # `block` has shape (b, ...trailing, W). Slice the latent columns
            # along the LAST axis regardless of whether the source is 3-D or
            # 4-D, so this works for both (b, L, W) and (b, NT, NX, W).
            lat_np = block[..., :LATENT_DIM]
            lat = torch.from_numpy(lat_np).to(
                device, non_blocking=True).float()                  # (b, ..., 47)
            lead_shape = lat.shape[:-1]                             # (b,) or (b, NT, NX)
            vel375 = decode_fn(lat.reshape(-1, LATENT_DIM))         # (M, 375)
            centroid = vel375[:, CENTROID_SLICE].reshape(
                *lead_shape, 3)                                     # (b, ..., 3)
            vel_np = centroid.detach().to("cpu").numpy().astype(np.float32)

            # Bounded write-ahead: allow up to `prefetch` write futures in
            # flight so gzip-encode doesn't stall the decode, but drain
            # the oldest one first once we reach the cap.
            if len(pending_writes) >= prefetch:
                pending_writes.popleft().result()
            pending_writes.append(
                io_pool.submit(write_chunk, start, end, block, vel_np))

            if (i % report_every == 0) or (i == n_chunks - 1):
                dt = time.time() - t0
                pct = 100.0 * (i + 1) / n_chunks
                rate = (end) / max(1e-6, dt)
                eta = (N - end) / max(1e-6, rate)
                log(f"[enrich]   chunk {i + 1}/{n_chunks}  seqs {end}/{N}  "
                    f"({pct:.1f}%)  {rate:.1f} seq/s  eta={eta:.1f}s")

        # Drain remaining writes.
        while pending_writes:
            pending_writes.popleft().result()
        io_pool.shutdown(wait=True)

    os.replace(tmp_path, dst_path)
    dt = time.time() - t0
    size_mb = os.path.getsize(dst_path) / 1e6
    log(_rainbow(
        f"[write:enriched] {os.path.abspath(dst_path)} ({size_mb:.1f} MB) "
        f"in {dt:.1f}s"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-h5", default=DEFAULT_TRAIN_H5)
    p.add_argument("--val-h5", default=DEFAULT_VAL_H5)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--decoder",
                   default=os.environ.get("PFD_DECODER_PATH",
                                          DEFAULT_DECODER_PATH),
                   help="Path to the scripted GEN3 AttentionSE decoder "
                        "(default: env PFD_DECODER_PATH, then "
                        "encoder/autoencoderGEN3/saved_models_production/"
                        "Model_GEN3_05_AttentionSE_absolute_best_scripted.pt)")
    p.add_argument("--device", default="auto",
                   help="'auto' | 'cuda' | 'mps' | 'cpu'")
    p.add_argument("--batch", type=int, default=4096,
                   metavar="TOKENS",
                   help="Approximate token count per decode chunk (default 4096; "
                        "the actual sequence-batch size is TOKENS // SEQ_LEN, "
                        "rounded up to at least 1).")
    p.add_argument("--workers", type=int, default=8,
                   help="ThreadPool size for overlapping H5 I/O with decode "
                        "(default 8; set to 1 to disable overlap). On a "
                        "high-IO Mac SSD 8 keeps gzip + shuffle-decode + "
                        "decompression all in flight while MPS runs the "
                        "decode; drop to 2 on constrained VMs.")
    p.add_argument("--prefetch", type=int, default=4,
                   help="How many read chunks are queued ahead of the "
                        "decode (default 4). Bounds peak memory to "
                        "prefetch*chunk sequences. Set to 1 to serialise.")
    p.add_argument("--compression", default=None,
                   choices=[None, "gzip", "lzf", "none"],
                   help="Force a specific compression codec on the output. "
                        "Default: preserve source; upgrade None -> gzip.")
    p.add_argument("--compression-opts", type=int, default=None,
                   help=f"Compression level (only meaningful for gzip: 1-9; "
                        f"default {_DEFAULT_GZIP_LEVEL}). Higher = smaller "
                        f"file but slower write.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable the HDF5 byte-shuffle filter. Shuffle is "
                        "ON by default because it typically halves the "
                        "on-disk size of gzipped float32 arrays at ~zero "
                        "CPU cost; only turn it off if a downstream "
                        "consumer can't handle shuffled datasets.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing *_enriched.h5 file.")
    return p


def enriched_path_for(src):
    root, ext = os.path.splitext(src)
    return f"{root}_enriched{ext}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.train_only and args.val_only:
        raise SystemExit("--train-only and --val-only are mutually exclusive")
    compression = None if args.compression in (None, "none") else args.compression

    device = resolve_device(args.device)
    log(_c(f"[enrich] device={device}  torch={torch.__version__}", "cyan"))

    decode_fn, _mod = load_decoder(args.decoder, device)

    # Sanity: decode a single all-zeros latent so a bad path / shape blows up
    # BEFORE we spend N sequences reading the H5.
    with torch.no_grad():
        probe = decode_fn(torch.zeros(1, LATENT_DIM, device=device))
    if probe.shape[-1] != DECODED_DIM:
        raise SystemExit(
            f"[enrich] decoder output width is {probe.shape[-1]}, expected "
            f"{DECODED_DIM}. Is the scripted archive the GEN3 AttentionSE "
            f"autoencoder?")
    log(f"[enrich] decoder probe: input(1, {LATENT_DIM}) -> output{tuple(probe.shape)}; "
        f"centroid slice = [{CENTROID_SLICE.start}:{CENTROID_SLICE.stop}] "
        f"(triplet idx {CENTROID_TRIPLET_IDX} of {N_TRIPLETS})")

    plan = []
    if not args.val_only:
        plan.append((args.train_h5, enriched_path_for(args.train_h5)))
    if not args.train_only:
        plan.append((args.val_h5, enriched_path_for(args.val_h5)))

    for src, dst in plan:
        enrich_one_file(src, dst, decode_fn, device,
                        batch_tokens=args.batch,
                        compression=compression,
                        compression_opts=args.compression_opts,
                        workers=args.workers,
                        overwrite=args.overwrite,
                        shuffle=not args.no_shuffle,
                        prefetch=args.prefetch)

    log(_c("[enrich] done.", "green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
