#!/usr/bin/env python3
"""
Produce smaller, randomly-sampled companions of train_80.h5 / val_80.h5, for
uploading to a bandwidth-throttled rented GPU box (e.g. a fresh RunPod
instance) when you only need enough data for a shallow sweep, not the full
dataset.

WHY THIS EXISTS
===============
train_80.h5 (~7.9 GB) and val_80.h5 (~3.4 GB) are the full datasets. The
Round-2 shallow sweep (run_sweep_mac.sh / run_sweep_h200.sh) already only
*uses* a fraction of train data via --subset-ratio, but that subsetting
happens AFTER the full file is loaded -- you still pay the full transfer
cost even for a 400-step throwaway experiment. This script makes the
smaller file up front, so only the smaller file needs to cross a throttled
link.

WHAT IT DOES
============
For each of train_80.h5 and val_80.h5: picks `--fraction` (default 0.30) of
the sequences uniformly at random, WITHOUT replacement, and writes them to a
new file with the same shape/dtype/compression and the same file-level
attrs as the source, plus provenance attrs recording exactly how the sample
was drawn (fraction, seed, source file, counts) so a sample file is never
mistaken for the real thing.

IMPORTANT -- FILENAME ON THE REMOTE BOX
========================================
Config.TRAIN_H5 / Config.VAL_H5 in train_production_transformer_deep_dive.py
are HARD-CODED to "data/train_80.h5" / "data/val_80.h5" and are in
PINNED_CONFIG_FIELDS (no --set override possible -- by design, see
OVERVIEW.md). This script deliberately does NOT write directly to those
filenames (so it never clobbers your real local files) -- it writes
train_80_sample<PCT>.h5 / val_80_sample<PCT>.h5 instead. On the remote box,
you must rename (or upload directly as) data/train_80.h5 / data/val_80.h5
for the trainer to pick them up:

    scp transformer_neurIPS/data/train_80_sample30.h5 \\
        user@box:.../transformer_neurIPS/data/train_80.h5
    scp transformer_neurIPS/data/val_80_sample30.h5 \\
        user@box:.../transformer_neurIPS/data/val_80.h5

Also: run_sweep_h200.sh passes --subset-ratio 0.3 by default (see its
SUBSET_RATIO env var). If you've ALREADY pre-sampled to 30% with this
script, set SUBSET_RATIO=1.0 when running run_sweep_h200.sh against the
sample files, or you'll compound to 0.3*0.3=9% of the original data
without meaning to. (val is unaffected -- the trainer always loads val at
subset_ratio=1.0 relative to whatever file VAL_H5 points at, so a
pre-sampled val file is automatically "the whole validation set" as far as
the trainer is concerned -- no flag needed there.)

USAGE
=====
    python transformer_neurIPS/make_sweep_sample_data.py
    python transformer_neurIPS/make_sweep_sample_data.py --fraction 0.3 --seed 1337
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def sample_one_file(src_path, dst_path, fraction, seed, write_chunk=256, log=print):
    if not os.path.exists(src_path):
        raise SystemExit(f"source file not found: {src_path}")
    if os.path.exists(dst_path):
        raise SystemExit(
            f"refusing to overwrite existing {dst_path}. Remove it first "
            f"if you want to regenerate it.")

    t0 = time.time()
    with h5py.File(src_path, "r") as src:
        src_ds = src["data"]
        n_total, num_time, num_x, feat = src_ds.shape
        n_sample = max(1, round(n_total * fraction))

        rng = np.random.default_rng(seed)
        # Sample without replacement, then sort for efficient sequential
        # reads against the source's chunk-per-sequence layout (see
        # prepare_data.py: chunks=(1, NUM_TIME, NUM_X, 52)) -- reading in
        # ascending index order avoids thrashing h5py's chunk cache with
        # random-access reads. Output order doesn't matter for training
        # (TransformerDataset holds the whole split as one resident tensor;
        # nothing depends on row order), so leaving it sorted is fine.
        indices = np.sort(rng.choice(n_total, size=n_sample, replace=False))

        log(f"[sample] {os.path.basename(src_path)}: {n_total:,} sequences "
            f"-> sampling {n_sample:,} ({fraction:.0%}), seed={seed}")

        tmp_path = dst_path + ".tmp"
        try:
            with h5py.File(tmp_path, "w") as dst:
                dst_ds = dst.create_dataset(
                    "data", shape=(n_sample, num_time, num_x, feat),
                    dtype=src_ds.dtype,
                    chunks=(1, num_time, num_x, feat),
                    compression="gzip")

                for start in range(0, n_sample, write_chunk):
                    end = min(start + write_chunk, n_sample)
                    idx_slice = indices[start:end]
                    # h5py requires a list (not ndarray) for this kind of
                    # fancy index on some versions; both work, but be explicit.
                    block = src_ds[list(idx_slice)]
                    dst_ds[start:end] = block
                    if start == 0 or end == n_sample or (start // write_chunk) % 10 == 0:
                        log(f"  [write-progress] {os.path.basename(dst_path)}: "
                            f"{end:,}/{n_sample:,}")

                # Carry over every source file-level attribute verbatim,
                # then add provenance attrs so this file can never be
                # mistaken for the real dataset.
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["sampled_from"] = os.path.abspath(src_path)
                dst.attrs["sampled_fraction"] = float(fraction)
                dst.attrs["sampled_seed"] = int(seed)
                dst.attrs["sampled_n"] = int(n_sample)
                dst.attrs["sampled_total_source_n"] = int(n_total)
                dst.attrs["sampled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

            os.replace(tmp_path, dst_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    src_size = os.path.getsize(src_path)
    dst_size = os.path.getsize(dst_path)
    log(f"[sample] wrote {dst_path}")
    log(f"  {src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB "
        f"({dst_size / src_size:.1%} of source), {time.time() - t0:.1f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-h5", default=os.path.join(_HERE, "data", "train_80.h5"))
    p.add_argument("--val-h5", default=os.path.join(_HERE, "data", "val_80.h5"))
    p.add_argument("--fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--val-only", action="store_true")
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit(f"--fraction must be in (0, 1], got {args.fraction}")

    pct = round(args.fraction * 100)

    def _sample_path(src):
        base, ext = os.path.splitext(src)
        return f"{base}_sample{pct}{ext}"

    if not args.val_only:
        sample_one_file(args.train_h5, _sample_path(args.train_h5),
                        args.fraction, args.seed)
    if not args.train_only:
        sample_one_file(args.val_h5, _sample_path(args.val_h5),
                        args.fraction, args.seed)

    print()
    print("Done. To use these on a remote box, upload them AS data/train_80.h5 / "
          "data/val_80.h5 (Config.TRAIN_H5/VAL_H5 are pinned to those exact "
          "names -- see this file's module docstring). If you also pass "
          "--subset-ratio to run_sweep_h200.sh, set it to 1.0 when running "
          "against an already-sampled train file, or the fractions compound.")


if __name__ == "__main__":
    main()
