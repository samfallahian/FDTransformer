"""One-off: convert train_80.h5/val_80.h5's `data` dataset into a flat raw
binary file (`<name>.raw` + a `<name>.raw.json` shape/dtype sidecar).

WHY: train_80.h5/val_80.h5 are chunked ONE ROW PER CHUNK, so a full read
is 59,280 (train) / 25,410 (val) separate HDF5 chunk-index lookups + reads
rather than one continuous stream. Fine on a fast local disk with a warm
page cache; potentially very slow -- one I/O round-trip per row -- on a
cold cache or network-backed pod volume, which is the leading suspect for
the unexplained >1 minute startup pause reported on a real H100 pod run
where the .h5 files were already local (i.e. not a network-transfer cost).
A flat raw binary file has no chunk structure at all: reading it is a
single sequential stream, the fastest access pattern any storage layer
supports, and TransformerDataset (train_production_transformer_deep_dive.py)
automatically prefers a `.raw`/`.raw.json` sibling over the HDF5 path when
one exists -- see convert_h5_to_raw()'s and _read_raw_binary_to_tensor()'s
own docstrings there.

Verified byte-for-byte against the source .h5 (streamed, batch_rows at a
time, memory stays bounded) before the .raw file is left in place -- same
convention as decompress_h5.py.

Usage:
    python h5_to_raw.py                      # converts val_80.h5, train_80.h5
    python h5_to_raw.py train_80.h5          # just one
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_production_transformer_deep_dive import convert_h5_to_raw  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


if __name__ == "__main__":
    names = sys.argv[1:] or ["val_80.h5", "train_80.h5"]
    for name in names:
        convert_h5_to_raw(os.path.join(DATA_DIR, name))
