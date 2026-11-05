"""One-off: re-export train_80.h5/val_80.h5 with compression=None.

See OVERVIEW.md's h5-load-bottleneck note: these files were writen with
gzip at a ~13% size reduction (poor ratio for this float32 physics data)
while still paying the full single-threaded zlib decompression cost on
every load. Storing raw removes that cost entirely at a modest disk-size
increase.

Streams row-by-row (matches prepare_data.py's own write chunk size of
(1, NUM_TIME, NUM_X, 52)) so memory stays bounded regardless of file
size. Writes to a .tmp sibling and only replaces the original after the
row count and a full-array byte-for-byte comparison both pass.
"""
import os
import sys
import h5py
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def convert(name, batch_rows=512):
    src_path = os.path.join(DATA_DIR, name)
    tmp_path = src_path + ".uncompressed.tmp"

    with h5py.File(src_path, "r") as f_in:
        shape = f_in["data"].shape
        dtype = f_in["data"].dtype
        attrs = dict(f_in.attrs)

        print(f"{name}: {shape} {dtype}, compression={f_in['data'].compression} -> None")

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        with h5py.File(tmp_path, "w") as f_out:
            dset = f_out.create_dataset(
                "data", shape=shape, dtype=dtype,
                chunks=(1,) + shape[1:], compression=None,
            )
            n = shape[0]
            for start in range(0, n, batch_rows):
                end = min(start + batch_rows, n)
                dset[start:end] = f_in["data"][start:end]
                if start % (batch_rows * 20) == 0:
                    print(f"  {name}: {end}/{n} rows")
            for k, v in attrs.items():
                f_out.attrs[k] = v

    # Verify before replacing: row count + full byte-for-byte equality,
    # read back in the same batched fashion (never load either full
    # array into memory at once).
    with h5py.File(src_path, "r") as f_src, h5py.File(tmp_path, "r") as f_new:
        assert f_src["data"].shape == f_new["data"].shape, "shape mismatch"
        n = f_src["data"].shape[0]
        for start in range(0, n, batch_rows):
            end = min(start + batch_rows, n)
            a = f_src["data"][start:end]
            b = f_new["data"][start:end]
            if not np.array_equal(a, b):
                raise RuntimeError(f"{name}: data mismatch in rows [{start}:{end})")
        assert dict(f_src.attrs.items()).keys() == dict(f_new.attrs.items()).keys(), "attr key mismatch"

    old_size = os.path.getsize(src_path)
    new_size = os.path.getsize(tmp_path)
    os.replace(tmp_path, src_path)
    print(f"{name}: verified + replaced. {old_size/1e9:.2f} GB -> {new_size/1e9:.2f} GB")


if __name__ == "__main__":
    names = sys.argv[1:] or ["val_80.h5", "train_80.h5"]
    for name in names:
        convert(name)
