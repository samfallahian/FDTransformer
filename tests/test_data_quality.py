"""Content-level sanity checks on the 80-frame training/validation files.

Reads the files the trainer actually uses (Config.TRAIN_H5 / Config.VAL_H5,
i.e. $PFD_DATA_DIR/{train,val}_80.h5, default <repo>/data/). Skips cleanly if
they are not present. Only a small sample of sequences is read, so this stays
fast even on the ~10 GB train file.

    PFD_DATA_DIR=/data/ptv-092026 uv run pytest tests/test_data_quality.py -s
"""
import os
import sys
import unittest

import h5py
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_production_transformer_deep_dive as T  # noqa: E402

SAMPLE_SEQS = 64


class _DataFileChecks:
    path = None

    def setUp(self):
        if not os.path.exists(self.path):
            self.skipTest(f"Data file {self.path} not found (set PFD_DATA_DIR).")

    def _sample(self):
        with h5py.File(self.path, "r") as f:
            ds = f["data"]
            n = min(SAMPLE_SEQS, ds.shape[0])
            return ds.shape, ds[:n]

    def test_shape_matches_trainer_config(self):
        shape, _ = self._sample()
        cfg = T.Config
        self.assertEqual(tuple(shape[1:]), (cfg.NUM_TIME, cfg.NUM_X, cfg.INPUT_DIM),
                         f"{self.path}: expected (N, {cfg.NUM_TIME}, {cfg.NUM_X}, "
                         f"{cfg.INPUT_DIM}), got {shape}")

    def test_dtypes_and_content(self):
        _, data = self._sample()
        lat = T.Config.LATENT_DIM
        # y and z (cols 48/49) are integer grid coordinates stored as float32.
        y, z = data[..., lat + 1], data[..., lat + 2]
        np.testing.assert_array_equal(y, y.astype(int), "Y coordinates should be integers")
        np.testing.assert_array_equal(z, z.astype(int), "Z coordinates should be integers")
        latents = data[..., :lat]
        for i in range(min(10, len(data))):
            self.assertFalse(np.all(latents[i] == 0), f"Sequence {i} contains only zero latents")
        zero_fraction = float(np.mean(latents == 0))
        print(f"\n{os.path.basename(self.path)}: zero fraction in latents = {zero_fraction:.4%}")
        self.assertLess(zero_fraction, 0.01, f"Way too many zeros in latents: {zero_fraction:.4%}")


class TestTrainFile(_DataFileChecks, unittest.TestCase):
    path = T.Config.TRAIN_H5


class TestValFile(_DataFileChecks, unittest.TestCase):
    path = T.Config.VAL_H5


if __name__ == "__main__":
    unittest.main()
