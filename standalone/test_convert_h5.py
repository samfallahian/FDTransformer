import h5py
import torch
import pytest

DATA_PATH = "/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.hdf"
NEW_FILE = "/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_corrected.hdf"

def test_record_counts_match():
    original_data = torch.load(DATA_PATH)
    original_count = len(original_data)

    with h5py.File(NEW_FILE, 'r') as f:
        corrected_data = f['my_data']
        corrected_count = len(corrected_data)

    assert original_count == corrected_count, f"Original count: {original_count}, Corrected count: {corrected_count}"
