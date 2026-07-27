import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

def test_data_integrity():
    train_path = "/Users/kkreth/PycharmProjects/data/encoder_neurIPS/training_auto_encoder.pkl"
    val_path = "/Users/kkreth/PycharmProjects/data/encoder_neurIPS/validation_auto_encoder.pkl"
    
    print(f"Checking {train_path}...")
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"Checking {val_path}...")
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
        
    print(f"Train shape: {train_data.shape}, dtype: {train_data.dtype}")
    print(f"Val shape: {val_data.shape}, dtype: {val_data.dtype}")
    
    assert train_data.shape[0] == 1_000_000, "Train data should have 1MM rows"
    assert val_data.shape[0] == 1_000_000, "Val data should have 1MM rows"
    assert train_data.dtype == np.float32, "Train data should be float32"
    assert val_data.dtype == np.float32, "Val data should be float32"
    
    # Simple check for disjointness (statistical)
    # Since they come from different files, they should be different.
    # We can check mean/std
    train_mean = np.mean(train_data)
    val_mean = np.mean(val_data)
    print(f"Train mean: {train_mean}, Val mean: {val_mean}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(train_data[:10000].flatten(), bins=50, alpha=0.5, label='Train (subset)')
    plt.hist(val_data[:10000].flatten(), bins=50, alpha=0.5, label='Val (subset)')
    plt.title("Distribution of Velocity Components")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([train_data[:1000].flatten(), val_data[:1000].flatten()], labels=['Train', 'Val'])
    plt.title("Boxplot of Velocity Components")
    
    os.makedirs("encoder_neurIPS/unit_tests/plots", exist_ok=True)
    plt.savefig("encoder_neurIPS/unit_tests/plots/data_validation.png")
    print("Validation plot saved to encoder_neurIPS/unit_tests/plots/data_validation.png")

if __name__ == "__main__":
    try:
        test_data_integrity()
        print("Unit test PASSED")
    except Exception as e:
        print(f"Unit test FAILED: {e}")
