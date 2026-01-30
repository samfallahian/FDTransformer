#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify all models load correctly
Tests each model with a small batch before running full experiments
"""

import os
import sys
import torch
import importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

MODELS = [
    ('encoder.permutations.model_01_standard_vae', 'StandardVAE'),
    ('encoder.permutations.model_02_beta_vae', 'BetaVAE'),
    ('encoder.permutations.model_03_sparse_ae', 'SparseAE'),
    ('encoder.permutations.model_04_contractive_ae', 'ContractiveAE'),
    ('encoder.permutations.model_05_denoising_ae', 'DenoisingAE'),
    ('encoder.permutations.model_06_adversarial_ae', 'AdversarialAE'),
    ('encoder.permutations.model_07_vq_vae', 'VQVAE'),
    ('encoder.permutations.model_08_deep_ae', 'DeepAE'),
    ('encoder.permutations.model_09_residual_ae', 'ResidualAE'),
    ('encoder.permutations.model_10_mixture_ae', 'MixtureDensityAE'),
]


def test_model(module_name, class_name):
    """Test if a model can be instantiated and run forward pass"""
    try:
        # Import
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Instantiate
        model = model_class()

        # Create dummy input (batch_size=4, input_dim=375)
        dummy_input = torch.randn(4, 375)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        # Check output
        if len(output) == 2:
            recon_x, z = output
            extra = None
        elif len(output) == 3:
            recon_x, z, extra = output
        else:
            return False, f"Unexpected output length: {len(output)}"

        # Check shapes
        if recon_x.shape != (4, 375):
            return False, f"Wrong recon_x shape: {recon_x.shape}"
        if z.shape[0] != 4 or z.shape[1] != 47:
            return False, f"Wrong latent shape: {z.shape}"

        # Test loss function
        if extra is None:
            loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, dummy_input, z)
        else:
            loss, recon_loss, aux1, aux2 = model.loss_function(recon_x, dummy_input, z, extra)

        # Check loss is scalar
        if not torch.is_tensor(loss) or loss.dim() != 0:
            return False, f"Loss is not a scalar: {loss}"

        return True, "OK"

    except Exception as e:
        import traceback
        return False, f"{str(e)}\n{traceback.format_exc()}"


def main():
    print("="*80)
    print("Testing all models...")
    print("="*80)

    results = []
    for module_name, class_name in MODELS:
        print(f"\nTesting {class_name}...", end=" ")
        success, message = test_model(module_name, class_name)

        if success:
            print("✓ PASS")
        else:
            print("✗ FAIL")
            print(f"  Error: {message}")

        results.append((class_name, success, message))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for class_name, success, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {class_name:<25} {message if not success else 'OK'}")

    print("="*80)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All models passed! Ready to run experiments.")
        print("\nRun: python -m encoder.permutations.run_all_experiments")
    else:
        print(f"\n⚠️  {total - passed} model(s) failed. Fix errors before running experiments.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
