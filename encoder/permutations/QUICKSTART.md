# Quick Start Guide

## Overview

This experiment compares 10 different autoencoder approaches on your fluid dynamics velocity data. All models compress 375-dimensional inputs to 47-dimensional latent representations.

## Prerequisites

- PyTorch installed
- Training data: `training_auto_encoder.pkl`
- Validation data: `validation_auto_encoder.pkl`
- Original WAE weights: `encoder/saved_models/WAE_Cached_012_H200_FINAL.pt`

## Step 1: Test All Models

First, verify all models load and run correctly:

```bash
cd /Users/kkreth/PycharmProjects/cgan
python -m encoder.permutations.quick_test
```

Expected output:
```
Testing all models...
Testing StandardVAE... ✓ PASS
Testing BetaVAE... ✓ PASS
...
🎉 All models passed! Ready to run experiments.
```

## Step 2: Configure Experiment

Edit `encoder/permutations/run_all_experiments.py`:

```python
NUMEPOCHS = 100      # Adjust epochs (100-500 recommended)
THREAD_COUNT = 2     # Number of simultaneous trainings (2-3 for local)
```

## Step 3: Run Experiments

```bash
cd /Users/kkreth/PycharmProjects/cgan
python -m encoder.permutations.run_all_experiments
```

This will:
- Train all 10 models (in parallel if THREAD_COUNT > 1)
- Track RMSE for each epoch
- Save results to `encoder/permutations/results/`

## Step 4: Review Results

Results are saved as:
- `experiment_results_YYYYMMDD_HHMMSS.json` (full data)
- `experiment_results_YYYYMMDD_HHMMSS.csv` (summary table)

Console output shows ranked results:

```
EXPERIMENT RESULTS SUMMARY
==================================================================================
Rank   Model                          Val RMSE     Train RMSE   Time (min)
----------------------------------------------------------------------------------
1      Model_03_Sparse_AE             0.012345     0.011234     12.34
2      Model_09_Residual_AE           0.012456     0.011345     15.67
...

🏆 BEST MODEL: Model_03_Sparse_AE
```

## Estimated Runtime

With THREAD_COUNT=2 and NUMEPOCHS=100:
- Total time: ~2-3 hours (depends on hardware)
- Per model: ~15-20 minutes

## Configuration Options

### Quick Test (10 epochs, fast debugging)
```python
NUMEPOCHS = 10
THREAD_COUNT = 2
```

### Standard Run (100 epochs, good comparison)
```python
NUMEPOCHS = 100
THREAD_COUNT = 2
```

### Thorough Run (500 epochs, publication-quality)
```python
NUMEPOCHS = 500
THREAD_COUNT = 2
```

### Sequential (single model at a time)
```python
NUMEPOCHS = 100
THREAD_COUNT = 1
```

## Interpreting Results

### Validation RMSE
- Lower is better
- Primary metric for model comparison
- Indicates generalization performance

### Training RMSE
- Lower indicates better fit to training data
- Large gap from validation RMSE suggests overfitting

### Time
- Consider RMSE/time trade-off for production
- Some models (Contractive, Adversarial) may be slower

## What to Look For

1. **Best overall:** Lowest validation RMSE
2. **Fast training:** Models that converge quickly (< 50 epochs)
3. **Stable training:** No divergence or NaN losses
4. **Small train/val gap:** Indicates good generalization

## Troubleshooting

### "Module not found" errors
```bash
cd /Users/kkreth/PycharmProjects/cgan
export PYTHONPATH="${PYTHONPATH}:/Users/kkreth/PycharmProjects/cgan"
```

### Out of memory
- Reduce BATCH_SIZE in run_all_experiments.py
- Reduce THREAD_COUNT to 1
- Use smaller NUMEPOCHS for testing

### NaN losses
- Some models may be sensitive to learning rate
- Check individual model training logs
- This is logged but won't stop other models

### Slow performance
- GPU not detected - check PyTorch CUDA/MPS setup
- Too many workers - THREAD_COUNT should be 2-3 for local
- Increase BATCH_SIZE if you have memory

## Files Created

```
encoder/permutations/
├── model_01_standard_vae.py          # Model architectures
├── model_02_beta_vae.py
├── ...
├── model_10_mixture_ae.py
├── train_permutation.py              # Generic training code
├── run_all_experiments.py            # Master experiment runner
├── quick_test.py                     # Validation script
├── results/                          # Results directory
│   ├── experiment_results_*.json
│   └── experiment_results_*.csv
├── README.md                         # Full documentation
├── APPROACHES_SUMMARY.md             # Theory and comparison
└── QUICKSTART.md                     # This file
```

## Next Steps After Experiments

1. **Identify top 3 models** by validation RMSE
2. **Visualize reconstructions** to check quality
3. **Examine latent space** structure and distribution
4. **Fine-tune best model** with hyperparameter search
5. **Compare with original WAE** baseline

## Example Workflow

```bash
# Navigate to project
cd /Users/kkreth/PycharmProjects/cgan

# Test models
python -m encoder.permutations.quick_test

# Run quick test (10 epochs)
# Edit run_all_experiments.py: NUMEPOCHS = 10
python -m encoder.permutations.run_all_experiments

# Review results, adjust if needed
cat encoder/permutations/results/experiment_results_*.csv

# Run full experiment (100+ epochs)
# Edit run_all_experiments.py: NUMEPOCHS = 100
python -m encoder.permutations.run_all_experiments

# Analyze results
python  # Start Python
>>> import json
>>> with open('encoder/permutations/results/experiment_results_20260129_120000.json') as f:
...     data = json.load(f)
>>> # Explore data['results']
```

## Support

See detailed documentation in:
- `README.md` - Full documentation
- `APPROACHES_SUMMARY.md` - Theory behind each approach

## Tips

- Start with NUMEPOCHS=10 for quick validation
- Use THREAD_COUNT=2 for balanced parallelism
- Monitor first few epochs to catch issues early
- Save results files for future reference
- Compare with original WAE baseline
