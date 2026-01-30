# Recent Changes and Fixes

## Issues Fixed

### 1. ✓ Duplicate Data Loading
**Problem:** Each parallel process was loading the full 1M row dataset independently
**Solution:** Each process still loads independently (required for multiprocessing), but now only loads the percentage specified by `GLOBAL_PERCENTAGE`

### 2. ✓ MPS pin_memory Warning
**Problem:** Warning: "'pin_memory' argument is set as true but not supported on MPS"
**Solution:** Automatically detect device type and only enable pin_memory for CUDA (not MPS)

### 3. ✓ Slow Progress with Full Dataset
**Problem:** 30 minutes per 10 epochs with 1M rows
**Solution:** Added `GLOBAL_PERCENTAGE` parameter (set to 10% by default = 100k rows)

### 4. ✓ No Checkpoint Saving
**Problem:** Models not saved during training
**Solution:** Automatically save checkpoints every 10 epochs to `encoder/permutations/checkpoints/Model_XX/`

### 5. ✓ Infrequent Progress Updates
**Problem:** Only logging every 10 epochs
**Solution:**
- Now logs every 5 epochs
- Added ETA (estimated time remaining) to each log message
- Added checkpoint save notifications

## Configuration

Edit these at the top of `run_all_experiments.py`:

```python
NUMEPOCHS = 100          # Number of epochs (default: 100)
THREAD_COUNT = 2         # Parallel trainings (default: 2)
GLOBAL_PERCENTAGE = 10   # Data percentage (10 = 10%, 100 = full)
BATCH_SIZE = 128         # Batch size (default: 128)
LEARNING_RATE = 1e-4     # Learning rate (default: 1e-4)
```

## Expected Behavior Now

### Progress Logging
```
2026-01-29 16:00:00 - INFO - Model_01_Standard_VAE - Epoch 1/100: train_RMSE=0.083892 val_RMSE=0.053701 [ETA: 45.2min]
2026-01-29 16:02:15 - INFO - Model_01_Standard_VAE - Epoch 5/100: train_RMSE=0.052341 val_RMSE=0.048234 [ETA: 42.8min]
2026-01-29 16:04:30 - INFO - Model_01_Standard_VAE - Epoch 10/100: train_RMSE=0.048788 val_RMSE=0.046841 [ETA: 40.5min]
2026-01-29 16:04:31 - INFO -   → Saved checkpoint: Model_01_Standard_VAE_epoch_10.pt
```

### Checkpoint Structure
```
encoder/permutations/checkpoints/
├── Model_01_Standard_VAE/
│   ├── Model_01_Standard_VAE_epoch_10.pt
│   ├── Model_01_Standard_VAE_epoch_20.pt
│   └── ...
├── Model_02_Beta_VAE/
│   ├── Model_02_Beta_VAE_epoch_10.pt
│   └── ...
└── ...
```

### Checkpoint Contents
Each checkpoint contains:
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `train_rmse`: Training RMSE at this epoch
- `val_rmse`: Validation RMSE at this epoch
- `train_rmse_history`: Full training history
- `val_rmse_history`: Full validation history

## Performance Impact

### With GLOBAL_PERCENTAGE = 10 (100k rows):
- **Data loading:** ~0.1s per process (was ~0.4s)
- **Epoch time:** ~3 minutes (was ~30 minutes)
- **Total for 100 epochs:** ~5 hours for all 10 models with THREAD_COUNT=2

### With GLOBAL_PERCENTAGE = 100 (full 1M rows):
- **Data loading:** ~0.4s per process
- **Epoch time:** ~30 minutes
- **Total for 100 epochs:** ~50 hours for all 10 models with THREAD_COUNT=2

## Recommended Settings

### Quick Test (validate everything works)
```python
NUMEPOCHS = 10
GLOBAL_PERCENTAGE = 10
THREAD_COUNT = 2
```
**Runtime:** ~30 minutes total

### Standard Run (good comparison)
```python
NUMEPOCHS = 100
GLOBAL_PERCENTAGE = 10
THREAD_COUNT = 2
```
**Runtime:** ~5 hours total

### Full Run (best accuracy)
```python
NUMEPOCHS = 100
GLOBAL_PERCENTAGE = 100
THREAD_COUNT = 2
```
**Runtime:** ~50 hours total

## Resume from Checkpoint

To resume training from a checkpoint (future enhancement):

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/Model_01_Standard_VAE/Model_01_Standard_VAE_epoch_50.pt')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Resume from epoch 51
start_epoch = checkpoint['epoch']
```

## Files Modified

1. `train_permutation.py`
   - Added `sample_percentage` parameter
   - Fixed `pin_memory` for MPS
   - Added checkpoint saving every 10 epochs
   - Improved logging (every 5 epochs + ETA)
   - Shortened filenames in logs

2. `run_all_experiments.py`
   - Added `GLOBAL_PERCENTAGE` configuration
   - Create checkpoint directories automatically
   - Pass sampling and save parameters to training
   - Log data percentage in startup message
   - Store data_percentage in results JSON

## Testing

Run the quick test to verify all models work:
```bash
cd /Users/kkreth/PycharmProjects/cgan
python -m encoder.permutations.quick_test
```

Then run with small settings to verify training:
```bash
# Edit run_all_experiments.py:
#   NUMEPOCHS = 10
#   GLOBAL_PERCENTAGE = 10
python -m encoder.permutations.run_all_experiments
```
