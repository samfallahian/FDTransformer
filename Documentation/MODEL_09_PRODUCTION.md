# Model_09_Residual_AE Production Training

## Winner! 🏆

Model_09_Residual_AE achieved the best results in the 10-model comparison:
- **Validation RMSE:** 0.006765 (best)
- **Training RMSE:** 0.006751
- **Training time:** 31.84 minutes (100 epochs, 10% data)

## What's Special About This Model

**Residual Architecture with Skip Connections:**
- Better gradient flow during training
- Prevents vanishing gradients
- Allows deeper networks to train effectively
- Each residual block has:
  - Two linear layers with LayerNorm
  - Skip connection (x + f(x))
  - ELU activation

**Architecture:**
```
Input (375) → ResBlock(250) → ResBlock(150) → ResBlock(100) → Latent (47)
              ↓                ↓               ↓
Latent (47) → ResBlock(100) → ResBlock(150) → ResBlock(250) → Output (375)
```

## Quick Start - Continue from Checkpoint

The model was trained for 100 epochs on 10% of data. Now train on full dataset:

```bash
cd /Users/kkreth/PycharmProjects/cgan

# Default: Continue from epoch 100, train to epoch 500, with wandb
python encoder/train_model_09_production.py

# Custom epochs and batch size
python encoder/train_model_09_production.py --epochs 1000 --batch_size 256

# Without wandb logging
python encoder/train_model_09_production.py --no_wandb

# Start from scratch (no checkpoint)
python encoder/train_model_09_production.py --resume_checkpoint ""
```

## Command Line Options

```
--resume_checkpoint PATH    Path to checkpoint (default: epoch_100.pt from permutations)
--epochs N                  Total epochs to train (default: 500)
--batch_size N              Batch size (default: 128)
--lr FLOAT                  Learning rate (default: 1e-4)
--data_percentage N         Data % to use (default: 100 = full dataset)
--no_wandb                  Disable wandb logging
--wandb_project NAME        Wandb project name (default: fluid-dynamics-ae)
--wandb_name NAME           Wandb run name (default: Model_09_Residual_AE_production)
```

## What Happens

1. **Loads checkpoint** from epoch 100 (trained on 10% data)
2. **Loads full dataset** (1M training, 1M validation samples)
3. **Continues training** from epoch 101 to 500 (or specified)
4. **Logs to wandb** with:
   - Model source code embedded in notes
   - Training/validation loss and RMSE
   - Epoch timing
   - Learning rate
5. **Saves checkpoints**:
   - Every 10 epochs: `encoder/saved_models/Model_09_Residual_AE_epoch_XXX.pt`
   - Best model: `encoder/saved_models/Model_09_Residual_AE_best.pt`
   - Uploaded to wandb automatically

## Expected Performance

With full dataset (1M rows):
- **Epoch time:** ~30 minutes
- **100 epochs:** ~50 hours
- **500 epochs:** ~250 hours (~10 days)

Consider:
- Run on GPU for 10x speedup
- Use smaller `--data_percentage` for testing
- Monitor wandb for early stopping

## Wandb Integration

The script automatically:
- Embeds full model source code in wandb notes
- Logs training metrics every epoch
- Saves checkpoints to wandb
- Tracks configuration parameters

View your runs at: https://wandb.ai/YOUR_USERNAME/fluid-dynamics-ae

## Checkpoint Contents

Each checkpoint includes:
```python
{
    'epoch': int,                    # Current epoch
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'train_rmse': float,             # Training RMSE at this epoch
    'val_rmse': float,               # Validation RMSE at this epoch
    'train_rmse_history': list,      # Full training history
    'val_rmse_history': list,        # Full validation history
    'best_val_rmse': float,          # Best validation RMSE so far
}
```

## Resume Training

If training is interrupted, simply re-run with the same command:

```bash
# Automatically resumes from latest checkpoint
python encoder/train_model_09_production.py
```

Or specify a different checkpoint:

```bash
python encoder/train_model_09_production.py \
    --resume_checkpoint encoder/saved_models/Model_09_Residual_AE_epoch_200.pt
```

## Testing Different Configurations

### Quick test (1 hour)
```bash
python encoder/train_model_09_production.py \
    --epochs 110 \
    --data_percentage 10
```

### Medium run (overnight, ~8 hours)
```bash
python encoder/train_model_09_production.py \
    --epochs 120 \
    --data_percentage 50
```

### Full production run (10 days)
```bash
python encoder/train_model_09_production.py \
    --epochs 500 \
    --data_percentage 100
```

## Model Architecture Details

From `encoder/permutations/model_09_residual_ae.py`:

```python
class ResidualBlock(nn.Module):
    # Two linear layers with LayerNorm and skip connection
    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = out + residual  # Skip connection!
        return self.activation(out)

class ResidualAE(nn.Module):
    # 375 → 250 → 150 → 100 → 47 (encoder)
    # 47 → 100 → 150 → 250 → 375 (decoder)
    # Each transition includes a ResidualBlock
```

**Loss Function:**
```
Total Loss = MSE(reconstruction, input) + 0.00005 * L2(latent)
```

## Comparison with Original WAE

| Metric | Original WAE | Model_09 (10% data) | Expected Model_09 (100% data) |
|--------|-------------|-------------------|----------------------------|
| Val RMSE | ~0.047 | 0.006765 | < 0.005 (estimated) |
| Architecture | Linear layers | Residual blocks | Residual blocks |
| Training | Stable | Very stable | Very stable |

Model_09 achieves **7x better RMSE** than the original WAE!

## Next Steps

1. **Start training on full dataset:**
   ```bash
   python encoder/train_model_09_production.py
   ```

2. **Monitor on wandb:** Check https://wandb.ai for live training metrics

3. **Use best model:** Load from `encoder/saved_models/Model_09_Residual_AE_best.pt`

4. **Compare with original:** Test on your fluid dynamics applications

## Loading the Trained Model

```python
import torch
from encoder.permutations.model_09_residual_ae import ResidualAE

# Load best model
model = ResidualAE()
checkpoint = torch.load('encoder/saved_models/Model_09_Residual_AE_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    reconstructed, latent = model(input_data)
```

## Files

- **Model:** `encoder/permutations/model_09_residual_ae.py`
- **Training script:** `encoder/train_model_09_production.py`
- **Checkpoints:** `encoder/saved_models/Model_09_Residual_AE_*.pt`
- **Initial checkpoint:** `encoder/permutations/checkpoints/Model_09_Residual_AE/Model_09_Residual_AE_epoch_100.pt`

## Support

The training script handles:
- ✓ CUDA/MPS/CPU detection
- ✓ Automatic checkpoint saving
- ✓ Best model tracking
- ✓ Wandb logging with model source
- ✓ Resume from any epoch
- ✓ Progress logging with ETA
- ✓ Data sampling for quick tests
