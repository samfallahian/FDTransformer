# Autoencoder Permutations Experiment

This directory contains 10 different autoencoder architectures for comparing approaches to encoding fluid dynamics velocity data (375-dimensional input → 47-dimensional latent space).

## Configuration

Edit these variables at the top of `run_all_experiments.py`:

```python
NUMEPOCHS = 100      # Number of training epochs per model
THREAD_COUNT = 2     # Number of models to train simultaneously
```

## Models

All models use the same input/output dimensions (375 → 47) to ensure fair comparison.

### 1. Standard VAE
**File:** `model_01_standard_vae.py`
**Loss:** MSE reconstruction + KL divergence
**Description:** Classic Variational Autoencoder with Gaussian prior on latent space.

### 2. β-VAE
**File:** `model_02_beta_vae.py`
**Loss:** MSE reconstruction + β·KL divergence (β=4.0)
**Description:** Adjustable β parameter to balance reconstruction vs. disentanglement.

### 3. Sparse Autoencoder
**File:** `model_03_sparse_ae.py`
**Loss:** MSE reconstruction + L1 penalty on latent activations
**Description:** Encourages sparse representations in latent space.

### 4. Contractive Autoencoder
**File:** `model_04_contractive_ae.py`
**Loss:** MSE reconstruction + λ·||∂h/∂x||²
**Description:** Penalizes Jacobian to make representations robust to input perturbations.

### 5. Denoising Autoencoder
**File:** `model_05_denoising_ae.py`
**Loss:** MSE between reconstruction and clean input
**Description:** Trains with Gaussian noise added to inputs (noise_factor=0.3).

### 6. Adversarial Autoencoder
**File:** `model_06_adversarial_ae.py`
**Loss:** MSE reconstruction + adversarial loss
**Description:** Uses discriminator network to match latent distribution to prior.

### 7. VQ-VAE (Vector Quantized VAE)
**File:** `model_07_vq_vae.py`
**Loss:** MSE reconstruction + codebook + commitment losses
**Description:** Discrete latent space with 512 learnable codebook vectors.

### 8. Deep Autoencoder
**File:** `model_08_deep_ae.py`
**Loss:** MSE reconstruction + L2 regularization
**Description:** 6+ hidden layers with varied activations (GELU, ELU, LeakyReLU) and LayerNorm.

### 9. Residual Autoencoder
**File:** `model_09_residual_ae.py`
**Loss:** MSE reconstruction + L2 regularization
**Description:** Skip connections (residual blocks) for better gradient flow.

### 10. Mixture Density Autoencoder
**File:** `model_10_mixture_ae.py`
**Loss:** Negative log likelihood of Gaussian mixture + MSE
**Description:** Predicts parameters of 3-component Gaussian mixture for each output dimension.

## Usage

### Run All Experiments

```bash
cd /Users/kkreth/PycharmProjects/cgan
python -m encoder.permutations.run_all_experiments
```

### Results

Results are saved to `encoder/permutations/results/` with timestamp:
- `experiment_results_YYYYMMDD_HHMMSS.json` - Full results with training history
- `experiment_results_YYYYMMDD_HHMMSS.csv` - Summary table

The script prints a ranked comparison table showing:
- Model names
- Final validation RMSE
- Final training RMSE
- Training time

### Example Output

```
EXPERIMENT RESULTS SUMMARY
==================================================================================
Rank   Model                          Val RMSE     Train RMSE   Time (min)
----------------------------------------------------------------------------------
1      Model_03_Sparse_AE             0.012345     0.011234     12.34
2      Model_09_Residual_AE           0.012456     0.011345     15.67
3      Model_01_Standard_VAE          0.012567     0.011456     11.89
...

🏆 BEST MODEL: Model_03_Sparse_AE
   Validation RMSE: 0.012345
   Description: Sparse AE with L1 regularization
```

## Training Details

All models use:
- **Optimizer:** Adam
- **Learning Rate:** 1e-4
- **Batch Size:** 128
- **Data:** Same training/validation split from `training_auto_encoder.pkl` and `validation_auto_encoder.pkl`
- **Architecture:** Same hidden layer sizes where applicable (250 → 150 → 100 → 47)

## Baseline Comparison

The original WAE (Wasserstein Autoencoder) model is located at:
- **Weights:** `encoder/saved_models/WAE_Cached_012_H200_FINAL.pt`
- **Training:** `encoder/train_WAE_01_cached.py`
- **Architecture:** `encoder/model_WAE_01.py`

This experiment compares 10 alternative approaches to the baseline WAE approach.

## Extending

To add a new model:

1. Create `model_XX_name.py` with a class that has:
   - `forward(x)` method returning `(recon_x, z)` or `(recon_x, z, extra)`
   - `loss_function(...)` returning `(total_loss, recon_loss, aux1, aux2)`

2. Add entry to `MODELS` list in `run_all_experiments.py`

3. Run experiments

## Notes

- Models are trained independently - no weights transferred between experiments
- RMSE is computed over all elements: sqrt(sum((pred - true)²) / total_elements)
- Training can be parallelized by adjusting `THREAD_COUNT` (2-3 recommended for local runs)
- Some models (e.g., Contractive AE) may be slower due to gradient computation
