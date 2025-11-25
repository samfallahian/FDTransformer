'''
Manifold Comparison for Two WAE Model Versions

This script compares the latent space representations of two different trained WAE models
by analyzing how well each preserves the local neighborhood structure from the original
(input) data space.

The key metrics used are:
  - Trustworthiness (T): Measures whether points that are neighbors in latent space
    were actually neighbors in the original space. Penalizes "false friends" - points
    that appear close in latent space but weren't close originally.

  - Continuity (C): Measures whether points that were neighbors in original space
    remain neighbors in latent space. Penalizes "missing neighbors" - points that
    were close originally but got separated in the encoding.

Both metrics range from 0 to 1, where 1 indicates perfect preservation of local structure.

Literature-based success criteria:
  - T, C > 0.9: Excellent local structure preservation
  - T, C > 0.85: Good preservation
  - T, C < 0.8: Poor preservation (latent space may not be reliable)

For comparing two models:
  - ΔT or ΔC < 0.05: Models are very similar in their neighborhood preservation
  - ΔT or ΔC > 0.1: Significant difference in how they encode local structure

References:
  - Venna & Kaski (2006): "Local multidimensional scaling" - introduced T&C metrics
  - Lee & Verleysen (2009): "Quality assessment of dimensionality reduction"
  - van der Maaten & Hinton (2008): t-SNE paper - used T to validate local structure
  - Espadoto et al. (2021): "Toward a Quantitative Survey of Dimension Reduction Techniques"
'''

import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Suppress numerical warnings from sklearn distance computations
# These occur with high-dimensional data but don't affect results
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

# Resolve project directories so we can import local modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Reuse existing model and helpers from training code
from encoder.model_WAE_01 import WAE  # noqa: E402
from Ordered_001_Initialize import HostPreferences  # noqa: E402
from encoder.train_WAE_01_cached import _load_cached_array, _load_preferences  # noqa: E402

# ------------------------------
# CONFIGURATION
# ------------------------------

# Paths to the two model checkpoints we want to compare
MODEL_V1_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_01_epoch_2870.pt"
MODEL_V2_PATH = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt"

# Random sample size from the validation set for analysis
# Larger samples give more robust metrics but take longer to compute
# NOTE: Computational complexity is O(N^2) for distance calculations
# 1000 samples: ~10 seconds, 2000 samples: ~40 seconds, 5000 samples: ~4 minutes
RANDOM_SAMPLE_SIZE = 2_000

# Inference batch size (tune if you hit memory limits)
BATCH_SIZE = 2048

# Reproducibility for row sampling
SEED = 12345

# Number of nearest neighbors to consider for T&C metrics
# Literature suggests k = 10-20 for datasets with thousands of points
# k = 30-50 for datasets with 10k+ points
# We'll compute multiple k values to see how metrics vary with neighborhood size
K_VALUES = [10, 15, 20, 30, 50]

# Output directory for results and figures
OUTPUT_DIR = os.path.join(PARENT_DIR, "encoder", "manifold_comparison")

# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def _pick_device() -> torch.device:
    """
    Select the best available device for PyTorch computations.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_validation_path(prefs: HostPreferences, default_name: str = 'validation_auto_encoder.pkl') -> str:
    """
    Determine the full path to the validation data file.

    Args:
        prefs: HostPreferences object containing root_path
        default_name: Name of the validation file

    Returns:
        str: Full path to the validation data file
    """
    root_dir = getattr(prefs, 'root_path', None) or os.getcwd()
    return os.path.join(root_dir, default_name)


def _load_model(checkpoint_path: str, device: torch.device) -> WAE:
    """
    Load a WAE model from a checkpoint file.

    This function handles multiple checkpoint formats robustly:
      - State dicts wrapped in a dictionary with 'model_state_dict' key
      - Raw state dicts

    Args:
        checkpoint_path: Path to the model checkpoint file
        device: PyTorch device to load the model onto

    Returns:
        WAE: The loaded model in evaluation mode

    Raises:
        ValueError: If the checkpoint format is not recognized
    """
    model = WAE().to(device)
    model.eval()

    # Load checkpoint - try with weights_only=False first for compatibility
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict):
        # Might be a raw state_dict
        state = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format at {checkpoint_path}")

    model.load_state_dict(state)
    return model


def _sample_rows(n_rows: int, sample_size: int, seed: int) -> np.ndarray:
    """
    Generate random row indices for sampling data.
    If sample_size >= n_rows, return all indices.

    Args:
        n_rows: Total number of rows available
        sample_size: Desired sample size
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Array of row indices to sample
    """
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_rows, size=sample_size, replace=False)


def _batched_encode(model: WAE, x_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    """
    Encode input data in batches to avoid memory issues.

    For a WAE model, the forward pass returns (reconstruction, latent).
    We only need the latent representation here.

    Args:
        model: The WAE model in eval mode
        x_np: Input data as numpy array (N x input_dim)
        device: PyTorch device
        batch_size: Number of samples per batch

    Returns:
        np.ndarray: Latent representations (N x latent_dim)
    """
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(0, x_np.shape[0], batch_size):
            batch = x_np[i:i+batch_size]
            x = torch.from_numpy(batch).to(device)
            # WAE forward pass returns (reconstruction, latent)
            _, z = model(x)
            latents.append(z.detach().cpu().numpy())
    return np.concatenate(latents, axis=0)


def continuity(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    """
    Compute the continuity metric for dimensionality reduction.

    Continuity measures whether points that were neighbors in the original space X
    remain neighbors in the reduced space Z. High continuity means the mapping
    preserves neighborhoods well (doesn't tear apart close points).

    The metric penalizes points that were k-nearest neighbors in X but are far apart
    in Z, weighted by how far apart they ended up.

    Mathematical definition:
        C(k) = 1 - (2/(N*k*(2N-3k-1))) * Σ_i Σ_{j∈U_i^X(k)} (r_Z(i,j) - k)

    where:
        - U_i^X(k) = k-nearest neighbors of point i in space X
        - r_Z(i,j) = rank of point j in space Z when sorted by distance from i
        - The sum penalizes neighbors in X that are NOT in the k-NN set in Z

    Args:
        X: Original/input space data (N x D_original)
        Z: Reduced/latent space data (N x D_latent)
        k: Number of nearest neighbors to consider

    Returns:
        float: Continuity score in [0, 1], where 1 = perfect preservation

    Reference:
        Venna & Kaski (2006), "Local multidimensional scaling"
    """
    n = X.shape[0]

    logger.debug(f"    Computing k-NN in original space...")
    # Find k-nearest neighbors in both spaces
    # We ask for k+1 because the first neighbor is the point itself
    nn_orig = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(X)
    nn_latent = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(Z)

    # Get the k-NN indices, excluding self (hence [:, 1:])
    orig_neighbors = nn_orig.kneighbors(X, return_distance=False)[:, 1:]

    logger.debug(f"    Computing k-NN in latent space...")
    # Get only k+1 neighbors in latent space (we don't need all N points)
    latent_dists, latent_neighbors = nn_latent.kneighbors(Z, return_distance=True)
    latent_neighbors = latent_neighbors[:, 1:]  # Exclude self

    # Convert to sets for fast lookup
    latent_neighbor_sets = [set(latent_neighbors[i]) for i in range(n)]

    logger.debug(f"    Computing ranks for missing neighbors...")
    # Accumulate the penalty V for missing neighbors
    V = 0

    # We need full distance matrix only for points with missing neighbors
    # This is more efficient than computing ranks for all points upfront
    for i in range(n):
        orig_neighbors_i = orig_neighbors[i]
        latent_neighbors_i = latent_neighbor_sets[i]

        # Find which original neighbors are missing from latent space
        missing = [j for j in orig_neighbors_i if j not in latent_neighbors_i]

        if missing:
            # Only now compute distances to all points for this query
            dists_i = np.linalg.norm(Z - Z[i:i+1], axis=1)
            # Get ranks (argsort gives indices that would sort the array)
            ranks = np.argsort(dists_i)
            # Create a reverse mapping: point_idx -> rank
            rank_map = {point_idx: rank for rank, point_idx in enumerate(ranks)}

            # Add penalties for missing neighbors
            for j in missing:
                r = rank_map[j]
                V += (r - k)

    # Normalize by the maximum possible penalty
    C = 1.0 - (2.0 / (n * k * (2*n - 3*k - 1))) * V
    return float(C)


def compute_manifold_metrics(X: np.ndarray, Z: np.ndarray, k_values: list) -> Tuple[dict, dict]:
    """
    Compute trustworthiness and continuity metrics for multiple neighborhood sizes.

    This gives us a more complete picture of how well the latent space preserves
    structure at different scales (small k = very local, large k = more global).

    Args:
        X: Original space data (N x D_original)
        Z: Latent space data (N x D_latent)
        k_values: List of k values (neighborhood sizes) to evaluate

    Returns:
        Tuple of two dictionaries:
            - T_scores: {k: trustworthiness_score}
            - C_scores: {k: continuity_score}
    """
    import time
    T_scores = {}
    C_scores = {}

    for idx, k in enumerate(k_values, 1):
        logger.info(f"  [{idx}/{len(k_values)}] Computing metrics for k={k}...")

        # Trustworthiness: from sklearn.manifold
        # This measures if latent neighbors were actually original neighbors
        t_start = time.time()
        T = trustworthiness(X, Z, n_neighbors=k)
        t_time = time.time() - t_start
        T_scores[k] = float(T)
        logger.info(f"    Trustworthiness: {T:.6f} (computed in {t_time:.1f}s)")

        # Continuity: custom implementation
        # This measures if original neighbors stayed together in latent space
        c_start = time.time()
        C = continuity(X, Z, k=k)
        c_time = time.time() - c_start
        C_scores[k] = float(C)
        logger.info(f"    Continuity:      {C:.6f} (computed in {c_time:.1f}s)")

    return T_scores, C_scores


def plot_comparison(T_v1: dict, C_v1: dict, T_v2: dict, C_v2: dict,
                   model_v1_name: str, model_v2_name: str, output_path: str) -> None:
    """
    Create a comparison plot showing T and C metrics for both models across k values.

    This visualization helps us see:
      1. How each model performs at different neighborhood scales
      2. Which model better preserves local structure (higher T & C)
      3. Whether the models diverge at certain scales

    Args:
        T_v1: Trustworthiness scores for model v1
        C_v1: Continuity scores for model v1
        T_v2: Trustworthiness scores for model v2
        C_v2: Continuity scores for model v2
        model_v1_name: Display name for model v1
        model_v2_name: Display name for model v2
        output_path: Where to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    k_vals = sorted(T_v1.keys())
    T_v1_vals = [T_v1[k] for k in k_vals]
    C_v1_vals = [C_v1[k] for k in k_vals]
    T_v2_vals = [T_v2[k] for k in k_vals]
    C_v2_vals = [C_v2[k] for k in k_vals]

    # Trustworthiness comparison
    ax1.plot(k_vals, T_v1_vals, 'o-', linewidth=2, markersize=8, label=model_v1_name, color='tab:blue')
    ax1.plot(k_vals, T_v2_vals, 's-', linewidth=2, markersize=8, label=model_v2_name, color='tab:orange')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (0.9)')
    ax1.axhline(y=0.85, color='yellow', linestyle='--', alpha=0.5, linewidth=1, label='Good (0.85)')
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Poor (0.8)')
    ax1.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax1.set_ylabel('Trustworthiness', fontsize=12)
    ax1.set_title('Trustworthiness: Do latent neighbors belong together?', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.75, 1.0])

    # Continuity comparison
    ax2.plot(k_vals, C_v1_vals, 'o-', linewidth=2, markersize=8, label=model_v1_name, color='tab:blue')
    ax2.plot(k_vals, C_v2_vals, 's-', linewidth=2, markersize=8, label=model_v2_name, color='tab:orange')
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (0.9)')
    ax2.axhline(y=0.85, color='yellow', linestyle='--', alpha=0.5, linewidth=1, label='Good (0.85)')
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Poor (0.8)')
    ax2.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax2.set_ylabel('Continuity', fontsize=12)
    ax2.set_title('Continuity: Do original neighbors stay together?', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.75, 1.0])

    plt.suptitle('Manifold Quality Comparison: Local Structure Preservation', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved comparison plot to: {output_path}")


def plot_difference(T_v1: dict, C_v1: dict, T_v2: dict, C_v2: dict,
                   model_v1_name: str, model_v2_name: str, output_path: str) -> None:
    """
    Create a difference plot showing ΔT and ΔC between the two models.

    Positive values mean model v1 is better, negative means model v2 is better.
    This helps us quickly identify where the models differ most.

    Args:
        T_v1, C_v1: Model v1 metrics
        T_v2, C_v2: Model v2 metrics
        model_v1_name: Display name for model v1
        model_v2_name: Display name for model v2
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    k_vals = sorted(T_v1.keys())
    delta_T = [T_v1[k] - T_v2[k] for k in k_vals]
    delta_C = [C_v1[k] - C_v2[k] for k in k_vals]

    ax.plot(k_vals, delta_T, 'o-', linewidth=2, markersize=8, label='ΔT (Trustworthiness)', color='tab:purple')
    ax.plot(k_vals, delta_C, 's-', linewidth=2, markersize=8, label='ΔC (Continuity)', color='tab:cyan')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1.5)
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='±0.05 (similar)')
    ax.axhline(y=-0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=1, label='±0.10 (significant)')
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax.set_ylabel(f'Difference ({model_v1_name} - {model_v2_name})', fontsize=12)
    ax.set_title('Model Comparison: Difference in Manifold Metrics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add annotation explaining interpretation
    textstr = f'Positive = {model_v1_name} better\nNegative = {model_v2_name} better'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved difference plot to: {output_path}")


def save_metrics_summary(T_v1: dict, C_v1: dict, T_v2: dict, C_v2: dict,
                        model_v1_name: str, model_v2_name: str, output_path: str) -> None:
    """
    Save a text summary of all computed metrics for easy reference.

    This creates a human-readable report with:
      - Raw T & C scores for both models at each k
      - Differences (ΔT, ΔC)
      - Mean scores across all k values
      - Interpretation guidance

    Args:
        T_v1, C_v1: Model v1 metrics
        T_v2, C_v2: Model v2 metrics
        model_v1_name: Display name for model v1
        model_v2_name: Display name for model v2
        output_path: Where to save the text file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MANIFOLD COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model v1: {model_v1_name}\n")
        f.write(f"Model v2: {model_v2_name}\n\n")

        f.write("Metrics computed:\n")
        f.write("  - Trustworthiness (T): Do latent neighbors belong together?\n")
        f.write("  - Continuity (C): Do original neighbors stay together?\n\n")

        f.write("=" * 80 + "\n")
        f.write("DETAILED SCORES\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'k':>5} | {'T_v1':>10} | {'C_v1':>10} | {'T_v2':>10} | {'C_v2':>10} | {'ΔT':>10} | {'ΔC':>10}\n")
        f.write("-" * 80 + "\n")

        k_vals = sorted(T_v1.keys())
        for k in k_vals:
            delta_T = T_v1[k] - T_v2[k]
            delta_C = C_v1[k] - C_v2[k]
            f.write(f"{k:5d} | {T_v1[k]:10.6f} | {C_v1[k]:10.6f} | {T_v2[k]:10.6f} | {C_v2[k]:10.6f} | "
                   f"{delta_T:+10.6f} | {delta_C:+10.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("MEAN SCORES (averaged over all k values)\n")
        f.write("=" * 80 + "\n\n")

        mean_T_v1 = np.mean([T_v1[k] for k in k_vals])
        mean_C_v1 = np.mean([C_v1[k] for k in k_vals])
        mean_T_v2 = np.mean([T_v2[k] for k in k_vals])
        mean_C_v2 = np.mean([C_v2[k] for k in k_vals])

        f.write(f"Model v1: T_mean = {mean_T_v1:.6f}, C_mean = {mean_C_v1:.6f}\n")
        f.write(f"Model v2: T_mean = {mean_T_v2:.6f}, C_mean = {mean_C_v2:.6f}\n\n")
        f.write(f"Difference: ΔT_mean = {mean_T_v1 - mean_T_v2:+.6f}, ΔC_mean = {mean_C_v1 - mean_C_v2:+.6f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION GUIDELINES\n")
        f.write("=" * 80 + "\n\n")

        f.write("Individual model quality (T & C scores):\n")
        f.write("  - Excellent: > 0.90\n")
        f.write("  - Good:      > 0.85\n")
        f.write("  - Poor:      < 0.80\n\n")

        f.write("Model similarity (|ΔT| & |ΔC| magnitudes):\n")
        f.write("  - Very similar:      < 0.05\n")
        f.write("  - Somewhat different: 0.05 - 0.10\n")
        f.write("  - Significantly different: > 0.10\n\n")

        f.write("Sign of differences:\n")
        f.write("  - Positive Δ: Model v1 better preserves that aspect of local structure\n")
        f.write("  - Negative Δ: Model v2 better preserves that aspect of local structure\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Saved metrics summary to: {output_path}")


def main():
    """
    Main execution function for manifold comparison analysis.

    Workflow:
      1. Load validation data (shared input for both models)
      2. Sample a subset for computational efficiency
      3. Load both model versions
      4. Encode the data with both models to get latent representations
      5. Compute T & C metrics for multiple k values for both models
      6. Generate comparison visualizations and summary report
    """
    logger.info("=" * 80)
    logger.info("STARTING MANIFOLD COMPARISON ANALYSIS")
    logger.info("=" * 80)

    # Step 1: Load validation data
    logger.info("\n[Step 1/6] Loading validation data...")
    prefs = _load_preferences(None)
    val_path = _resolve_validation_path(prefs)
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation cached file not found: {val_path}")

    val_np = _load_cached_array(val_path, limit=None)
    n_rows, n_dim = val_np.shape[0], val_np.shape[1]
    logger.info(f"Loaded validation data: {n_rows} samples, {n_dim} dimensions")

    if n_dim != 375:
        logger.warning(f"Expected 375 features, found {n_dim}. Proceeding anyway.")

    # Step 2: Sample a subset of the data
    logger.info(f"\n[Step 2/6] Sampling {RANDOM_SAMPLE_SIZE} rows for analysis...")
    idx = _sample_rows(n_rows, RANDOM_SAMPLE_SIZE, SEED)
    sample_np = val_np[idx]
    del val_np  # Free memory
    logger.info(f"Sample shape: {sample_np.shape}")

    # Normalize the input data for better numerical stability in distance calculations
    # This ensures all features contribute equally to distance metrics
    logger.info("  Normalizing input data (StandardScaler)...")
    scaler = StandardScaler()
    sample_np_normalized = scaler.fit_transform(sample_np)
    logger.info(f"  Normalized data: mean≈0, std≈1")

    # Step 3: Select device and load models
    device = _pick_device()
    logger.info(f"\n[Step 3/6] Loading models on device: {device}")

    model_v1 = _load_model(MODEL_V1_PATH, device)
    logger.info(f"Loaded model v1 from: {MODEL_V1_PATH}")

    model_v2 = _load_model(MODEL_V2_PATH, device)
    logger.info(f"Loaded model v2 from: {MODEL_V2_PATH}")

    # Step 4: Encode data with both models
    logger.info(f"\n[Step 4/6] Encoding data with both models (batch_size={BATCH_SIZE})...")

    logger.info("  Encoding with model v1...")
    Z_v1 = _batched_encode(model_v1, sample_np, device, BATCH_SIZE)
    logger.info(f"  Model v1 latent shape: {Z_v1.shape}")

    logger.info("  Encoding with model v2...")
    Z_v2 = _batched_encode(model_v2, sample_np, device, BATCH_SIZE)
    logger.info(f"  Model v2 latent shape: {Z_v2.shape}")

    # Step 5: Compute manifold metrics
    # Use normalized input data for better numerical stability
    logger.info(f"\n[Step 5/6] Computing manifold metrics for k={K_VALUES}...")

    logger.info("Model v1 metrics:")
    T_v1, C_v1 = compute_manifold_metrics(sample_np_normalized, Z_v1, K_VALUES)

    logger.info("Model v2 metrics:")
    T_v2, C_v2 = compute_manifold_metrics(sample_np_normalized, Z_v2, K_VALUES)

    # Step 6: Generate outputs
    logger.info("\n[Step 6/6] Generating comparison visualizations and summary...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_v1_basename = os.path.splitext(os.path.basename(MODEL_V1_PATH))[0]
    model_v2_basename = os.path.splitext(os.path.basename(MODEL_V2_PATH))[0]

    # Comparison plot
    comparison_plot_path = os.path.join(
        OUTPUT_DIR,
        f"manifold_comparison_{timestamp}.png"
    )
    plot_comparison(T_v1, C_v1, T_v2, C_v2, model_v1_basename, model_v2_basename, comparison_plot_path)

    # Difference plot
    difference_plot_path = os.path.join(
        OUTPUT_DIR,
        f"manifold_difference_{timestamp}.png"
    )
    plot_difference(T_v1, C_v1, T_v2, C_v2, model_v1_basename, model_v2_basename, difference_plot_path)

    # Text summary
    summary_path = os.path.join(
        OUTPUT_DIR,
        f"manifold_summary_{timestamp}.txt"
    )
    save_metrics_summary(T_v1, C_v1, T_v2, C_v2, model_v1_basename, model_v2_basename, summary_path)

    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {OUTPUT_DIR}")
    logger.info(f"  - Comparison plot: {os.path.basename(comparison_plot_path)}")
    logger.info(f"  - Difference plot: {os.path.basename(difference_plot_path)}")
    logger.info(f"  - Text summary:    {os.path.basename(summary_path)}")

    # Print quick summary to console
    logger.info("\n" + "=" * 80)
    logger.info("QUICK SUMMARY")
    logger.info("=" * 80)

    mean_T_v1 = np.mean([T_v1[k] for k in K_VALUES])
    mean_C_v1 = np.mean([C_v1[k] for k in K_VALUES])
    mean_T_v2 = np.mean([T_v2[k] for k in K_VALUES])
    mean_C_v2 = np.mean([C_v2[k] for k in K_VALUES])

    logger.info(f"\n{model_v1_basename}:")
    logger.info(f"  Mean Trustworthiness: {mean_T_v1:.4f}")
    logger.info(f"  Mean Continuity:      {mean_C_v1:.4f}")

    logger.info(f"\n{model_v2_basename}:")
    logger.info(f"  Mean Trustworthiness: {mean_T_v2:.4f}")
    logger.info(f"  Mean Continuity:      {mean_C_v2:.4f}")

    logger.info(f"\nDifferences:")
    logger.info(f"  ΔT_mean: {mean_T_v1 - mean_T_v2:+.4f}")
    logger.info(f"  ΔC_mean: {mean_C_v1 - mean_C_v2:+.4f}")

    # Interpretation
    abs_delta_T = abs(mean_T_v1 - mean_T_v2)
    abs_delta_C = abs(mean_C_v1 - mean_C_v2)

    if abs_delta_T < 0.05 and abs_delta_C < 0.05:
        logger.info("\nInterpretation: Models are VERY SIMILAR in local structure preservation.")
    elif abs_delta_T < 0.1 and abs_delta_C < 0.1:
        logger.info("\nInterpretation: Models are SOMEWHAT DIFFERENT in local structure preservation.")
    else:
        logger.info("\nInterpretation: Models are SIGNIFICANTLY DIFFERENT in local structure preservation.")


if __name__ == "__main__":
    main()
