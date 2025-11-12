import os
import sys
import math
import random
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

# Resolve project directories so we can import local modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# Reuse existing model and helpers from training code as much as possible
from encoder.model_WAE_01 import WAE  # noqa: E402
from Ordered_001_Initialize import HostPreferences  # noqa: E402
from encoder.train_WAE_01_cached import _load_cached_array, _load_preferences  # noqa: E402

# ------------------------------
# CONFIG (as requested: CAPS var)
# ------------------------------
MODEL_CHECKPOINT_PATH = \
    "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_01_epoch_2870.pt"
    #"/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_Cached_012_H200_FINAL.pt"

# Random sample size from the validation set
RANDOM_SAMPLE_SIZE = 100_000
# Inference batch size (tune if you hit memory limits)
BATCH_SIZE = 2048
# Reproducibility for row sampling
SEED = 12345
# Whether to use absolute residuals (False keeps signed residuals)
ABSOLUTE_RESIDUALS = False

# Output directory for figures
OUTPUT_DIR = os.path.join(PARENT_DIR, "encoder", "position_error_analysis")

# Plotting knobs
VIOLIN_SYMLOG_LINTHRESH = 1e-3  # symlog linear range around 0 for violin/split-violin
VIOLIN_SYMLOG_LINSCALE = 1.0    # scale of linear region size in decades
RIDGELINE_BINS = 100            # histogram bins for ridgeline
RIDGELINE_SMOOTH_WIN = 3        # simple moving-average smoothing window (odd)
SUBSET_STEP = 10                # sampling stride for subset plots (violin/ridgeline/ECDF)
SUBSET_MAX_POSITIONS = 60       # maximum number of subset positions to plot


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def _pick_device() -> torch.device:
    # Prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_validation_path(prefs: HostPreferences, default_name: str = 'validation_auto_encoder.pkl') -> str:
    root_dir = getattr(prefs, 'root_path', None) or os.getcwd()
    return os.path.join(root_dir, default_name)


def _load_model(checkpoint_path: str, device: torch.device) -> WAE:
    model = WAE().to(device)
    model.eval()
    # Load checkpoint formats robustly
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

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
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_rows, size=sample_size, replace=False)


def _batched_reconstruct(model: WAE, x_np: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_np.shape[0], batch_size):
            batch = x_np[i:i+batch_size]
            x = torch.from_numpy(batch).to(device)
            recon, _ = model(x)
            preds.append(recon.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_percentiles(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # residuals shape: [N, D]
    q15 = np.percentile(residuals, 15, axis=0)
    q25 = np.percentile(residuals, 25, axis=0)
    q50 = np.percentile(residuals, 50, axis=0)
    q75 = np.percentile(residuals, 75, axis=0)
    q85 = np.percentile(residuals, 85, axis=0)
    return q15, q25, q50, q75, q85


def plot_boxplot_all_positions(residuals: np.ndarray, title: Optional[str], output_path: str,
                                 q25: Optional[np.ndarray] = None, q50: Optional[np.ndarray] = None,
                                 q75: Optional[np.ndarray] = None) -> None:
    # residuals: [N, D]
    D = residuals.shape[1]
    data = [residuals[:, j] for j in range(D)]
    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(
        data,
        showfliers=False,           # remove outlier markers from the plot
        whis=(15, 85),              # whiskers at 15th and 85th percentiles
        widths=0.6,
        medianprops=dict(color='black', linewidth=0.8),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )
    # Optional overlays for Q1/Median/Q3
    if q25 is not None and q50 is not None and q75 is not None:
        xs = np.arange(1, D + 1)
        ax.plot(xs, q50, color='tab:red', linestyle='none', marker='.', markersize=2.5, alpha=0.7, label='Median (p50)')
        ax.plot(xs, q25, color='tab:blue', linestyle='none', marker='.', markersize=2.0, alpha=0.5, label='Q1 (p25)')
        ax.plot(xs, q75, color='tab:green', linestyle='none', marker='.', markersize=2.0, alpha=0.5, label='Q3 (p75)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.3)

    # Axis labels/title
    ax.set_xlabel('Position index (0-based)')
    ax.set_ylabel('Residual' + (' (absolute)' if ABSOLUTE_RESIDUALS else ''))
    if title:
        ax.set_title(title)

    # X-axis: show labels every 5 positions, with minor ticks every 1; rotate labels 90°
    ax.set_xlim(0, D + 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    # Format major tick labels to show 0-based position strings
    def _fmt_major_tick(x, pos=None):
        # x are at 1-based tick locations (5, 10, ...); convert to 0-based index
        xi = int(round(x))
        if xi < 1 or xi > D:
            return ''
        return str(xi - 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos=None: _fmt_major_tick(x, pos)))

    # Apply rotation and small font for x tick labels
    for label in ax.get_xticklabels(which='both'):
        label.set_rotation(90)
        label.set_fontsize(7)

    # Grid: keep Y-grid; add X-grid for major and light minor lines
    ax.grid(True, axis='y', alpha=0.3)
    ax.grid(True, axis='x', which='major', alpha=0.25, linewidth=0.6)
    ax.grid(True, axis='x', which='minor', alpha=0.12, linewidth=0.4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_violin_subset(residuals: np.ndarray, title: Optional[str], output_path: str,
                       step: int = SUBSET_STEP, max_positions: int = SUBSET_MAX_POSITIONS) -> None:
    """Violin plot for a subset of positions to visualize tails/density.
    - Picks every `step`-th position up to `max_positions` violins.
    - Uses a symlog Y scale for readability across magnitudes while preserving sign.
    """
    D = residuals.shape[1]
    # Choose indices
    idxs = list(range(0, D, step))
    if len(idxs) > max_positions:
        idxs = idxs[:max_positions]
    data = [residuals[:, j] for j in idxs]
    fig, ax = plt.subplots(figsize=(16, 6))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=False)
    # Light styling
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(1.0)
    # Symlog Y-scale for signed residuals (also works for absolute residuals)
    try:
        ax.set_yscale('symlog', linthresh=VIOLIN_SYMLOG_LINTHRESH, linscale=VIOLIN_SYMLOG_LINSCALE)
        ax.annotate(f"symlog (linthresh={VIOLIN_SYMLOG_LINTHRESH:g})", xy=(1.0, 1.02), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=8, color='gray')
    except Exception:
        pass
    ax.set_xlabel('Position index (subset)')
    ax.set_ylabel('Residual' + (' (absolute)' if ABSOLUTE_RESIDUALS else ''))
    if title:
        ax.set_title(title + ' | violin subset (symlog y)')
    ax.set_xticks(np.arange(1, len(idxs) + 1))
    ax.set_xticklabels([str(j) for j in idxs], rotation=90, fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_violin_subset_split(residuals: np.ndarray, title: Optional[str], output_path: str,
                             step: int = SUBSET_STEP, max_positions: int = SUBSET_MAX_POSITIONS,
                             widths: float = 0.9) -> None:
    """Split violin plot (negative vs positive residuals) for a subset of positions.
    - For each selected position, draw negative residuals (<=0) as left half, positives (>0) as right half.
    - Handles cases where one side has no data by skipping that half.
    - Uses a symlog Y scale for readability across magnitudes while preserving sign.
    """
    import matplotlib.patches as mpatches

    D = residuals.shape[1]
    # Choose indices
    idxs = list(range(0, D, step))
    if len(idxs) > max_positions:
        idxs = idxs[:max_positions]

    # Build per-position splits
    neg_data = []
    neg_posns = []
    pos_data = []
    pos_posns = []
    # x positions 1..K
    xs = np.arange(1, len(idxs) + 1)
    # Global y-limits from subset to construct clip rectangles
    all_vals = []
    for j in idxs:
        all_vals.append(residuals[:, j])
    y_min = float(np.min([np.min(v) for v in all_vals]))
    y_max = float(np.max([np.max(v) for v in all_vals]))

    for k, j in enumerate(idxs):
        col = residuals[:, j]
        neg = col[col <= 0.0]
        pos = col[col > 0.0]
        x = xs[k]
        if neg.size > 0:
            neg_data.append(neg)
            neg_posns.append(x)
        if pos.size > 0:
            pos_data.append(pos)
            pos_posns.append(x)

    fig, ax = plt.subplots(figsize=(16, 6))

    # Draw negatives and clip to left half
    neg_parts = None
    if len(neg_data) > 0:
        neg_parts = ax.violinplot(neg_data, positions=neg_posns, widths=widths,
                                   showmeans=False, showextrema=True, showmedians=False)
        for b, x in zip(neg_parts['bodies'], neg_posns):
            b.set_facecolor('#1f77b4')  # blue
            b.set_edgecolor('black')
            b.set_alpha(0.6)
            rect = mpatches.Rectangle((x - widths/2, y_min), widths/2, y_max - y_min,
                                      transform=ax.transData)
            b.set_clip_path(rect)
        # Style extrema lines
        if 'cbars' in neg_parts:
            neg_parts['cbars'].set_color('#1f77b4')
            neg_parts['cbars'].set_linewidth(0.8)
        if 'cmins' in neg_parts:
            neg_parts['cmins'].set_color('#1f77b4')
            neg_parts['cmins'].set_linewidth(0.8)
        if 'cmaxes' in neg_parts:
            neg_parts['cmaxes'].set_color('#1f77b4')
            neg_parts['cmaxes'].set_linewidth(0.8)

    # Draw positives and clip to right half
    pos_parts = None
    if len(pos_data) > 0:
        pos_parts = ax.violinplot(pos_data, positions=pos_posns, widths=widths,
                                   showmeans=False, showextrema=True, showmedians=False)
        for b, x in zip(pos_parts['bodies'], pos_posns):
            b.set_facecolor('#ff7f0e')  # orange
            b.set_edgecolor('black')
            b.set_alpha(0.6)
            rect = mpatches.Rectangle((x, y_min), widths/2, y_max - y_min, transform=ax.transData)
            b.set_clip_path(rect)
        # Style extrema lines
        if 'cbars' in pos_parts:
            pos_parts['cbars'].set_color('#ff7f0e')
            pos_parts['cbars'].set_linewidth(0.8)
        if 'cmins' in pos_parts:
            pos_parts['cmins'].set_color('#ff7f0e')
            pos_parts['cmins'].set_linewidth(0.8)
        if 'cmaxes' in pos_parts:
            pos_parts['cmaxes'].set_color('#ff7f0e')
            pos_parts['cmaxes'].set_linewidth(0.8)

    # Symlog Y-scale
    try:
        ax.set_yscale('symlog', linthresh=VIOLIN_SYMLOG_LINTHRESH, linscale=VIOLIN_SYMLOG_LINSCALE)
    except Exception:
        pass

    # Axes formatting
    ax.set_xlabel('Position index (subset) — split violin: left=neg, right=pos')
    ax.set_ylabel('Residual' + (' (absolute)' if ABSOLUTE_RESIDUALS else ''))
    if title:
        ax.set_title(title + ' | split violin subset (symlog y)')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(j) for j in idxs], rotation=90, fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # Legend
    neg_patch = mpatches.Patch(color='#1f77b4', alpha=0.6, label='Negative (<=0)')
    pos_patch = mpatches.Patch(color='#ff7f0e', alpha=0.6, label='Positive (>0)')
    ax.legend(handles=[neg_patch, pos_patch], loc='upper right', fontsize=8, framealpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode='same')


def plot_ridgeline_subset(residuals: np.ndarray, title: Optional[str], output_path: str,
                          step: int = SUBSET_STEP, max_positions: int = SUBSET_MAX_POSITIONS,
                          bins: int = RIDGELINE_BINS, smooth_win: int = RIDGELINE_SMOOTH_WIN) -> None:
    """Ridgeline (joy) plot for a subset of positions using shared residual axis.
    - Uses normalized histograms per position stacked with vertical offsets.
    - Avoids external dependencies (no seaborn/joypy).
    """
    D = residuals.shape[1]
    idxs = list(range(0, D, step))
    if len(idxs) > max_positions:
        idxs = idxs[:max_positions]

    subset_data = [residuals[:, j] for j in idxs]
    # Shared x-range across subset to align ridges
    x_min = float(min(np.min(d) for d in subset_data))
    x_max = float(max(np.max(d) for d in subset_data))
    # Slight padding
    pad = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    x_min -= pad
    x_max += pad

    fig, ax = plt.subplots(figsize=(16, 9))
    offsets = np.arange(len(idxs))  # vertical offsets
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(idxs)))

    for k, (j, off, color) in enumerate(zip(idxs, offsets, colors)):
        vals = residuals[:, j]
        hist, edges = np.histogram(vals, bins=bins, range=(x_min, x_max), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist = _moving_average(hist.astype(float), smooth_win)
        # Normalize each ridge to max 1 for consistent height
        if hist.max() > 0:
            hist = hist / hist.max()
        ax.fill_between(centers, off, off + hist, color=color, alpha=0.6, linewidth=0.5, edgecolor='k')
        # Optional median line
        med = float(np.median(vals))
        ax.plot([med, med], [off, off + 1.05], color='k', linewidth=0.6, alpha=0.7)

    ax.set_xlabel('Residual' + (' (absolute)' if ABSOLUTE_RESIDUALS else ''))
    ax.set_ylabel('Position index (subset)')
    if title:
        ax.set_title(title + ' | ridgeline subset')
    ax.set_yticks(offsets)
    ax.set_yticklabels([str(j) for j in idxs], fontsize=7)
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ecdf_subset(residuals: np.ndarray, title: Optional[str], output_path: str,
                     step: int = SUBSET_STEP, max_positions: int = SUBSET_MAX_POSITIONS,
                     highlight: int = 5) -> None:
    """ECDF plot for the same subset as violins.
    - Draws thin lines for each selected position; highlights the first few.
    """
    D = residuals.shape[1]
    idxs = list(range(0, D, step))
    if len(idxs) > max_positions:
        idxs = idxs[:max_positions]

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = plt.cm.tab20
    for k, j in enumerate(idxs):
        x = np.sort(residuals[:, j])
        y = np.linspace(0, 1, num=x.size, endpoint=True)
        lw = 1.5 if k < highlight else 0.7
        alpha = 0.9 if k < highlight else 0.4
        color = cmap(k % 20)
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=(str(j) if k < highlight else None))

    ax.set_xlabel('Residual' + (' (absolute)' if ABSOLUTE_RESIDUALS else ''))
    ax.set_ylabel('ECDF')
    if title:
        ax.set_title(title + ' | ECDF subset')
    if highlight > 0:
        ax.legend(title='Highlighted positions', fontsize=8, title_fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    # Preferences and validation data path
    prefs = _load_preferences(None)
    val_path = _resolve_validation_path(prefs)
    if not os.path.isfile(val_path):
        raise FileNotFoundError(f"Validation cached file not found: {val_path}")

    # Load validation array
    val_np = _load_cached_array(val_path, limit=None)  # dtype float32 ensured inside
    n_rows, n_dim = val_np.shape[0], val_np.shape[1]
    if n_dim != 375:
        logger.warning(f"Expected 375 features, found {n_dim}. Proceeding anyway.")

    # Subsample rows
    idx = _sample_rows(n_rows, RANDOM_SAMPLE_SIZE, SEED)
    sample_np = val_np[idx]
    del val_np

    device = _pick_device()
    logger.info(f"Using device: {device}")

    # Load model
    model = _load_model(MODEL_CHECKPOINT_PATH, device)
    logger.info(f"Loaded model checkpoint from: {MODEL_CHECKPOINT_PATH}")

    # Reconstruct
    recon_np = _batched_reconstruct(model, sample_np, device, BATCH_SIZE)

    # Compute residuals
    residuals = sample_np - recon_np
    if ABSOLUTE_RESIDUALS:
        residuals = np.abs(residuals)

    # Compute percentiles (p15, p25, p50, p75, p85) for logging and overlays
    p15, p25, p50, p75, p85 = compute_percentiles(residuals)
    logger.info("Per-position percentiles computed. Example (first 5 positions):")
    for j in range(min(5, residuals.shape[1])):
        logger.info(f"pos {j:3d}: p15={p15[j]: .4e} p25={p25[j]: .4e} p50={p50[j]: .4e} p75={p75[j]: .4e} p85={p85[j]: .4e}")

    # Plot boxplot across all positions with whis=(15,85) and quartile overlays
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.splitext(os.path.basename(MODEL_CHECKPOINT_PATH))[0]
    abs_suffix = "_ABS" if ABSOLUTE_RESIDUALS else ""
    out_png = os.path.join(
        OUTPUT_DIR,
        f"residual_boxplot_{base}{abs_suffix}_whis15_85_{timestamp}.png"
    )
    title = f"Residuals per position (N={residuals.shape[0]}) | {base}{' | abs' if ABSOLUTE_RESIDUALS else ''}"
    plot_boxplot_all_positions(residuals, title, out_png, q25=p25, q50=p50, q75=p75)
    logger.info(f"Saved residual boxplot to: {out_png}")

    # Violin plots (symlog): standard and split by sign
    out_violin_std = os.path.join(
        OUTPUT_DIR,
        f"residual_violin_subset_{base}{abs_suffix}_symlog_{timestamp}.png"
    )
    plot_violin_subset(residuals, title, out_violin_std, step=SUBSET_STEP, max_positions=SUBSET_MAX_POSITIONS)
    logger.info(f"Saved residual violin subset (symlog) to: {out_violin_std}")

    out_violin_split = os.path.join(
        OUTPUT_DIR,
        f"residual_violin_subset_split_{base}{abs_suffix}_{timestamp}.png"
    )
    plot_violin_subset_split(residuals, title, out_violin_split, step=SUBSET_STEP, max_positions=SUBSET_MAX_POSITIONS)
    logger.info(f"Saved residual split violin subset (symlog) to: {out_violin_split}")

    # Ridgeline plot (subset)
    out_ridge = os.path.join(
        OUTPUT_DIR,
        f"residual_ridgeline_subset_{base}{abs_suffix}_{timestamp}.png"
    )
    plot_ridgeline_subset(residuals, title, out_ridge, step=SUBSET_STEP, max_positions=SUBSET_MAX_POSITIONS,
                          bins=RIDGELINE_BINS, smooth_win=RIDGELINE_SMOOTH_WIN)
    logger.info(f"Saved residual ridgeline subset to: {out_ridge}")

    # ECDF plot (subset)
    out_ecdf = os.path.join(
        OUTPUT_DIR,
        f"residual_ecdf_subset_{base}{abs_suffix}_{timestamp}.png"
    )
    plot_ecdf_subset(residuals, title, out_ecdf, step=SUBSET_STEP, max_positions=SUBSET_MAX_POSITIONS, highlight=5)
    logger.info(f"Saved residual ECDF subset to: {out_ecdf}")


if __name__ == "__main__":
    main()
