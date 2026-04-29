import os
import sys
import pandas as pd
import numpy as np
import torch
import time
import argparse

# Add the root directory to the path for import resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the model from GEN3
from encoder.autoencoderGEN3.models import get_model_by_index
from pipeline_config import add_config_argument, resolve_path

# === Accelerator detection & colorful report ===
CSI = "\033["
RESET = f"{CSI}0m"
COLORS = [31, 33, 32, 36, 34, 35]  # R, Y, G, C, B, M

def rainbow(msg: str) -> str:
    out = []
    k = 0
    for ch in msg:
        if ch.strip():
            out.append(f"{CSI}{COLORS[k % len(COLORS)]}m{ch}{RESET}")
            k += 1
        else:
            out.append(ch)
    return ''.join(out)

def resolve_device(device_name: str = "auto") -> torch.device:
    """Resolve the requested compute device."""
    has_cuda = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available()

    if device_name == "auto":
        if has_cuda:
            return torch.device('cuda')
        if mps_avail:
            return torch.device('mps')
        return torch.device('cpu')

    if device_name == "cuda" and not has_cuda:
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if device_name == "mps" and not mps_avail:
        raise RuntimeError("MPS was requested with --device mps, but it is not available.")

    return torch.device(device_name)


def accelerator_report(device_name: str = "auto"):
    """Detect CUDA/MPS/CPU and print a colorful diagnostic. Returns device."""
    has_cuda = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available()
    device = resolve_device(device_name)

    lines = [
        f"Requested device: {device_name}",
        f"Selected device: {device}",
        f"CUDA available: {has_cuda}",
        f"MPS available: {mps_avail}",
    ]

    print("\n" + "="*50)
    for ln in lines:
        print(rainbow(ln))
    print("="*50 + "\n")

    return device

def main():
    parser = argparse.ArgumentParser(description="Validate one latent-enriched file by decoding latent vectors.")
    add_config_argument(parser)
    parser.add_argument("--input_file", help="Latent-enriched .pkl.gz file to validate.")
    parser.add_argument("--model_path", help="Autoencoder checkpoint path.")
    parser.add_argument("--output_csv", help="CSV path for validation results.")
    parser.add_argument("--model_index", type=int, default=4, help="Model architecture index.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Compute device for model inference. Defaults to auto.")
    args = parser.parse_args()

    try:
        device = accelerator_report(args.device)
    except RuntimeError as exc:
        print(f"Device error: {exc}")
        return

    input_file = resolve_path(args.config, "validation_input_file", args.input_file)
    model_path = resolve_path(args.config, "autoencoder_model_path", args.model_path)
    output_csv = resolve_path(args.config, "validation_results_csv", args.output_csv)

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 1. Load the model (Model GEN3 05 AttentionSE)
    print(f"Loading model architecture (index {args.model_index})...")
    model = get_model_by_index(args.model_index)
    
    print(f"Loading weights from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return

    model.to(device)
    model.eval()

    # 2. Load the data
    print(f"Loading data from {input_file}...")
    df = pd.read_pickle(input_file, compression='gzip')
    
    # Identify latent columns (47)
    latent_cols = [f"latent_{i}" for i in range(1, 48)]
    if not all(col in df.columns for col in latent_cols):
        print("Error: Missing some latent columns in the input file.")
        return

    # Identify original velocity columns (375)
    # Following the logic from Ordered_200
    metadata_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'time', 'distance', 'vx_original', 'vy_original', 'vz_original', 'centroid_x', 'centroid_y', 'centroid_z', 'original_vx', 'original_vy', 'original_vz']
    velocity_cols = [c for c in df.columns if c.startswith('velocity_')]
    
    if len(velocity_cols) != 375:
        # Fallback if prefix matching fails
        velocity_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('latent_')]
        if len(velocity_cols) != 375:
             print(f"Error: Expected 375 velocity columns, found {len(velocity_cols)}")
             return

    print(f"Found {len(velocity_cols)} velocity columns and {len(latent_cols)} latent columns.")

    # 3. Invoke the model to DECODE
    latent_data = df[latent_cols].values.astype(np.float32)
    latent_tensor = torch.from_numpy(latent_data).to(device)

    print("Decoding latent space...")
    with torch.no_grad():
        decoded_output = model.decode(latent_tensor)
        decoded_np = decoded_output.cpu().numpy()

    # 4. Place validation results in validation_1 to validation_375
    validation_cols = [f"validation_{i}" for i in range(1, 376)]
    validation_df = pd.DataFrame(decoded_np, columns=validation_cols, index=df.index)
    
    # 5. Calculate error columns
    # original input columns vs validation columns
    original_np = df[velocity_cols].values.astype(np.float32)
    
    # Absolute error per element
    abs_error_matrix = np.abs(original_np - decoded_np)
    
    # Average error (Mean Absolute Error per row)
    df['average_error'] = np.mean(abs_error_matrix, axis=1)
    
    # Absolute amount of error (Sum of Absolute Errors per row)
    df['absolute_error'] = np.sum(abs_error_matrix, axis=1)

    # Concatenate validation columns
    df = pd.concat([df, validation_df], axis=1)

    # 6. Save to CSV
    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
