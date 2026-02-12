import os
import sys
import pandas as pd
import numpy as np
import torch
import time

# Add the root directory to the path for import resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the model
from encoder.permutations.model_09_residual_ae import ResidualAE

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

def accelerator_report():
    """Detect CUDA/MPS/CPU and print a colorful diagnostic. Returns device."""
    has_cuda = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available()

    if has_cuda:
        device = torch.device('cuda')
    elif mps_avail:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    lines = [
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
    device = accelerator_report()

    # Paths
    input_file = "/Users/kkreth/PycharmProjects/data/simplified_cubes_wLatent/6p4/0006.pkl.gz"
    model_path = os.path.join(project_root, "encoder/saved_models/Model_09_Residual_AE_epoch_500.pt")
    output_csv = "validation_results_0006.csv"

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = ResidualAE()
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
    # Following the logic from Ordered_050
    metadata_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'time', 'distance', 'vx_original', 'vy_original', 'vz_original']
    # We also need to exclude the latent columns themselves
    original_velocity_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('latent_')]
    
    if len(original_velocity_cols) != 375:
        # Fallback to 'velocity_' prefix if needed
        original_velocity_cols = [c for c in df.columns if c.startswith('velocity_')]
        if len(original_velocity_cols) != 375:
             print(f"Error: Expected 375 velocity columns, found {len(original_velocity_cols)}")
             return

    print(f"Found {len(original_velocity_cols)} original velocity columns and {len(latent_cols)} latent columns.")

    # 3. Invoke the model to DECODE
    latent_data = df[latent_cols].values.astype(np.float32)
    latent_tensor = torch.from_numpy(latent_data).to(device)

    print("Decoding latent space...")
    with torch.no_grad():
        decoded_output = model.decode(latent_tensor)
        decoded_np = decoded_output.cpu().numpy()

    # 4. Place results in validation_1 to validation_375
    validation_cols = [f"validation_{i}" for i in range(1, 376)]
    validation_df = pd.DataFrame(decoded_np, columns=validation_cols, index=df.index)
    
    # 5. Calculate error columns
    # original input columns vs validation columns
    # We compare them in order. original_velocity_cols order should match validation_cols order.
    original_np = df[original_velocity_cols].values.astype(np.float32)
    
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
