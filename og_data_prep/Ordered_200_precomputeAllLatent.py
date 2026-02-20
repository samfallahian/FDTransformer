"""
Ordered_200_precomputeAllLatent.py

This script enhances the cubed original data by adding a latent space representation.
It passes the 375-dimensional velocity vectors through a pre-trained AttentionSE 
Autoencoder (Model GEN3 05) to generate 47 latent dimensions for each row.

Loosely based on cube_centroid_mapping/Ordered_050_Add_Latent_Space.py

Flow:
    [ Final Cubed OG Data (.pkl.gz) ]       [ Pre-trained Model GEN3 05 (.pt) ]
                |                                        |
                |                                        v
                |                          1. Load model architecture (models.py)
                |                             and weights (_absolute_best.pt).
                |                                        |
                v                                        |
    2. Process Files (Sequential) <----------------------+
       For each file:
         a. Extract 375 velocity columns.
         b. Batch process through Model Encoder (using GPU/MPS if avail).
         c. Generate 47 latent features.
         d. Append latent columns to original DataFrame.
                |
                v
    [ Output: Final_Cubed_OG_Data_wLatent/{input_file_path}.pkl.gz ]
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import time
import argparse
from tqdm import tqdm

# Add the root directory to the path for import resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the model
from encoder.autoencoderGEN3.models import get_model_by_index

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

def process_file(input_path, output_path, model, device, batch_size=4096):
    """Process a single file, computing latent space for each row in batches."""
    try:
        df = pd.read_pickle(input_path, compression='gzip')
        if df.empty:
            return True

        # Identify velocity columns (should be 375 columns)
        # Standard metadata columns to exclude
        metadata_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'time', 'distance', 'vx_original', 'vy_original', 'vz_original', 'centroid_x', 'centroid_y', 'centroid_z']
        velocity_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith('latent_')]
        
        # If we have exactly 375 columns that start with 'velocity_', use those
        vel_prefix_cols = [c for c in df.columns if c.startswith('velocity_')]
        if len(vel_prefix_cols) == 375:
            velocity_cols = vel_prefix_cols
        
        if len(velocity_cols) != 375:
             return f"Error in {os.path.basename(input_path)}: Expected 375 velocity columns, found {len(velocity_cols)}"

        # Convert to numpy array for batching
        data_np = df[velocity_cols].values.astype(np.float32)
        
        latent_outputs = []
        
        with torch.no_grad():
            for i in range(0, len(data_np), batch_size):
                batch = data_np[i : i + batch_size]
                batch_tensor = torch.from_numpy(batch).to(device)
                
                # Model's encode method returns the latent representation
                latent_batch = model.encode(batch_tensor)
                latent_outputs.append(latent_batch.cpu().numpy())
        
        latent_all = np.concatenate(latent_outputs, axis=0)
        
        # Create new columns for latent space (47 columns)
        latent_df = pd.DataFrame(
            latent_all, 
            columns=[f"latent_{i}" for i in range(1, 48)],
            index=df.index,
            dtype=np.float32
        )
        
        # Append to original dataframe
        # Drop existing latent columns if they exist to avoid duplicates if re-run on already processed data
        df = df.drop(columns=[c for c in df.columns if c.startswith('latent_')], errors='ignore')
        final_df = pd.concat([df, latent_df], axis=1)
        
        # Save to net-new file with pkl.gz compression
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_pickle(output_path, compression='gzip')
        
        return True
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Precompute latent space for cubed OG data.")
    parser.add_argument("--first_only", action="store_true", help="Only process the first file and exit.")
    parser.add_argument("--batch_size", type=int, default=4096, help="Number of rows to process at once in the model.")
    parser.add_argument("--input_file", type=str, help="Process a specific input file.")
    args = parser.parse_args()

    # Hardware acceleration check with colorful banner
    device = accelerator_report()

    # Paths
    input_root = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data"
    output_root = "/Users/kkreth/PycharmProjects/data/Final_Cubed_OG_Data_wLatent"
    
    # Construct model path
    model_path = os.path.join(project_root, "encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best.pt")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        # Try relative path if absolute fails (though project_root should be correct)
        model_path = "encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best.pt"
        if not os.path.exists(model_path):
            print(f"Model still not found at {model_path}")
            return

    # Load the Model GEN3 05 (AttentionSE)
    print(f"Loading model architecture (Model GEN3 05)...")
    model = get_model_by_index(4) # 4 is Model_GEN3_05_AttentionSE
    
    # Load weights
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

    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Input file not found: {args.input_file}")
            return
        files_to_process = [args.input_file]
    else:
        # Find files
        files_to_process = []
        for root, dirs, files in os.walk(input_root):
            for file in files:
                if file.endswith(".pkl.gz"):
                    files_to_process.append(os.path.join(root, file))
        
        files_to_process.sort()

    if not files_to_process:
        print(f"No .pkl.gz files found in {input_root}")
        return

    if args.first_only and not args.input_file:
        files_to_process = files_to_process[:1]
        print("Flag --first_only is set. Processing 1 file.")

    print(f"Starting processing of {len(files_to_process)} file(s)...")
    
    start_time = time.time()
    results_count = 0
    errors = []
    
    # Process files
    for input_file in tqdm(files_to_process, desc="Progress"):
        if args.input_file:
            # If a specific file is given, we might need to figure out its relative path to input_root
            # to determine output location, or just put it in output_root.
            # Let's try to be smart if it's within input_root.
            if input_file.startswith(input_root):
                rel_path = os.path.relpath(input_file, input_root)
            else:
                rel_path = os.path.basename(input_file)
        else:
            rel_path = os.path.relpath(input_file, input_root)
            
        output_file = os.path.join(output_root, rel_path)
        
        res = process_file(input_file, output_file, model, device, args.batch_size)
        if res is True:
            results_count += 1
        else:
            errors.append(res)
            
    elapsed = time.time() - start_time
    print(f"\nCompleted {results_count} files in {elapsed:.2f} seconds.")
    
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
