import os
import sys
import numpy as np
import h5py
import torch
import random
from terminaltables import AsciiTable

# Add root to sys.path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from transformer.dataset import TransformerDataset
except ImportError:
    # If run from within transformer dir
    from dataset import TransformerDataset

# ANSI Colors
CSI = "\033["
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"
RED = f"{CSI}31m"
GREEN = f"{CSI}32m"
YELLOW = f"{CSI}33m"
BLUE = f"{CSI}34m"
MAGENTA = f"{CSI}35m"
CYAN = f"{CSI}36m"

def rainbow(msg: str) -> str:
    COLORS = [31, 33, 32, 36, 34, 35]
    out = []
    k = 0
    for ch in msg:
        if ch.strip():
            out.append(f"{CSI}{COLORS[k % len(COLORS)]}m{ch}{RESET}")
            k += 1
        else:
            out.append(ch)
    return ''.join(out)

def section_header(title):
    print("\n" + "="*80)
    print(rainbow(f"  {title.upper()}  "))
    print("="*80)

def validate_sample(sample, sample_idx):
    """
    Validate a single sample of shape (8, 26, 52).
    Returns a report dict.
    """
    # Features: 47 (latent) + 3 (x,y,z) + 1 (rel_time) + 1 (param_val) = 52
    # Indices: 47:x, 48:y, 49:z, 50:rel_t, 51:param
    
    x_data = sample[:, :, 47]
    y_data = sample[:, :, 48]
    z_data = sample[:, :, 49]
    t_data = sample[:, :, 50]
    p_data = sample[:, :, 51]
    
    # Check consistency of Y, Z, Param across all points in sample
    y_val = y_data[0, 0]
    z_val = z_data[0, 0]
    p_val = p_data[0, 0]
    
    y_consistent = np.all(y_data == y_val)
    z_consistent = np.all(z_data == z_val)
    p_consistent = np.all(p_data == p_val)
    
    # Check X coordinates consistency across time slices
    x_slices_consistent = True
    for i in range(1, 8):
        if not np.all(x_data[i] == x_data[0]):
            x_slices_consistent = False
            break
            
    # Check relative time incrementing
    t_consistent = True
    for i in range(8):
        if not np.all(t_data[i] == i):
            t_consistent = False
            break
            
    return {
        'y': y_val,
        'z': z_val,
        'param': p_val,
        'x_coords': x_data[0].tolist(),
        'y_consistent': y_consistent,
        'z_consistent': z_consistent,
        'p_consistent': p_consistent,
        'x_consistent': x_slices_consistent,
        't_consistent': t_consistent,
        'latent_mean': float(sample[:, :, :47].mean()),
        'latent_std': float(sample[:, :, :47].std())
    }

def print_validation_report(report, sample_idx):
    print(f"\n{BOLD}Sample Index: {CYAN}{sample_idx}{RESET}")
    
    # Metadata Table
    meta_data = [
        [BOLD + "Property" + RESET, BOLD + "Value" + RESET, BOLD + "Status" + RESET],
        ["Parameter Set", f"{report['param']:.1f}", GREEN + "OK" if report['p_consistent'] else RED + "FAIL"],
        ["Y Coordinate", f"{int(report['y'])}", GREEN + "OK" if report['y_consistent'] else RED + "FAIL"],
        ["Z Coordinate", f"{int(report['z'])}", GREEN + "OK" if report['z_consistent'] else RED + "FAIL"],
        ["X Consistency", "Across Time Slices", GREEN + "OK" if report['x_consistent'] else RED + "FAIL"],
        ["Time Sequences", "0 to 7", GREEN + "OK" if report['t_consistent'] else RED + "FAIL"],
    ]
    table = AsciiTable(meta_data)
    print(table.table)
    
    # Latent Stats
    print(f"{YELLOW}Latent Statistics: Mean={report['latent_mean']:.4f}, Std={report['latent_std']:.4f}{RESET}")
    
    # X Coordinates
    x_str = ", ".join([f"{int(x)}" for x in report['x_coords']])
    print(f"{BLUE}X-Axis Coordinates (26 points):{RESET}")
    print(f"[{x_str}]")

def main():
    h5_files = [
        os.path.join(project_root, "transformer/training_data.h5"),
        os.path.join(project_root, "transformer/validation_data.h5")
    ]
    
    section_header("Transformer Dataset Validation")
    
    for h5_path in h5_files:
        if not os.path.exists(h5_path):
            print(f"{RED}File not found: {h5_path}{RESET}")
            continue
            
        print(f"\n{BOLD}Interrogating: {MAGENTA}{os.path.basename(h5_path)}{RESET}")
        
        try:
            dataset = TransformerDataset(h5_path)
            print(f"Total samples: {BOLD}{len(dataset)}{RESET}")
            print(f"Feature description: {YELLOW}{dataset.feature_description}{RESET}")
            
            # Pick 3 random samples
            indices = random.sample(range(len(dataset)), min(3, len(dataset)))
            for idx in indices:
                sample = dataset[idx].numpy()
                report = validate_sample(sample, idx)
                print_validation_report(report, idx)
                
            # Perform a quick bulk validation
            print(f"\n{BOLD}Performing bulk consistency check (up to 100 samples)...{RESET}")
            bulk_indices = random.sample(range(len(dataset)), min(100, len(dataset)))
            all_ok = True
            for idx in bulk_indices:
                s = dataset[idx].numpy()
                r = validate_sample(s, idx)
                if not all([r['y_consistent'], r['z_consistent'], r['p_consistent'], r['x_consistent'], r['t_consistent']]):
                    print(f"{RED}Consistency failure at index {idx}!{RESET}")
                    all_ok = False
                    break
            
            if all_ok:
                print(f"{GREEN}SUCCESS: Bulk consistency check passed for {len(bulk_indices)} samples.{RESET}")
            
            dataset.close()
            
        except Exception as e:
            print(f"{RED}Error processing {h5_path}: {e}{RESET}")
            import traceback
            traceback.print_exc()

    print("\n" + rainbow("Validation Complete!") + "\n")

if __name__ == "__main__":
    main()
