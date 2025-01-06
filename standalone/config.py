from pathlib import Path
import torch

class Config:
    ROOT_DIR = Path("/work/pi_bseyedaghazadeh_umassd_edu")
    INPUT_DIR = ROOT_DIR / "DL-PTV.backup"
    OUTPUT_BASE_DIR = ROOT_DIR / "DL-PTV/roshni"

    # Data types mapping
    COLUMN_DTYPES = {
        "x": torch.int32,
        "y": torch.int32,
        "z": torch.int32,
        "vx": torch.float32,
        "vy": torch.float32,
        "vz": torch.float32,
        "time": torch.int32,
        "distance": torch.int32
    }
    
    # Experiments configuration
    EXPERIMENTS = {
        "3p6": {
            "input_file": INPUT_DIR / "3p6.pkl",
            "output_dir": OUTPUT_BASE_DIR / "3p6",
        }
    }
    
    # Columns to drop
    COLUMNS_TO_DROP = ["px", "py", "pz"] 