import socket
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
class Config:
    HOSTNAME = socket.gethostname()
    logger.info(f"Detected hostname: {HOSTNAME}")

    if "unity" in HOSTNAME or "gypsum" in HOSTNAME:
        ROOT_DIR = Path("/work/pi_bseyedaghazadeh_umassd_edu")
        logger.info(f"ROOT_DIR set to: {ROOT_DIR}")
    else:
        ROOT_DIR = Path("/default/root/dir")  # Default for other hosts
        logger.warning(f"ROOT_DIR set to default: {ROOT_DIR}")

    USERNAME = Path.home().parts[-1]
    HOME_DIR = Path(f"/home/{USERNAME}")
    logger.info(f"HOME_DIR set to: {HOME_DIR}")
    INPUT_DIR = ROOT_DIR / "DL-PTV.backup"
    OUTPUT_BASE_DIR = ROOT_DIR / "DL-PTV/roshni"
    LOG_FILE = HOME_DIR / "logs/raw_data_process.log"
    METADATA_PATH = HOME_DIR / "cgan_deployment/configs/Umass_experiments.txt"

    # Data types mapping
    COLUMN_DTYPES = {
        "x": "int32",
        "y": "int32",
        "z": "int32",
        "vx": "float32",
        "vy": "float32",
        "vz": "float32",
        "time": "int32",
        "distance": "int32",
    }

    EXPERIMENTS = {
        "10p4": {
            "input_file": INPUT_DIR / "10p4.pkl",
            "output_dir": OUTPUT_BASE_DIR / "10p4",
        },
        "3p6": {
            "input_file": INPUT_DIR / "3p6.pkl",
        "output_dir": OUTPUT_BASE_DIR / "3p6",
        }
    }
    COLUMNS_TO_DROP = ["px", "py", "pz"]