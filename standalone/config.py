import os
import socket
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Config:
    SUBMIT_HOSTNAME = os.environ.get('HOSTNAME', socket.gethostname())
    logger.info(f"Hostname detected: {SUBMIT_HOSTNAME}")

    flag = True #Found a cluster

    CLUSTER_PATTERN = re.compile(r"(gpu|unity|gypsum|login\d+)", re.IGNORECASE)
    try:
        if CLUSTER_PATTERN.search(SUBMIT_HOSTNAME):
            ROOT_DIR = Path("/work/pi_bseyedaghazadeh_umassd_edu")
            LOGS_DIR = Path.home() / "logs"
            logger.info(f"Cluster environment detected. ROOT_DIR set to: {ROOT_DIR}")
        else:
            flag = False
            ROOT_DIR = Path.home()
            logging.info(f"Local Host Path: {ROOT_DIR}")
            LOGS_DIR = Path.home() / "logs"
    except Exception as e:
        logging.error(f"Failed to get config: {e}")
        raise
    # Retrieve the current username from the home directory (last part of the path)
    if flag:
        USERNAME = Path.home().parts[-1]
        logger.info(f"USERNAME detected: {USERNAME}")

        # Set the home directory based on the retrieved username
        HOME_DIR = Path(f"/home/{USERNAME}")  # Home directory path
        logger.info(f"HOME_DIR set to: {HOME_DIR}")

        # Define the input data directory (relative to the root directory)
        INPUT_DIR = ROOT_DIR / "DL-PTV.backup"  # Input directory for backup files
        logger.info(f"INPUT_DIR set to: {INPUT_DIR}")

        # Define the base directory for output data
        OUTPUT_BASE_DIR = ROOT_DIR / "DL-PTV/roshni"  # Output base directory for processed data
        logger.info(f"OUTPUT_BASE_DIR set to: {OUTPUT_BASE_DIR}")

        # Path to the log file for raw data processing
        LOG_FILE_RAW_DATA_PROCESSOR = LOGS_DIR / "Raw_Data_Process/raw_data_process.log"  # Raw data processing log file
        logger.info(f"LOG_FILE set to: {LOG_FILE_RAW_DATA_PROCESSOR}")

        # Path to the log file for velocity loading process
        LOAD_VELOCITIES_LOG_FILE = LOGS_DIR / "loadVelocities/load_velocities.log"  # Velocity logs
        logger.info(f"LOAD_VELOCITIES_LOG_FILE set to: {LOAD_VELOCITIES_LOG_FILE}")

        # Path to the metadata configuration file for experiments
        METADATA_PATH = HOME_DIR / "cgan_deployment/configs/Experiment_MetaData.txt"  # Experiment metadata path
        logger.info(f"METADATA_PATH set to: {METADATA_PATH}")
    else: #if cluster not detected

        USERNAME = Path.home().parts[-1]
        logger.info(f"USERNAME detected: {USERNAME}")

        # Set the home directory based on the retrieved username
        HOME_DIR = Path(f"/home/{USERNAME}")  # Home directory path
        logger.info(f"HOME_DIR set to: {HOME_DIR}")

        # Define the input data directory (relative to the root directory)
        INPUT_DIR = "data/DL-PTV.backup"  # Input directory for backup files
        logger.info(f"INPUT_DIR set to: {INPUT_DIR}")

        # Define the base directory for output data
        os.makedirs("data/DL-PTV", exist_ok=True)
        OUTPUT_BASE_DIR = "data/DL-PTV"
        logger.info(f"OUTPUT_BASE_DIR set to: {OUTPUT_BASE_DIR}")

        # Path to the log file for raw data processing
        os.makedirs("../../logs/Raw_Data_Process/raw_data_process.log", exist_ok=True)
        LOG_FILE_RAW_DATA_PROCESSOR = "../../logs/Raw_Data_Process/raw_data_process.log"  # Raw data processing log file
        logger.info(f"LOG_FILE set to: {LOG_FILE_RAW_DATA_PROCESSOR}")

        # Path to the log file for velocity loading process
        os.makedirs("../../logs/loadVelocities/load_velocities.log", exist_ok=True)
        LOAD_VELOCITIES_LOG_FILE = "../../logs/loadVelocities/load_velocities.log"  # Velocity logs
        logger.info(f"LOAD_VELOCITIES_LOG_FILE set to: {LOAD_VELOCITIES_LOG_FILE}")

        # Path to the metadata configuration file for experiments
        METADATA_PATH = "configs/cgan_deployment/configs/Experiment_MetaData.txt"  # Experiment metadata path
        logger.info(f"METADATA_PATH set to: {METADATA_PATH}")

    # Dictionary specifying data types for each column
    COLUMN_DTYPES = {
        "x": "int32",
        "y": "int32",
        "z": "int32",
        "time": "int32",
        "distance": "int32",
    }

    # Dictionary defining experiment configurations
    EXPERIMENTS = {
        "10p4": {
            "input_file": INPUT_DIR / "10p4.pkl",
            "output_dir": OUTPUT_BASE_DIR / "10p4",
        }
    }

    # List specifying columns to drop during data processing
    COLUMNS_TO_DROP = ["px", "py", "pz"]
    # Options for data transformation during processing
    TRANSFORM_OPTIONS = ["float32", "linear", "tensor"]
    logger.info(f"TRANSFORM_OPTIONS set to: {TRANSFORM_OPTIONS}")






