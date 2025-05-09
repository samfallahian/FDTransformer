import os
import pandas as pd
import logging
import psutil
import time  # Keep time for potential debugging/performance logging if needed

# --- Configuration ---
# !!! PLEASE SET THIS TO YOUR DIRECTORY CONTAINING THE .pkl FILES !!!
TARGET_DIRECTORY = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training/6p4"
# Set to True for more detailed logging, False for less verbose output
DEBUG_MODE = False
# --- End Configuration ---

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")


def read_pickle_file(file_path):
    """
    Read a pickle file with different compression methods.
    Adapted from PickleToCsvConverter.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        pandas.DataFrame or None: The dataframe from the pickle file or None if reading fails.
    """
    logger.debug(f"Starting to read pickle file: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f} MB")
    except OSError as e:
        logger.error(f"Could not get size for file {file_path}: {e}")
        return None

    # Try to determine compression method from file extension first
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.gz':
        compressions = ['gzip']
    elif file_ext == '.zip':
        compressions = ['zip']
    else:
        # Default to trying common compressions if extension is ambiguous (like .pkl)
        compressions = [None, 'gzip', 'zip']

    for compression in compressions:
        try:
            start_time = time.time()
            logger.debug(f"Trying to read with compression: {compression}")

            # Open the file and read it
            with open(file_path, 'rb') as f:
                # Use pd.read_pickle which handles compression automatically in newer versions
                # Specifying compression explicitly helps if auto-detect fails
                if compression:
                    df = pd.read_pickle(f, compression=compression)
                else:
                    df = pd.read_pickle(f)  # Try with no compression

            # Log performance information
            duration = time.time() - start_time
            logger.debug(f"Successfully read pickle file with compression: {compression}")
            logger.debug(f"Time taken to read: {duration:.2f} seconds")
            if isinstance(df, pd.DataFrame):
                logger.debug(f"DataFrame shape: {df.shape}")
            else:
                logger.warning(f"Read object is not a DataFrame (type: {type(df)}). Skipping shape.")
            _log_memory_usage()
            return df

        except ModuleNotFoundError as e:
            logger.error(
                f"Error reading {file_path} with compression {compression}: Missing module {e}. The pickle file might require a specific library installed.")
            return None  # Stop trying if a module is missing for this file
        except Exception as e:
            logger.debug(f"Failed to read {file_path} with compression {compression}: {str(e)}")
            # If this was the last attempt, log the final error
            if compression == compressions[-1]:
                logger.error(f"Could not read file {file_path} after trying compressions {compressions}: {str(e)}")
                return None
            # Otherwise, continue to try the next compression method
            continue

    # Should not be reached if compressions list is not empty, but added for safety
    logger.error(f"Failed to read {file_path}. No suitable compression method found or other error occurred.")
    return None


def scan_and_extract_times(directory_path):
    """
    Scans a directory for .pkl files, reads them, and extracts the 'time'
    value from the first row.

    Args:
        directory_path (str): The path to the directory to scan.

    Returns:
        list: A list of dictionaries, each containing 'filename' (base name)
              and 'first_time_value'. Returns an empty list if no valid files
              or data are found.
    """
    extracted_data = []

    if not os.path.isdir(directory_path):
        logger.error(f"Provided path is not a valid directory: {directory_path}")
        return extracted_data

    logger.info(f"Scanning directory: {directory_path}")

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pkl"):
            file_path = os.path.join(directory_path, filename)
            logger.info(f"Processing file: {filename}")

            df = read_pickle_file(file_path)

            if df is None:
                logger.warning(f"Could not read or process DataFrame from {filename}. Skipping.")
                continue

            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Object read from {filename} is not a Pandas DataFrame (type: {type(df)}). Skipping.")
                continue

            if df.empty:
                logger.warning(f"DataFrame in {filename} is empty. Skipping.")
                continue

            if 'time' not in df.columns:
                logger.warning(f"Column 'time' not found in {filename}. Skipping.")
                continue

            try:
                first_time_value = df.iloc[0]['time']
                # Extract base filename (e.g., "1" from "1.pkl")
                base_filename = os.path.splitext(filename)[0]

                extracted_data.append({
                    'filename': base_filename,
                    'first_time_value': first_time_value
                })
                logger.info(f"Successfully extracted time '{first_time_value}' from {filename}")

            except IndexError:
                # This case should be caught by df.empty check, but added for safety
                logger.warning(f"Could not access first row (index 0) in {filename}. Skipping.")
            except KeyError:
                # This case should be caught by 'time' in df.columns check
                logger.warning(f"Column 'time' disappeared unexpectedly in {filename}. Skipping.")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing the first row of {filename}: {e}")

    return extracted_data


if __name__ == "__main__":

    # --- Basic Check ---
    if TARGET_DIRECTORY == "/path/to/your/pickle_files":
        print("=" * 50)
        print("!!! IMPORTANT !!!")
        print("Please update the 'TARGET_DIRECTORY' variable in the script")
        print("to point to the directory containing your .pkl files.")
        print("=" * 50)
    else:
        results = scan_and_extract_times(TARGET_DIRECTORY)

        if results:
            print("\n--- Extracted First Time Values ---")
            # Sort results numerically by filename if possible, otherwise alphabetically
            try:
                results.sort(key=lambda x: int(x['filename']))
                print("(Sorted numerically by filename)")
            except ValueError:
                results.sort(key=lambda x: x['filename'])
                print("(Sorted alphabetically by filename)")

            print(f"{'Filename':<15} {'First Time Value'}")
            print("-" * 35)
            for item in results:
                print(f"{item['filename']:<15} {item['first_time_value']}")
        else:
            print("\nNo data was extracted. Check logs for details.")
            if not os.path.isdir(TARGET_DIRECTORY):
                print(f"Error: The specified directory does not exist or is not a directory: {TARGET_DIRECTORY}")