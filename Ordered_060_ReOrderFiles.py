'''
This code is only necessary to fix the apparently "random" way that files were created when split.
What was intended was that the "time" would match the filename, so 2.pkl would contain the information
for time=2 for all eligible coordinates. What we did instead was split the 1200 times into 1200 files...
just named in the order in which they were discovered! To that end, we will rename files twice...first appending the
actual number...so if we have 1.pkl (which might have time=273) we will rename this:
1_nee_273.pkl
In this way in the next step (a zsh script) we can simply rename all the files
and get back to where we need to be.
Last but not least, we will write some unit tests to verify that we have the setup we envisioned.
'''

import os
import pandas as pd
import logging
import psutil
import time  # Keep time for potential debugging/performance logging if needed

# --- Configuration ---
# !!! PLEASE SET THIS TO YOUR DIRECTORY CONTAINING THE .pkl FILES !!!
TARGET_DIRECTORY = "/Users/kkreth/PycharmProjects/data/all_data_ready_for_training/"
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
        if DEBUG_MODE:
            logger.error(f"File not found: {file_path}")
        return None

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.debug(f"File size: {file_size_mb:.2f} MB")
    except OSError as e:
        if DEBUG_MODE:
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
                if compression:
                    df = pd.read_pickle(f, compression=compression)
                else:
                    df = pd.read_pickle(f)

            duration = time.time() - start_time
            logger.debug(f"Successfully read pickle file with compression: {compression}")
            logger.debug(f"Time taken to read: {duration:.2f} seconds")
            if isinstance(df, pd.DataFrame):
                logger.debug(f"DataFrame shape: {df.shape}")
            else:
                if DEBUG_MODE:
                    logger.warning(
                        f"Read object from {file_path} is not a DataFrame (type: {type(df)}). Skipping shape.")
            _log_memory_usage()
            return df

        except ModuleNotFoundError as e:
            if DEBUG_MODE:
                logger.error(
                    f"Error reading {file_path} with compression {compression}: Missing module {e}. The pickle file might require a specific library installed.")
            return None
        except Exception as e:
            logger.debug(f"Failed to read {file_path} with compression {compression}: {str(e)}")
            if compression == compressions[-1]:
                if DEBUG_MODE:
                    logger.error(f"Could not read file {file_path} after trying compressions {compressions}: {str(e)}")
                return None
            continue

    if DEBUG_MODE:
        logger.error(f"Failed to read {file_path}. No suitable compression method found or other error occurred.")
    return None


def process_pickle_files_in_directory(directory_path):
    """
    Scans a directory recursively for .pkl files, reads them, extracts the 'time'
    value from the first row, and renames the file accordingly.
    Logs operations only if DEBUG_MODE is True.

    Args:
        directory_path (str): The path to the directory to scan.
    """
    if not os.path.isdir(directory_path):
        if DEBUG_MODE:
            logger.error(f"Provided path is not a valid directory: {directory_path}")
        return

    if DEBUG_MODE:
        logger.info(f"Scanning directory recursively: {directory_path}")

    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(".pkl"):
                file_path = os.path.join(root, filename)
                if DEBUG_MODE:
                    logger.info(f"Processing file: {file_path}")

                df = read_pickle_file(file_path)

                if df is None:
                    if DEBUG_MODE:
                        logger.warning(f"Could not read or process DataFrame from {file_path}. Skipping.")
                    continue

                if not isinstance(df, pd.DataFrame):
                    if DEBUG_MODE:
                        logger.warning(
                            f"Object read from {file_path} is not a Pandas DataFrame (type: {type(df)}). Skipping.")
                    continue

                if df.empty:
                    if DEBUG_MODE:
                        logger.warning(f"DataFrame in {file_path} is empty. Skipping.")
                    continue

                if 'time' not in df.columns:
                    if DEBUG_MODE:
                        logger.warning(f"Column 'time' not found in {file_path}. Skipping.")
                    continue

                try:
                    first_time_value = df.iloc[0]['time']
                    base_filename_no_ext, ext = os.path.splitext(filename)

                    # Ensure first_time_value is a string for the filename
                    time_str = str(first_time_value)

                    new_filename = f"{base_filename_no_ext}_nee_{time_str}{ext}"
                    new_file_path = os.path.join(root, new_filename)

                    try:
                        os.rename(file_path, new_file_path)
                        if DEBUG_MODE:
                            logger.info(f"Successfully renamed '{file_path}' to '{new_file_path}'")
                    except OSError as e_rename:
                        if DEBUG_MODE:
                            logger.error(f"Failed to rename '{file_path}' to '{new_file_path}': {e_rename}")

                except IndexError:
                    if DEBUG_MODE:
                        logger.warning(f"Could not access first row (index 0) in {file_path}. Skipping rename.")
                except KeyError:
                    if DEBUG_MODE:
                        logger.warning(f"Column 'time' disappeared unexpectedly in {file_path}. Skipping rename.")
                except Exception as e:
                    if DEBUG_MODE:
                        logger.error(f"An unexpected error occurred while processing file {file_path} for rename: {e}")


if __name__ == "__main__":
    if TARGET_DIRECTORY == "/path/to/your/pickle_files":
        print("=" * 50)
        print("!!! IMPORTANT !!!")
        print("Please update the 'TARGET_DIRECTORY' variable in the script")
        print("to point to the directory containing your .pkl files.")
        print("=" * 50)
    else:
        process_pickle_files_in_directory(TARGET_DIRECTORY)
        if DEBUG_MODE:
            print("\n--- File processing and renaming attempt complete ---")
            print("Check logs for details if DEBUG_MODE was enabled and there were issues.")
        elif not os.path.isdir(TARGET_DIRECTORY):
            # This error is critical enough to show even if not in DEBUG_MODE
            logger.error(
                f"Error: The specified TARGET_DIRECTORY does not exist or is not a directory: {TARGET_DIRECTORY}")
