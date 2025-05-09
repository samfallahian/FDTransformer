import os
import sys
import pandas as pd
import numpy as np
import time
import json
import logging
from typing import List, Tuple, Dict, Optional

# Assuming these imports exist and are correct
from Ordered_001_Initialize import HostPreferences
from CoordinateSpace import givenXYZreplyVelocityCube # <--- Uncomment this line

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor(HostPreferences):
    """
    Class to process a single pickle file (representing a specific time step derived
    from its filename), extract data based on coordinates found within,
    and connect with original data from a master file for velocity processing.
    """

    def __init__(self, filename="experiment.preferences", input_path=None):
        """Initialize with preferences file and input path"""
        init_start_time = time.time()
        super().__init__(filename)

        if input_path is None:
            if len(sys.argv) > 1:
                input_path = sys.argv[1]
            else:
                raise ValueError("Input file path must be specified either as an argument or via command line")

        self.input_path = input_path
        logger.info(f"Processing input file: {self.input_path}")

        # --- New: Extract time value from filename ---
        try:
            base_filename = os.path.basename(self.input_path)
            time_str = os.path.splitext(base_filename)[0] # Get filename without extension
            self.file_time_val = int(time_str)
            logger.info(f"Derived time value '{self.file_time_val}' from filename '{base_filename}'")
        except ValueError:
            logger.error(f"Could not extract integer time value from filename: {base_filename}")
            raise ValueError(f"Filename '{base_filename}' does not yield a valid integer time.")
        except Exception as e:
            logger.error(f"Error extracting time from filename {self.input_path}: {e}")
            raise
        # --- End New ---

        # Extract experiment name from the second-to-last directory in the path
        path_parts = os.path.normpath(input_path).split(os.sep)
        if len(path_parts) < 2:
            raise ValueError(f"Input path doesn't have enough components to extract experiment name: {input_path}")

        # Extract experiment name (assumed to be the directory containing the time-step file)
        self.experiment_name = path_parts[-2]
        logger.info(f"Extracted experiment name: {self.experiment_name}")
        logger.debug(f"DEBUG: Initialized with input_path='{self.input_path}', experiment_name='{self.experiment_name}', file_time_val={self.file_time_val}")

        # Validate required paths
        if not hasattr(self, 'root_path') or not self.root_path:
            raise ValueError("Root path not specified in preferences")
        if not hasattr(self, 'metadata_location') or not self.metadata_location:
            raise ValueError("Metadata location not specified in preferences")

        # Load metadata
        metadata_load_start = time.time()
        if os.path.exists(self.metadata_location):
            try:
                with open(self.metadata_location, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata from {self.metadata_location} (took {time.time() - metadata_load_start:.4f}s)")
            except Exception as e:
                logger.error(f"Failed to load metadata: {str(e)}")
                raise
        else:
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_location}")

        # Cache for original dataframe
        self.original_df = None

        # Set the logging level from preferences if available
        if hasattr(self, 'logging_level'):
            level = getattr(logging, self.logging_level.upper(), None)
            if level is not None:
                effective_level = min(level, logger.level) if logger.level != logging.NOTSET else level
                logger.setLevel(effective_level)
                if effective_level <= logging.DEBUG:
                    logger.debug(f"DEBUG: Logging level set to DEBUG (or lower) based on preferences.")
                else:
                     logger.info(f"Set logging level to {logging.getLevelName(effective_level)}")


        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: Initialization time: {time.time() - init_start_time:.4f}s")

    def load_pickle_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a pickle file with intelligent detection of compression method.
        Handles cross-platform differences between macOS and Linux pickle files.

        Args:
            file_path: Path to the pickle file

        Returns:
            Pandas DataFrame or None if loading fails
        """
        # --- No changes needed in this method ---
        # First, examine the file header to determine if it's compressed
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check the file signature
                header = f.read(2)
                f.seek(0)  # Reset file position

                # Standard gzip files start with magic number b'\x1f\x8b'
                if header == b'\x1f\x8b':
                    compression = 'gzip'
                    logger.debug(f"Detected gzip compression for {file_path} based on header")
                elif header.startswith(b'PK'):
                    compression = 'zip'
                    logger.debug(f"Detected zip compression for {file_path} based on header")
                else:
                    compression = None
                    logger.debug(f"Detected uncompressed pickle file {file_path} with header: {header}")

            # First try with the detected compression method
            start_time = time.time()
            logger.info(f"Loading {file_path} with {compression or 'no'} compression")
            df = pd.read_pickle(file_path, compression=compression)
            logger.debug(f"Successfully loaded {file_path} in {time.time() - start_time:.2f}s")
            return df

        except Exception as e:
            logger.warning(f"Auto-detection failed for {file_path}: {str(e)}")
            logger.info("Trying all compression methods as fallback")

            for comp in [None, 'gzip', 'zip']:
                try:
                    start_time = time.time()
                    logger.debug(f"Trying to load {file_path} with {comp or 'no'} compression")
                    df = pd.read_pickle(file_path, compression=comp)
                    logger.info(
                        f"Successfully loaded {file_path} with {comp or 'no'} compression in {time.time() - start_time:.2f}s")
                    return df
                except Exception as retry_e:
                    logger.debug(f"Failed loading with {comp or 'no'} compression: {str(retry_e)}")
                    if comp == 'zip':
                        logger.error(f"Failed to load {file_path} with any compression method: {str(retry_e)}")
            return None

    def get_original_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the original dataframe from the root path based on experiment name.
        """
        # --- No changes needed in this method, but added timing log ---
        if self.original_df is not None:
            logger.debug("DEBUG: Returning cached original dataframe.")
            return self.original_df

        original_filename = f"{self.experiment_name}.pkl"
        original_path = os.path.join(self.root_path, original_filename)
        logger.debug(f"DEBUG: Attempting to load original dataframe from path: {original_path}")

        if not os.path.exists(original_path):
            logger.error(f"Original file not found: {original_path}")
            return None

        load_start = time.time()
        df = self.load_pickle_file(original_path)
        load_time = time.time() - load_start
        if df is not None:
            self.original_df = df
            logger.info(f"Loaded original data from {original_path} (took {load_time:.4f}s)")
            logger.debug(f"DEBUG: Original dataframe '{original_filename}' loaded successfully. Shape: {df.shape}")
        else:
             logger.error(f"Failed to load original dataframe from {original_path} after {load_time:.4f}s")


        return self.original_df

    def process_row(self, row: pd.Series) -> Dict:
        """
        Process a single row from the input dataframe. Uses the time value
        derived from the input filename (`self.file_time_val`) to query the
        original master dataframe.

        Args:
            row: DataFrame row with at least x, y, z columns

        Returns:
            Dictionary with processed data (coordinates and velocities)
        """
        # --- Define time_val_from_file EARLIER using the instance variable ---
        time_val_from_file = self.file_time_val # Use the time derived from the filename

        try:
            # Get original dataframe
            original_df = self.get_original_dataframe()
            if original_df is None:
                raise RuntimeError("Failed to load original dataframe")

            # Log initial row count
            initial_row_count = len(original_df)
            logger.info(f"Original dataframe row count before filtering: {initial_row_count}")

            # Log the filter being applied (NOW CORRECTLY USES time_val_from_file)
            logger.info(f"Filtering original dataframe by time value: {time_val_from_file}")

            # Apply the filter (NOW CORRECTLY USES time_val_from_file)
            # Assuming the time column in the original dataframe is named 'time'
            original_df = original_df[original_df['time'] == time_val_from_file].copy() # Added .copy() to avoid SettingWithCopyWarning

            # Log final row count
            final_row_count = len(original_df)
            logger.info(f"Original dataframe row count after filtering: {final_row_count}")

            # Add a check in case the filtering resulted in an empty dataframe
            if original_df.empty:
                logger.warning(f"Filtering by time {time_val_from_file} resulted in an empty dataframe.")
                # Depending on the desired behavior, you might want to raise an error or return early here
                # For now, we'll just log a warning and continue, but this might need adjustment.

            # --- Continue with the rest of the processing using the filtered original_df ---

            # --- Modified: Use self.file_time_val instead of row['time'] ---
            # time_val = int(row['time']) # REMOVED
            # time_val_from_file = self.file_time_val # REMOVED FROM HERE

            # Extract coordinates (ensuring they are integers)
            # Check if columns exist before accessing
            if not all(col in row for col in ['x', 'y', 'z']):
                missing_cols = [col for col in ['x', 'y', 'z'] if col not in row]
                logger.error(f"Input row missing required coordinate columns: {missing_cols}")
                # Decide how to handle: skip row, raise error, return partial? Returning error for now.
                return {'error': f"Missing columns: {missing_cols}"}

            x_val = int(row['x'])
            y_val = int(row['y'])
            z_val = int(row['z'])
            logger.debug(f"DEBUG: Processing input row coords x={x_val}, y={y_val}, z={z_val} using file_time={time_val_from_file}")
            # --- End Modified ---

            # Initialize coordinate space processor
            coord_processor = givenXYZreplyVelocityCube(
                pickle_filename=f"{self.experiment_name}.pkl", # Still uses experiment name for context if needed by the class
                x=x_val,
                y=y_val,
                z=z_val
            )

            # Get neighboring coordinates
            coordinates = coord_processor.locateNeighbors() # This likely depends only on x,y,z, not time

            # --- Modified: Filter original dataframe by time_val_from_file (used the filtered one now) ---
            # No need to filter again, original_df is already filtered
            time_filtered_df = original_df # Use the already filtered DataFrame
            logger.debug(f"DEBUG: Using pre-filtered original_df with {len(time_filtered_df)} rows for file_time={time_val_from_file}")
            # --- End Modified ---

            if time_filtered_df.empty:
                 logger.warning(f"No data found in original dataframe for file_time={time_val_from_file} after filtering. Cannot find velocities.")
                 # Depending on requirements, might return error or empty velocities
                 return {
                    'x': x_val, 'y': y_val, 'z': z_val,
                    'coordinates': coordinates, 'velocities': [], # Return empty list
                    'warning': f'No original data for time {time_val_from_file}'
                 }

            # Extract velocity data for each neighbor coordinate from the time-filtered original data
            velocity_data = []
            for coord in coordinates:
                cx, cy, cz = coord
                logger.debug(f"DEBUG: Searching time_filtered_df for neighbor coord ({cx},{cy},{cz}) at file_time {time_val_from_file}")
                match = time_filtered_df[(time_filtered_df['x'] == cx) &
                                         (time_filtered_df['y'] == cy) &
                                         (time_filtered_df['z'] == cz)]

                if len(match) == 1:
                    vx = float(match['vx'].iloc[0])
                    vy = float(match['vy'].iloc[0])
                    vz = float(match['vz'].iloc[0])
                    velocity_data.append((vx, vy, vz))
                    logger.debug(f"DEBUG: Found match for coord ({cx},{cy},{cz}). Velocities: ({vx:.4f}, {vy:.4f}, {vz:.4f})")
                elif len(match) == 0:
                     logger.warning(f"No match found for neighbor coordinate {coord} at file_time {time_val_from_file} in original_df.")
                     # Handle missing neighbor: append NaN? Skip? Append (0,0,0)? Appending NaN for clarity.
                     velocity_data.append((np.nan, np.nan, np.nan))
                else: # len(match) > 1
                    logger.error(f"Found {len(match)} matches for neighbor coordinate {coord} at file_time {time_val_from_file}. This indicates duplicate data in the original file.")
                    # Handle duplicate neighbor: take first? average? error? Taking first for now, but logging error.
                    vx = float(match['vx'].iloc[0])
                    vy = float(match['vy'].iloc[0])
                    vz = float(match['vz'].iloc[0])
                    velocity_data.append((vx, vy, vz))
                    logger.debug(f"DEBUG: Used first match for duplicate coord ({cx},{cy},{cz}). Velocities: ({vx:.4f}, {vy:.4f}, {vz:.4f})")


            # Return processed data (x, y, z identify the row in the *input* file)
            return {
                'x': x_val,
                'y': y_val,
                'z': z_val,
                'coordinates': coordinates, # Neighbor coordinates
                'velocities': velocity_data # Velocities of neighbors found at file_time_val
            }

        except Exception as e:
            # Log error with coordinates if available
            x = row.get('x', 'N/A')
            y = row.get('y', 'N/A')
            z = row.get('z', 'N/A')
            # Use self.file_time_val here as local time_val_from_file might not be defined if error happened early
            logger.error(f"Error processing row (coords x={x}, y={y}, z={z}, file_time={self.file_time_val}): {str(e)}")
            # Include traceback for debugging
            logger.exception("Exception details:")
            return {
                'x': row.get('x'), 'y': row.get('y'), 'z': row.get('z'),
                'error': str(e)
            }

    def update_output_with_velocities(self, processed_data: List[Dict]) -> bool:
        """
        Update the input dataframe (loaded from `self.input_path`) with processed
        velocity data. Locates rows in the input dataframe based ONLY on x, y, z
        coordinates, as the 'time' context is implicitly the entire file
        (derived from `self.file_time_val`).
        """
        update_start_time = time.time()
        try:
            # Load dataframe from input path
            load_start = time.time()
            df = self.load_pickle_file(self.input_path)
            load_time = time.time() - load_start
            logger.debug(f"Load time for input file update: {load_time:.4f}s")

            if df is None:
                logger.error(f"Failed to load input file for updating: {self.input_path}")
                return False

            # Ensure velocity columns exist (create if needed), handle potential large number
            max_neighbors = 125 # Assuming based on previous code's vx_1..vx_125
            required_cols = []
            for i in range(1, max_neighbors + 1):
                required_cols.extend([f'vx_{i}', f'vy_{i}', f'vz_{i}'])

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Input dataframe missing velocity columns: {missing_cols}. Adding them with NaN.")
                for col in missing_cols:
                    # Determine dtype (float32 seems reasonable for velocities)
                    df[col] = np.nan
                    df[col] = df[col].astype(np.float32) # Or float64 if precision needed

            update_rows_start = time.time()
            rows_updated = 0
            rows_skipped_no_match = 0
            rows_skipped_duplicate_match = 0
            rows_with_errors = 0

            # Process updates
            for data in processed_data:
                if 'error' in data:
                    rows_with_errors += 1
                    # Log the specific error if needed, already logged in process_row
                    logger.debug(f"Skipping update for row with processing error: {data.get('error', 'Unknown error')}")
                    continue

                if not data.get('velocities'):
                    # This might happen if original data for file_time_val was missing or no neighbors found
                    logger.debug(f"Skipping update for x={data.get('x')}, y={data.get('y')}, z={data.get('z')} due to missing/empty velocity data.")
                    continue

                # --- Modified: Locate row in input df using ONLY x, y, z ---
                x_val = data['x']
                y_val = data['y']
                z_val = data['z']

                logger.debug(f"DEBUG: Preparing update for input row with x={x_val}, y={y_val}, z={z_val}")

                # Create a boolean mask using only coordinates
                mask = (df['x'] == x_val) & (df['y'] == y_val) & (df['z'] == z_val)
                # --- End Modified ---

                mask_sum = mask.sum()
                if mask_sum == 0:
                    logger.warning(f"Could not find unique row in input file {self.input_path} for x={x_val}, y={y_val}, z={z_val}. Skipping update.")
                    rows_skipped_no_match += 1
                    continue
                if mask_sum > 1:
                     logger.warning(f"Found {mask_sum} duplicate rows in input file {self.input_path} matching x={x_val}, y={y_val}, z={z_val}. Skipping update for this entry to avoid ambiguity.")
                     rows_skipped_duplicate_match += 1
                     continue

                # Get index of the unique row
                idx = df.index[mask][0]
                logger.debug(f"DEBUG: Found unique index {idx} in input df for x={x_val}, y={y_val}, z={z_val}")

                # Create dictionary of velocity updates for this row
                updates = {}
                velocities = data['velocities']
                num_neighbors_found = len(velocities)

                for i, (vx, vy, vz) in enumerate(velocities, 1):
                    if i > max_neighbors:
                        logger.warning(f"More than {max_neighbors} velocities found ({num_neighbors_found}) for row index {idx}. Truncating.")
                        break
                    # Assign even if NaN (e.g., if neighbor wasn't found in original_df)
                    updates[f'vx_{i}'] = vx
                    updates[f'vy_{i}'] = vy
                    updates[f'vz_{i}'] = vz

                # Fill remaining columns up to max_neighbors with NaN if fewer neighbors were found/processed
                for i in range(num_neighbors_found + 1, max_neighbors + 1):
                     updates[f'vx_{i}'] = np.nan
                     updates[f'vy_{i}'] = np.nan
                     updates[f'vz_{i}'] = np.nan


                # Apply all updates for this row
                if updates: # Check if there's anything to update
                    logger.debug(f"DEBUG: Applying {len(updates)//3} sets of (vx, vy, vz) updates to index {idx}")
                    # Ensure dtypes match before assignment if strict
                    update_keys = list(updates.keys())
                    update_values = [np.float32(v) if not pd.isna(v) else np.nan for v in updates.values()] # Cast to float32
                    try:
                        df.loc[idx, update_keys] = update_values
                        rows_updated += 1
                    except Exception as update_err:
                         logger.error(f"Failed to apply update to index {idx} for x={x_val}, y={y_val}, z={z_val}: {update_err}")
                         # Potentially increment another counter here
                else:
                    logger.debug(f"DEBUG: No velocity updates to apply for index {idx}")


            update_duration = time.time() - update_rows_start
            logger.info(f"Finished preparing updates. Rows updated: {rows_updated}, Skipped (no match): {rows_skipped_no_match}, Skipped (duplicates): {rows_skipped_duplicate_match}, Skipped (errors): {rows_with_errors}.")
            logger.debug(f"Total row update preparation time: {update_duration:.4f}s")

            # Save updated dataframe
            save_start = time.time()
            logger.debug(f"DEBUG: Saving updated dataframe to {self.input_path} (gzip compressed)")
            try:
                df.to_pickle(self.input_path, compression='gzip')
                save_time = time.time() - save_start
                logger.debug(f"Save time for updated dataframe: {save_time:.4f}s")
                logger.info(f"Successfully saved updated data to {self.input_path}")
            except Exception as save_err:
                 logger.error(f"Failed to save updated dataframe to {self.input_path}: {save_err}")
                 return False


            total_update_method_time = time.time() - update_start_time
            logger.info(f"Update process completed for {self.input_path} in {total_update_method_time:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Critical error during update_output_with_velocities for {self.input_path}: {str(e)}")
            logger.exception("Exception details:")
            logger.debug(f"Update failed after {time.time() - update_start_time:.4f}s")
            return False

    def run(self, max_rows: int = None):
        """
        Run the processing on the input file. Derives time from the filename.

        Args:
            max_rows: Maximum number of rows from the input file to process (for debugging)
        """
        run_start_time = time.time()

        try:
            logger.debug(f"DEBUG: Starting run method for {self.input_path} (file_time={self.file_time_val})")
            # Load the input file
            df = self.load_pickle_file(self.input_path)
            if df is None:
                logger.error(f"Failed to load input file: {self.input_path}, cannot run processing.")
                return

            # Pre-load original dataframe to avoid repeated loading in loop
            logger.info("Pre-loading original dataframe...")
            if self.get_original_dataframe() is None:
                 logger.error(f"Failed to load the original dataframe ({self.experiment_name}.pkl). Aborting run.")
                 return
            logger.info("Original dataframe loaded.")


            # Limit number of rows if requested
            if max_rows is not None and max_rows < len(df):
                df_to_process = df.head(max_rows).copy() # Use copy to avoid SettingWithCopyWarning if df is modified later
                logger.info(f"Limiting processing to first {max_rows} rows of {self.input_path}.")
            else:
                df_to_process = df
                max_rows = len(df) # For accurate logging below

            row_count = len(df_to_process)
            if row_count == 0:
                 logger.warning(f"Input file {self.input_path} is empty. Nothing to process.")
                 return

            logger.info(f"Processing {row_count} rows from {self.input_path} using file_time={self.file_time_val}")

            # Process each row
            process_loop_start = time.time()
            processed_data = []
            for i, (_, row) in enumerate(df_to_process.iterrows()):
                # Log progress less frequently now, maybe every 10% or 1000 rows
                if (i + 1) % max(1, row_count // 10) == 0 or i == 0:
                     logger.info(f"Processing input row {i + 1}/{row_count}...")

                result = self.process_row(row)
                processed_data.append(result)
            process_loop_time = time.time() - process_loop_start
            logger.info(f"Finished processing {row_count} rows in {process_loop_time:.2f}s.")


            # Update input file with results
            if processed_data:
                logger.info(f"Starting update process for {self.input_path} with {len(processed_data)} results.")
                success = self.update_output_with_velocities(processed_data)
                if success:
                    logger.info(f"Successfully processed and updated {self.input_path}")
                else:
                    logger.error(f"Failed to update {self.input_path} after processing.")
            else:
                logger.info("No data was processed successfully, skipping update step.")


            elapsed = time.time() - run_start_time
            logger.info(f"Completed run for {self.input_path} in {elapsed:.2f}s")
            logger.debug(f"DEBUG: Run method finished for {self.input_path}")

        except Exception as e:
            logger.error(f"Unhandled error during run for file {self.input_path}: {str(e)}")
            logger.exception("Exception details:")
            logger.debug(f"DEBUG: Run method failed with exception.")
            # Optionally re-raise or exit depending on desired behavior in batch processing
            # raise e # Uncomment to propagate error upwards


if __name__ == "__main__":
    main_start_time = time.time()
    try:
        # Create processor instance with input path from command line if provided
        input_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
        if not input_path_arg:
             logger.error("No input file path provided via command line argument.")
             sys.exit(1) # Exit if no input file given

        processor = DataProcessor(input_path=input_path_arg)

        # For full processing:
        processor.run()

        # Example for testing with limited rows:
        # logger.info("--- RUNNING WITH MAX_ROWS LIMIT FOR TESTING ---")
        # processor.run(max_rows=10)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
         logger.error(f"Value error (e.g., invalid filename or path format): {str(e)}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred in main execution: {str(e)}")
        logger.exception("Unhandled exception details:") # Log traceback
        sys.exit(1) # Exit with error status

    finally:
        logger.info(f"Total script execution time: {time.time() - main_start_time:.2f} seconds.")