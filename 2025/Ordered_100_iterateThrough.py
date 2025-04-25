import os
import sys
import pandas as pd
import numpy as np
import time
import json
import logging
from typing import List, Tuple, Dict, Optional

from Ordered_001_Initialize import HostPreferences
from CoordinateSpace import givenXYZreplyVelocityCube

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor(HostPreferences):
    """
    Class to process a single pickle file, extract data based on coordinates,
    and connect with original data for CoordinateSpace processing.
    """

    def __init__(self, filename="experiment.preferences", input_path=None):
        """Initialize with preferences file and input path"""
        start_time = time.time()
        super().__init__(filename)
        
        if input_path is None:
            if len(sys.argv) > 1:
                input_path = sys.argv[1]
            else:
                raise ValueError("Input file path must be specified either as an argument or via command line")
        
        self.input_path = input_path
        logger.info(f"Processing input file: {self.input_path}")
        
        # Extract experiment name from the second-to-last directory in the path
        path_parts = os.path.normpath(input_path).split(os.sep)
        if len(path_parts) < 2:
            raise ValueError(f"Input path doesn't have enough components to extract experiment name: {input_path}")
        
        # Extract experiment name from the second-to-last directory
        self.experiment_name = path_parts[-2]
        logger.info(f"Extracted experiment name: {self.experiment_name}")
        
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
                logger.info(f"Loaded metadata from {self.metadata_location}")
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
                logger.setLevel(level)
                logger.debug(f"Set logging level to {self.logging_level.upper()}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: Initialization time: {time.time() - start_time:.4f}s")

    def load_pickle_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a pickle file with intelligent detection of compression method.
        Handles cross-platform differences between macOS and Linux pickle files.
        
        Args:
            file_path: Path to the pickle file
        
        Returns:
            Pandas DataFrame or None if loading fails
        """
        # First, examine the file header to determine if it's compressed
        try:
            with open(file_path, 'rb') as f:
                # Read first few bytes to check the file signature
                header = f.read(2)
                f.seek(0)  # Reset file position
                
                # Standard gzip files start with magic number b'\x1f\x8b'
                if header == b'\x1f\x8b':
                    # File has gzip header
                    compression = 'gzip'
                    logger.debug(f"Detected gzip compression for {file_path} based on header")
                elif header.startswith(b'PK'):
                    # ZIP files start with 'PK' signature
                    compression = 'zip'
                    logger.debug(f"Detected zip compression for {file_path} based on header")
                else:
                    # File appears to be a raw pickle file (not compressed)
                    # b'\x80\x05' is the signature for pickle protocol 5
                    compression = None
                    logger.debug(f"Detected uncompressed pickle file {file_path} with header: {header}")
                
            # First try with the detected compression method
            start_time = time.time()
            logger.info(f"Loading {file_path} with {compression or 'no'} compression")
            df = pd.read_pickle(file_path, compression=compression)
            logger.debug(f"Successfully loaded {file_path} in {time.time() - start_time:.2f}s")
            return df
        
        except Exception as e:
            # If auto-detection fails, try all possible compression methods as fallback
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
                    # Log the specific error for debugging
                    logger.debug(f"Failed loading with {comp or 'no'} compression: {str(retry_e)}")
                    
                    # Only log as error when we've exhausted all options
                    if comp == 'zip':  # Last option in our list
                        logger.error(f"Failed to load {file_path} with any compression method: {str(retry_e)}")
            
            # If we've tried all methods and none worked
            return None

    def get_original_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the original dataframe from the root path based on experiment name.
        """
        if self.original_df is not None:
            return self.original_df

        # Use experiment name to locate the original file
        original_filename = f"{self.experiment_name}.pkl"
        original_path = os.path.join(self.root_path, original_filename)
        
        if not os.path.exists(original_path):
            logger.error(f"Original file not found: {original_path}")
            return None

        df = self.load_pickle_file(original_path)
        if df is not None:
            self.original_df = df
            logger.info(f"Loaded original data from {original_path}")

        return self.original_df

    def process_row(self, row: pd.Series) -> Dict:
        """
        Process a single row from the dataframe.

        Args:
            row: DataFrame row with time, x, y, z columns

        Returns:
            Dictionary with processed data
        """
        try:
            # Get original dataframe
            original_df = self.get_original_dataframe()
            if original_df is None:
                raise RuntimeError("Failed to load original dataframe")

            # Extract time and coordinates (ensuring they are integers)
            time_val = int(row['time'])
            x_val = int(row['x'])
            y_val = int(row['y'])
            z_val = int(row['z'])

            # Initialize coordinate space processor
            coord_processor = givenXYZreplyVelocityCube(
                pickle_filename=f"{self.experiment_name}.pkl",
                x=x_val,
                y=y_val,
                z=z_val
            )

            # Get neighboring coordinates
            coordinates = coord_processor.locateNeighbors()

            # Filter original dataframe by time
            time_filtered_df = original_df[original_df['time'] == time_val]

            # Extract velocity data for each coordinate
            velocity_data = []
            for coord in coordinates:
                cx, cy, cz = coord
                # Find matching row in original dataframe
                match = time_filtered_df[(time_filtered_df['x'] == cx) &
                                         (time_filtered_df['y'] == cy) &
                                         (time_filtered_df['z'] == cz)]

                if len(match) == 1:
                    # Extract velocity components
                    vx = float(match['vx'].iloc[0])
                    vy = float(match['vy'].iloc[0])
                    vz = float(match['vz'].iloc[0])
                    velocity_data.append((vx, vy, vz))
                else:
                    # Log as critical error instead of warning
                    logger.critical(f"Found {len(match)} matches for coordinate {coord} at time {time_val}")
                    # Exit the program with error code 1
                    sys.exit(1)

            # Return processed data
            return {
                'time': time_val,
                'x': x_val,
                'y': y_val,
                'z': z_val,
                'coordinates': coordinates,
                'velocities': velocity_data
            }

        except Exception as e:
            logger.error(f"Error processing row: {str(e)}")
            return {
                'time': row.get('time', None),
                'x': row.get('x', None),
                'y': row.get('y', None),
                'z': row.get('z', None),
                'error': str(e)
            }

    def update_output_with_velocities(self, processed_data: List[Dict]) -> bool:
        """
        Update the input dataframe with processed velocity data.
        Uses batch updates instead of individual assignments.
        """
        update_start_time = time.time()
        try:
            # Load dataframe from input path
            load_start = time.time()
            df = self.load_pickle_file(self.input_path)
            logger.debug(f"Load time for update: {time.time() - load_start:.4f}s")
        
            if df is None:
                logger.error(f"Failed to load input file for updating: {self.input_path}")
                return False

            # Update rows in batches
            update_rows_start = time.time()
            rows_updated = 0
        
            # Prepare a list of updates to be applied at once
            for data in processed_data:
                # Ensure we have valid data
                if 'error' in data or not data.get('velocities'):
                    continue

                # Find the corresponding row in the dataframe
                time_val = data['time']
                x_val = data['x']
                y_val = data['y']
                z_val = data['z']

                # Create a boolean mask to find the row
                mask = ((df['time'] == time_val) &
                        (df['x'] == x_val) &
                        (df['y'] == y_val) &
                        (df['z'] == z_val))

                if mask.sum() != 1:
                    logger.warning(f"Found {mask.sum()} rows matching time={time_val}, x={x_val}, y={y_val}, z={z_val}")
                    continue

                # Get index of the row
                idx = df[mask].index[0]
            
                # Create a dictionary of all updates for this row
                updates = {}
                velocities = data['velocities']
                for i, (vx, vy, vz) in enumerate(velocities, 1):
                    if i <= 125:  # Ensure we don't exceed column count
                        updates[f'vx_{i}'] = vx
                        updates[f'vy_{i}'] = vy
                        updates[f'vz_{i}'] = vz
            
                # Apply all updates for this row at once
                df.loc[idx, list(updates.keys())] = list(updates.values())
                rows_updated += 1
            
            logger.debug(f"Total rows update time ({rows_updated} rows): {time.time() - update_rows_start:.4f}s")

            # Save updated dataframe
            save_start = time.time()
            df.to_pickle(self.input_path, compression='gzip')
            logger.debug(f"Save time for updated dataframe: {time.time() - save_start:.4f}s")
        
            logger.info(f"Updated {self.input_path} with velocity data for {rows_updated} rows")
            logger.debug(f"Total update time: {time.time() - update_start_time:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating input file: {str(e)}")
            logger.debug(f"Update failed after {time.time() - update_start_time:.4f}s")
            return False

    def run(self, max_rows: int = None):
        """
        Run the processing on the input file.

        Args:
            max_rows: Maximum number of rows to process (for debugging)
        """
        start_time = time.time()

        try:
            # Load the input file
            df = self.load_pickle_file(self.input_path)
            if df is None:
                logger.error(f"Failed to load input file: {self.input_path}")
                return

            # Limit number of rows if requested
            if max_rows is not None:
                df = df.head(max_rows)
                
            row_count = len(df)
            logger.info(f"Processing {row_count} rows from {self.input_path}")

            # Process each row
            processed_data = []
            for i, (_, row) in enumerate(df.iterrows()):
                if i % 10 == 0:  # Log progress periodically
                    logger.info(f"Processing row {i + 1}/{row_count}")

                result = self.process_row(row)
                processed_data.append(result)

            # Update input file with results
            if processed_data:
                success = self.update_output_with_velocities(processed_data)
                if success:
                    logger.info(f"Successfully updated {self.input_path}")
                else:
                    logger.error(f"Failed to update {self.input_path}")

            elapsed = time.time() - start_time
            logger.info(f"Completed processing {row_count} rows in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        # Create processor instance with input path from command line if provided
        input_path = sys.argv[1] if len(sys.argv) > 1 else None
        processor = DataProcessor(input_path=input_path)
        
        # For testing, process only a limited number of rows
        # processor.run(max_rows=500)
        
        # For full processing
        processor.run()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise