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


class DataIterator(HostPreferences):
    """
    Class to iterate through pickle files in the output directory, process rows,
    and connect with original data for CoordinateSpace processing.
    """

    # Don't change the logging level directly
    # Instead, check if DEBUG level messages should be logged

    def __init__(self, filename="experiment.preferences"):
        """Initialize with preferences file"""
        start_time = time.time()
        super().__init__(filename)
        # Validate required paths
        if not hasattr(self, 'output_directory') or not self.output_directory:
            raise ValueError("Output directory not specified in preferences")
        if not hasattr(self, 'root_path') or not self.root_path:
            raise ValueError("Root path not specified in preferences")
        if not hasattr(self, 'metadata_location') or not self.metadata_location:
            raise ValueError("Metadata location not specified in preferences")

        # Ensure directories exist
        if not os.path.exists(self.output_directory):
            raise FileNotFoundError(f"Output directory not found: {self.output_directory}")
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Root path not found: {self.root_path}")

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
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: Metadata loading time: {time.time() - metadata_load_start:.4f}s")
        
        # Cache for original dataframes
        self.original_df_cache = {}
        
        # Set the logging level from preferences if available
        if hasattr(self, 'logging_level'):
            level = getattr(logging, self.logging_level.upper(), None)
            if level is not None:
                logger.setLevel(level)
                logger.debug(f"Set logging level to {self.logging_level.upper()}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DEBUG: Initialization time: {time.time() - start_time:.4f}s")

    def get_output_files(self) -> List[str]:
        """Get list of pickle files in the output directory"""
        try:
            files = [f for f in os.listdir(self.output_directory)
                     if f.endswith('.pkl')]
            logger.info(f"Found {len(files)} pickle files in output directory")
            return files
        except Exception as e:
            logger.error(f"Error listing files in output directory: {str(e)}")
            raise

    def load_pickle_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load a pickle file with different compression methods"""
        for compression in [None, 'zip', 'gzip']:
            try:
                start_time = time.time()
                df = pd.read_pickle(file_path, compression=compression)
                logger.debug(
                    f"Loaded {file_path} with {compression or 'no'} compression in {time.time() - start_time:.2f}s")
                return df
            except Exception as e:
                if compression == 'gzip':  # If we've tried all methods
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    return None
                continue

    def get_original_dataframe(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Get the original dataframe from the root path.
        Uses caching to avoid reloading the same file.
        """
        if filename in self.original_df_cache:
            return self.original_df_cache[filename]

        original_path = os.path.join(self.root_path, filename)
        if not os.path.exists(original_path):
            logger.error(f"Original file not found: {original_path}")
            return None

        df = self.load_pickle_file(original_path)
        if df is not None:
            self.original_df_cache[filename] = df

        return df

    def process_row(self, row: pd.Series, original_df: pd.DataFrame,
                    filename: str) -> Dict:
        """
        Process a single row from the dataframe.

        Args:
            row: DataFrame row with time, x, y, z columns
            original_df: Original dataframe from root_path
            filename: Name of the file (for metadata lookup)

        Returns:
            Dictionary with processed data
        """
        try:
            # Extract time and coordinates (ensuring they are integers)
            time_val = int(row['time'])
            x_val = int(row['x'])
            y_val = int(row['y'])
            z_val = int(row['z'])

            # Initialize coordinate space processor
            coord_processor = givenXYZreplyVelocityCube(
                pickle_filename=filename,
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

    def process_file(self, filename: str, max_rows: int = None) -> List[Dict]:
        """
        Process a single file, iterating through rows.

        Args:
            filename: Pickle filename in output directory
            max_rows: Maximum number of rows to process (for debugging)

        Returns:
            List of dictionaries with processed data for each row
        """
        start_time = time.time()
        logger.info(f"Processing file: {filename}")

        # Load file from output directory
        output_path = os.path.join(self.output_directory, filename)
        output_df = self.load_pickle_file(output_path)
        if output_df is None:
            logger.error(f"Failed to load output file: {output_path}")
            return []

        # Get original dataframe from root path
        original_df = self.get_original_dataframe(filename)
        if original_df is None:
            logger.error(f"Failed to load original file: {filename}")
            return []

        # Process each row
        results = []
        row_count = len(output_df) if max_rows is None else min(max_rows, len(output_df))

        logger.info(f"Processing {row_count} rows from {filename}")

        for i, (_, row) in enumerate(output_df.head(row_count).iterrows()):
            if i % 10 == 0:  # Log progress periodically
                logger.info(f"Processing row {i + 1}/{row_count} in {filename}")

            result = self.process_row(row, original_df, filename)
            results.append(result)

        elapsed = time.time() - start_time
        logger.info(f"Processed {len(results)} rows from {filename} in {elapsed:.2f}s")

        return results

    def update_output_with_velocities(self, filename: str, processed_data: List[Dict]) -> bool:
        """
        Update the output dataframe with processed velocity data.
        Uses batch updates instead of individual assignments.
        """
        update_start_time = time.time()
        try:
            # Load dataframe from output directory
            load_start = time.time()
            output_path = os.path.join(self.output_directory, filename)
            df = self.load_pickle_file(output_path)
            logger.debug(f"Load time for update: {time.time() - load_start:.4f}s")
        
            if df is None:
                logger.error(f"Failed to load output file for updating: {output_path}")
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
            df.to_pickle(output_path, compression='gzip')
            logger.debug(f"Save time for updated dataframe: {time.time() - save_start:.4f}s")
        
            logger.info(f"Updated {output_path} with velocity data")
            logger.debug(f"Total update time: {time.time() - update_start_time:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error updating output file: {str(e)}")
            logger.debug(f"Update failed after {time.time() - update_start_time:.4f}s")
            return False

    def run(self, max_files: int = None, max_rows_per_file: int = None):
        """
        Run the processing on all files.

        Args:
            max_files: Maximum number of files to process (for debugging)
            max_rows_per_file: Maximum number of rows to process per file
        """
        start_time = time.time()

        # Get list of files
        files = self.get_output_files()

        # Limit number of files if requested
        if max_files is not None:
            files = files[:max_files]

        logger.info(f"Processing {len(files)} files with max {max_rows_per_file} rows per file")

        # Process each file
        for i, filename in enumerate(files):
            logger.info(f"Processing file {i + 1}/{len(files)}: {filename}")

            # Process the file
            processed_data = self.process_file(filename, max_rows=max_rows_per_file)

            # Update output file with results
            if processed_data:
                success = self.update_output_with_velocities(filename, processed_data)
                if success:
                    logger.info(f"Successfully updated {filename}")
                else:
                    logger.error(f"Failed to update {filename}")

        elapsed = time.time() - start_time
        logger.info(f"Completed processing {len(files)} files in {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        # Create iterator instance
        iterator = DataIterator()

        # For testing, process only a few files with a limited number of rows
        # Remove these limits for production runs
        #iterator.run(max_files=1, max_rows_per_file=500)
        iterator.run()
        # For full processing, use:
        # iterator.run()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise