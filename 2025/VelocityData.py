import logging
import sys
import os
import pandas as pd
import numpy as np
import time
from Ordered_001_Initialize import HostPreferences
from dataclasses import dataclass
from typing import List, Tuple, Optional
from CoordinateSpace import givenXYZreplyVelocityCube

# Constants
EXPECTED_COORDINATE_COUNT = 125  # Expected number of coordinate combinations

# Initialize host preferences
preferences = HostPreferences()

# Configure logging based on preferences
numeric_level = getattr(logging, preferences.logging_level.upper(), None)
if not isinstance(numeric_level, int):
    # Default to INFO if level from preferences is invalid
    numeric_level = logging.INFO
    print(f"Invalid logging level: {preferences.logging_level}. Defaulting to INFO.")

# Set up logging
logging.basicConfig(level=numeric_level)
logger = logging.getLogger(__name__)

logger.debug(f"Logging initialized at level: {preferences.logging_level}")


@dataclass
class VelocityData:
    """Class to store velocity data extracted from dataframe"""
    vx_values: List[float]
    vy_values: List[float]
    vz_values: List[float]
    combinations: List[Tuple[float, float, float]]


class VelocitySpace:
    """
    A class to extract velocity data from a dataframe at a specific time point,
    for a given set of 3D coordinates.
    """
    
    def __init__(self, dataframe: pd.DataFrame, time_point: int, coordinate_source=None):
        """
        Initialize VelocitySpace with a dataframe and time point.
        
        Args:
            dataframe (pd.DataFrame): Input dataframe containing velocity data
            time_point (int): Time point to filter the data
            coordinate_source: Optional object that provides coordinates (like givenXYZreplyVelocityCube)
        """
        self.df = dataframe
        self.time_point = time_point
        self.filtered_df = None
        self.coordinate_source = coordinate_source
        
        start_time = time.time()
        
        # Validate the dataframe
        self._validate_dataframe()
        
        # Filter the dataframe by time
        self._filter_by_time(time_point)
        
        init_time = time.time() - start_time
        logger.debug(f"Initialized VelocitySpace with {len(self.filtered_df)} data points at time {time_point} (took {init_time:.4f} seconds)")
    
    def _validate_dataframe(self):
        """Validate that the input dataframe has the required structure."""
        start_time = time.time()
        
        if self.df is None:
            error_msg = "Input dataframe cannot be None"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
            
        # Check for required columns
        required_cols = ['time', 'vx', 'vy', 'vz', 'x', 'y', 'z']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            error_msg = f"Input dataframe missing required columns: {missing_cols}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
        validate_time = time.time() - start_time
        logger.debug(f"Input dataframe validated with {len(self.df)} rows (took {validate_time:.4f} seconds)")
    
    def _filter_by_time(self, time_point: int):
        """
        Filter the dataframe by the specified time point.
        
        Args:
            time_point (int): The time point to filter by
        """
        start_time = time.time()
        
        self.filtered_df = self.df[self.df['time'] == time_point].copy()
        
        filter_time = time.time() - start_time
        logger.debug(f"Filtered dataframe to {len(self.filtered_df)} rows at time {time_point} (took {filter_time:.4f} seconds)")
        
        # Remove the incorrect validation - we only expect 125 rows after filtering
        # by both time AND coordinates, not just by time
        if len(self.filtered_df) == 0:
            error_msg = f"No data found after filtering by time {time_point}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    def get_velocities_from_coordinates(self, coordinates: List[Tuple[int, int, int]]) -> VelocityData:
        """
        Get velocity values for the specified coordinates.
        
        Args:
            coordinates (List[Tuple[int, int, int]]): List of coordinate tuples (x, y, z)
            
        Returns:
            VelocityData: Dataclass containing velocity values and combinations
        """
        start_time = time.time()
        
        # Validate input coordinates count
        if len(coordinates) != EXPECTED_COORDINATE_COUNT:
            error_msg = f"Expected exactly {EXPECTED_COORDINATE_COUNT} coordinate sets, but received {len(coordinates)}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
            
        vx_values = []
        vy_values = []
        vz_values = []
        velocity_combinations = []
        
        # Create a lookup dictionary for fast access
        dict_start = time.time()
        coord_to_velocity = {}
        for _, row in self.filtered_df.iterrows():
            coord_key = (int(row['x']), int(row['y']), int(row['z']))
            coord_to_velocity[coord_key] = (float(row['vx']), float(row['vy']), float(row['vz']))
        dict_time = time.time() - dict_start
        logger.debug(f"Created lookup dictionary with {len(coord_to_velocity)} entries (took {dict_time:.4f} seconds)")
        
        # Process each coordinate
        missing_coords = []
        for coord in coordinates:
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            coord_key = (x, y, z)
            
            if coord_key in coord_to_velocity:
                vx, vy, vz = coord_to_velocity[coord_key]
                vx_values.append(vx)
                vy_values.append(vy)
                vz_values.append(vz)
                velocity_combinations.append((vx, vy, vz))
            else:
                missing_coords.append(coord_key)
        
        # Check if we found all coordinates
        if missing_coords:
            error_msg = f"Could not find velocity data for {len(missing_coords)} coordinates. First few missing: {missing_coords[:5]}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
        # Validate we found all velocities
        if len(velocity_combinations) != EXPECTED_COORDINATE_COUNT:
            error_msg = f"Expected to retrieve {EXPECTED_COORDINATE_COUNT} velocity combinations, but got {len(velocity_combinations)}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
        total_time = time.time() - start_time
        logger.debug(f"Retrieved {len(velocity_combinations)} velocity combinations (took {total_time:.4f} seconds)")
        
        # Log all velocity data similar to how coordinates are logged in CoordinateSpace
        logger.debug("Velocity data for all coordinates:")
        for i, ((x, y, z), (vx, vy, vz)) in enumerate(zip(coordinates, velocity_combinations), 1):
            logger.debug(f"Point {i:3d}/{len(velocity_combinations)}: Coord ({x}, {y}, {z}) → Velocity (vx={vx:.4f}, vy={vy:.4f}, vz={vz:.4f})")
        
        return VelocityData(
            vx_values=vx_values,
            vy_values=vy_values,
            vz_values=vz_values,
            combinations=velocity_combinations
        )
    
    def get_velocities_from_source(self) -> VelocityData:
        """
        Get velocity values using the attached coordinate source.
        
        Returns:
            VelocityData: Dataclass containing velocity values and combinations
        """
        start_time = time.time()
        
        if not self.coordinate_source:
            error_msg = "No coordinate source is attached to this VelocitySpace instance"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
            
        # Get coordinates from the source
        try:
            coord_start = time.time()
            coordinates = self.coordinate_source.locateNeighbors()
            coord_time = time.time() - coord_start
            logger.debug(f"Retrieved {len(coordinates)} coordinates from source (took {coord_time:.4f} seconds)")
        except Exception as e:
            error_msg = f"Failed to get coordinates from source: {str(e)}"
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
            
        # Use these coordinates to get velocities
        result = self.get_velocities_from_coordinates(coordinates)
        
        total_time = time.time() - start_time
        logger.debug(f"Total time to get velocities from source: {total_time:.4f} seconds")
        
        return result


def main():
    # Example usage
    try:
        overall_start = time.time()
        
        # Use the same pickle filename and coordinates as in CoordinateSpace
        pickle_filename = "3p6.pkl"
        x_coord = -113
        y_coord = 35
        z_coord = 3
        time_point = 1  # Example time point
        
        # Use your custom pickle path as specified
        pickle_path = '/Users/kkreth/PycharmProjects/data/all_data_cleaned_dtype_correct/3p6.pkl'
        
        print(f"Loading dataframe from {pickle_path}")
        # Load the dataframe
        df_start = time.time()
        df = None
        for compression in ['gzip', 'zip', None]:
            try:
                df = pd.read_pickle(pickle_path, compression=compression)
                df_time = time.time() - df_start
                print(f"Successfully loaded dataframe with {len(df)} rows using {compression or 'no'} compression (took {df_time:.4f} seconds)")
                break
            except Exception as e:
                if compression is None:
                    print(f"Failed to read pickle file {pickle_path}: {str(e)}", file=sys.stderr)
                    sys.exit(1)
                continue
        
        print(f"Initializing coordinate processor for {pickle_filename}")
        # Initialize the coordinate processor
        coord_start = time.time()
        coord_processor = givenXYZreplyVelocityCube(
            pickle_filename=pickle_filename,
            x=x_coord,
            y=y_coord,
            z=z_coord
        )
        coord_init_time = time.time() - coord_start
        print(f"Coordinate processor initialized (took {coord_init_time:.4f} seconds)")
        
        # Get the coordinates
        neighbors_start = time.time()
        coordinates = coord_processor.locateNeighbors()
        neighbors_time = time.time() - neighbors_start
        print(f"Retrieved {len(coordinates)} coordinate combinations (took {neighbors_time:.4f} seconds)")
        
        # Initialize VelocitySpace with the dataframe and time point
        print(f"Initializing VelocitySpace with dataframe containing {len(df)} rows")
        velocity_start = time.time()
        velocity_space = VelocitySpace(
            dataframe=df,
            time_point=time_point,
            coordinate_source=coord_processor
        )
        velocity_init_time = time.time() - velocity_start
        print(f"VelocitySpace initialized (took {velocity_init_time:.4f} seconds)")
        
        # Get velocities using the pre-computed coordinates
        velocities_start = time.time()
        velocity_data = velocity_space.get_velocities_from_coordinates(coordinates)
        velocities_time = time.time() - velocities_start
        print(f"Velocity data retrieved (took {velocities_time:.4f} seconds)")
        
        # Display results
        print(f"\nSuccessfully retrieved {len(velocity_data.combinations)} velocity values")
        print("\nSample of first 5 velocity data points:")
        for i in range(min(5, len(velocity_data.combinations))):
            coord = coordinates[i]
            velocity = velocity_data.combinations[i]
            print(f"Coordinate {coord} → Velocity: (vx={velocity[0]:.4f}, vy={velocity[1]:.4f}, vz={velocity[2]:.4f})")
        
        # Calculate statistics for velocity components
        vx_array = np.array(velocity_data.vx_values)
        vy_array = np.array(velocity_data.vy_values)
        vz_array = np.array(velocity_data.vz_values)
        
        print("\nVelocity component statistics:")
        print(f"vx: min={vx_array.min():.4f}, max={vx_array.max():.4f}, mean={vx_array.mean():.4f}, std={vx_array.std():.4f}")
        print(f"vy: min={vy_array.min():.4f}, max={vy_array.max():.4f}, mean={vy_array.mean():.4f}, std={vy_array.std():.4f}")
        print(f"vz: min={vz_array.min():.4f}, max={vz_array.max():.4f}, mean={vz_array.mean():.4f}, std={vz_array.std():.4f}")
        
        total_time = time.time() - overall_start
        print(f"\nTotal execution time: {total_time:.4f} seconds")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()