import logging
from Ordered_001_Initialize import HostPreferences
import numpy as np
import json
import sys
from dataclasses import dataclass
from typing import List, Tuple

'''
This is SO complex, really just because...well...Python. These can be retreived with code similar to:

        for i, (x, y, z) in enumerate(combinations, 1):
            assert all(isinstance(coord, int) for coord in (x, y, z)), f"Non-integer coordinates found: ({type(x)}, {type(y)}, {type(z)})"
            logger.debug(f"Point {i:3d}/{len(combinations)}: ({x}, {y}, {z})")

Should the need ever arise! :)
'''

# Constants
NEIGHBORS_EACH_DIRECTION = 2  # This gives us 5 points per dimension (2 below + center + 2 above)

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
def find_neighbors(value: int, enum_list: List[int], dimension: str) -> List[int]:
    try:
        current_idx = enum_list.index(value)
        # Convert all values to integers explicitly
        enum_list = [int(x) for x in enum_list]

        start_idx = max(0, current_idx - NEIGHBORS_EACH_DIRECTION)
        end_idx = min(len(enum_list), current_idx + NEIGHBORS_EACH_DIRECTION + 1)

        # Ensure we return integers
        return [int(enum_list[i]) for i in range(start_idx, end_idx)]
    except ValueError:
        raise ValueError(f"Value {value} not found in {dimension} enumerated list")

@dataclass
class CoordinateSpace:
    x_values: List[int]
    y_values: List[int]
    z_values: List[int]
    combinations: List[Tuple[int, int, int]]

    @classmethod
    def create_from_point(cls, x: int, y: int, z: int,
                         x_enumerated: List[int],
                         y_enumerated: List[int],
                         z_enumerated: List[int]) -> 'CoordinateSpace':
        # Ensure all input values are integers
        x_values = [int(x) for x in find_neighbors(x, x_enumerated, 'x')]
        y_values = [int(y) for y in find_neighbors(y, y_enumerated, 'y')]
        z_values = [int(z) for z in find_neighbors(z, z_enumerated, 'z')]

        combinations = [
            (int(x), int(y), int(z))  # Explicit integer conversion for each coordinate
            for x in x_values
            for y in y_values
            for z in z_values
        ]

        return cls(x_values=x_values,
                  y_values=y_values,
                  z_values=z_values,
                  combinations=combinations)


class givenXYZreplyVelocityCube(HostPreferences):
    def __init__(self, filename="experiment.preferences", pickle_filename=None, x=None, y=None, z=None):
        """
        Initialize the velocity cube generator with coordinate validation.

        Args:
            filename (str): Configuration file name (inherited from HostPreferences)
            pickle_filename (str): Name of the pickle file (used as key in metadata)
            x (float): X coordinate value to look up
            y (float): Y coordinate value to look up
            z (float): Z coordinate value to look up
        """
        super().__init__(filename)
        if not all([pickle_filename, x is not None, y is not None, z is not None]):
            raise ValueError("Must provide pickle_filename and x, y, z coordinates")

        self.pickle_filename = pickle_filename
        self.x = float(x)  # Convert to float for comparison
        self.y = float(y)
        self.z = float(z)

        # Load and validate metadata
        self._load_metadata()
        # Validate input coordinates against enumerated lists
        self._validate_coordinates()

    def _load_metadata(self):
        """Load and validate metadata from file."""
        try:
            with open(self.metadata_location, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.metadata_location}: {str(e)}")

        if self.pickle_filename not in self.metadata:
            raise ValueError(f"No metadata found for file: {self.pickle_filename}")

        self.file_metadata = self.metadata[self.pickle_filename]

        # Get enumerated coordinates with empty list defaults
        self.x_enumerated = self.file_metadata.get('x_enumerated', [])
        self.y_enumerated = self.file_metadata.get('y_enumerated', [])
        self.z_enumerated = self.file_metadata.get('z_enumerated', [])

        if not all([self.x_enumerated, self.y_enumerated, self.z_enumerated]):
            raise ValueError(f"Missing enumerated coordinates in metadata for {self.pickle_filename}")

    def _validate_coordinates(self):
        """
        Validate that input coordinates exist in their respective enumerated lists.
        Raises ValueError with detailed message if any coordinate is not found.
        """
        # Check each coordinate
        if self.x not in self.x_enumerated:
            raise ValueError(
                f"X coordinate {self.x} not found in available x-coordinates.\n"
                f"Available x-coordinates: {self.x_enumerated}"
            )

        if self.y not in self.y_enumerated:
            raise ValueError(
                f"Y coordinate {self.y} not found in available y-coordinates.\n"
                f"Available y-coordinates: {self.y_enumerated}"
            )

        if self.z not in self.z_enumerated:
            raise ValueError(
                f"Z coordinate {self.z} not found in available z-coordinates.\n"
                f"Available z-coordinates: {self.z_enumerated}"
            )

    def locateNeighbors(self) -> List[Tuple[int, int, int]]:
        """
        Locate two neighbors above and below the current x, y, z coordinates.

        Returns:
            List of coordinate tuples (x,y,z) for all neighboring points in 3D space.
        """
        coordinate_space = CoordinateSpace.create_from_point(
            self.x, self.y, self.z,
            self.x_enumerated, self.y_enumerated, self.z_enumerated
        )

        combinations = coordinate_space.combinations

        # Debug logging of all coordinates
        logger.debug("Generated coordinate combinations:")
        for i, (x, y, z) in enumerate(combinations, 1):
            assert all(isinstance(coord, int) for coord in (x, y, z)), f"Non-integer coordinates found: ({type(x)}, {type(y)}, {type(z)})"
            logger.debug(f"Point {i:3d}/{len(combinations)}: ({x}, {y}, {z})")

        return combinations


def main():
    # Example usage
    try:
        processor = givenXYZreplyVelocityCube(
            pickle_filename="3p6.pkl",
            x=-113,
            y=35,
            z=3
        )
        neighbor_coordinates = processor.locateNeighbors()
        print(f"Total number of neighboring coordinates: {len(neighbor_coordinates)}")
        print("Sample of first few coordinates:")
        for coord in neighbor_coordinates[:5]:
            print(f"(x={coord[0]}, y={coord[1]}, z={coord[2]})")
        print("Successfully initialized with metadata")
        print(f"Number of x coordinates: {len(processor.x_enumerated)}")
        print(f"Number of y coordinates: {len(processor.y_enumerated)}")
        print(f"Number of z coordinates: {len(processor.z_enumerated)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()