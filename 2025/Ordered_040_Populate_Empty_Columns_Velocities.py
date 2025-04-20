from Ordered_001_Initialize import HostPreferences
import numpy as np
import json
import sys


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


def main():
    # Example usage
    try:
        processor = givenXYZreplyVelocityCube(
            pickle_filename="3p6.pkl",
            x=-113,
            y=35,
            z=3
        )
        print("Successfully initialized with metadata")
        print(f"Number of x coordinates: {len(processor.x_enumerated)}")
        print(f"Number of y coordinates: {len(processor.y_enumerated)}")
        print(f"Number of z coordinates: {len(processor.z_enumerated)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()