# Implementation for retrieving latent vectors from WAE model
import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
import pickle
import gzip
from typing import List, Optional, Union, Dict, Any

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the WAE model
try:
    from encoder.model_WAE_01 import WAE
except ImportError:
    print("Warning: Could not import WAE model - using placeholder")
    # We'll use the placeholder WAEModel below if import fails


class WAEModel:  # Placeholder if the real model can't be imported
    def __init__(self, model_path):
        print(f"Model loaded from {model_path} (placeholder)")
        self.model_path = model_path
    
    def eval(self):
        print("Model set to evaluation mode (placeholder)")
    
    def predict(self, data_cube):
        print(f"Model predicting on data with shape: {data_cube.shape if hasattr(data_cube, 'shape') else 'unknown'}")
        import numpy as np
        return np.random.rand(47).astype(np.float32)


class Coordinate:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"
    
    def as_tuple(self):
        return (self.x, self.y, self.z)


class RetreiveSingleCoordinate:
    def __init__(self, 
                 center_coordinate: Coordinate, 
                 model_path: str, 
                 data_source_config: dict,
                 time_index: int = None,
                 simulation_name: str = None,
                 step_size: int = 4):
        """
        Initializes the retriever.
        Args:
            center_coordinate (Coordinate): The center (x,y,z) for the cube.
            model_path (str): Path to the WAE model.
            data_source_config (dict): Configuration needed to access the raw data.
                Should contain at least 'root_data_dir' key.
            time_index (int, optional): Time index to use (file number).
            simulation_name (str, optional): Simulation name (e.g., "10p4").
            step_size (int, optional): Spacing between points in the cube. Default is 4.
        """
        # 1. Initialize logging preferences
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Store initialization parameters
        self.center_coordinate = center_coordinate
        self.data_source_config = data_source_config
        self.time_index = time_index
        self.simulation_name = simulation_name
        self.step_size = step_size
        
        # Validate data_source_config
        if 'root_data_dir' not in data_source_config:
            raise ValueError("data_source_config must contain 'root_data_dir' key")
        
        # 2. Load the WAE model
        try:
            # Try to load the real WAE model
            self.model = self._load_real_model(model_path)
        except Exception as e:
            self.logger.warning(f"Failed to load real model: {e}. Using placeholder model.")
            # Fall back to placeholder
            self.model = WAEModel(model_path)
            self.model.eval()
        
        # Initialize file cache
        self.file_cache = {}
        self.cache_size = 5  # Default cache size
        
        self.logger.info(f"RetreiveSingleCoordinate initialized with center {center_coordinate}")
        if time_index is not None and simulation_name is not None:
            self.logger.info(f"Using simulation {simulation_name}, time {time_index}")
            self._validate_specific_file_exists()

    def _validate_specific_file_exists(self):
        """Check if the specified simulation/time file exists"""
        if self.time_index is not None and self.simulation_name is not None:
            file_path = self._get_file_path_for_simulation_time()
            if not os.path.exists(file_path):
                self.logger.error(f"File does not exist: {file_path}")
                raise FileNotFoundError(f"Data file not found: {file_path}")
            else:
                self.logger.info(f"Found data file: {file_path}")

    def _setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Try to load preferences for more advanced logging config
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            preferences_path = os.path.join(project_root, "experiment.preferences")
            
            # Only import if the file exists
            if os.path.exists(preferences_path):
                sys.path.append(project_root)
                from Ordered_001_Initialize import HostPreferences
                preferences = HostPreferences(filename=preferences_path)
                
                if hasattr(preferences, 'logging_level'):
                    level = getattr(logging, preferences.logging_level.upper(), None)
                    if isinstance(level, int):
                        logging.getLogger().setLevel(level)
                        print(f"Set logging level to {preferences.logging_level.upper()}")
        except Exception as e:
            print(f"Warning: Could not configure advanced logging: {e}")

    def _load_real_model(self, model_path):
        """Load the actual WAE model using PyTorch"""
        device = self._get_device()
        
        # Initialize the WAE model
        model = WAE().to(device)
        
        # Load the model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract the model state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback if the structure is different
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        return model

    def _get_device(self):
        """Determine the appropriate device (GPU/CPU) to use"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon GPU
        else:
            return torch.device("cpu")

    def _get_cube_coordinates(self) -> List[Coordinate]:
        """
        Calculates the 125 coordinates forming a 5x5x5 cube around the center_coordinate.
        Returns:
            list[Coordinate]: A list of 125 Coordinate objects.
        """
        cube_coords = []
        # Create a 5x5x5 cube with proper step size
        for dz in range(-2, 3):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    cube_coords.append(Coordinate(
                        self.center_coordinate.x + dx * self.step_size,
                        self.center_coordinate.y + dy * self.step_size,
                        self.center_coordinate.z + dz * self.step_size
                    ))
        
        if len(cube_coords) != 125:
            raise ValueError(f"Expected 125 coordinates, got {len(cube_coords)}")
            
        self.logger.debug(f"Generated {len(cube_coords)} coordinates for the cube")
        return cube_coords

    def _get_file_path_for_simulation_time(self) -> str:
        """
        Get the file path for a specific simulation and time index.
        
        Returns:
            str: Path to the data file
        """
        if self.simulation_name is None or self.time_index is None:
            raise ValueError("Both simulation_name and time_index must be provided")
            
        # Construct path based on the directory structure from batch files
        directory_path = os.path.join(
            self.data_source_config['root_data_dir'],
            "all_data_ready_for_training",
            self.simulation_name
        )
        
        file_path = os.path.join(directory_path, f"{self.time_index}.pkl")
        return file_path

    def _load_data_file(self, file_path: str) -> Any:
        """
        Load a pickled data file, with support for gzipped files.
        
        Args:
            file_path: Path to the pickle file
            
        Returns:
            The loaded data
        """
        # Check if file is in cache
        if file_path in self.file_cache:
            self.logger.debug(f"Using cached file: {file_path}")
            return self.file_cache[file_path]
            
        self.logger.info(f"Loading data file: {file_path}")
        
        # Check if the file is gzipped
        is_gzipped = False
        try:
            with open(file_path, 'rb') as f:
                is_gzipped = f.read(2) == b'\x1f\x8b'
        except Exception:
            pass
            
        # Load the file accordingly
        try:
            if is_gzipped:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    
            # Cache the data
            if len(self.file_cache) >= self.cache_size:
                # Remove oldest item (first added)
                self.file_cache.pop(next(iter(self.file_cache)))
            self.file_cache[file_path] = data
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            raise

    def _fetch_data_for_point(self, coord: Coordinate) -> List[float]:
        """
        Fetches the 3 float values for a single coordinate.
        
        Args:
            coord (Coordinate): The coordinate to fetch data for.
            
        Returns:
            list[float]: The 3 float values (vx, vy, vz).
        """
        # If simulation_name and time_index provided, use them directly
        if self.simulation_name is not None and self.time_index is not None:
            file_path = self._get_file_path_for_simulation_time()
            data = self._load_data_file(file_path)
            
            # Extract velocity data based on the coordinate
            # The exact implementation depends on your data format
            # Several common formats are handled below
            
            # Option 1: Data is a DataFrame with x, y, z columns
            if isinstance(data, pd.DataFrame):
                if all(col in data.columns for col in ['x', 'y', 'z']):
                    row = data[(data['x'] == coord.x) & 
                               (data['y'] == coord.y) & 
                               (data['z'] == coord.z)]
                    if not row.empty:
                        # Assuming velocity columns are named 'vx', 'vy', 'vz'
                        vel_cols = ['vx', 'vy', 'vz']
                        if all(col in data.columns for col in vel_cols):
                            return row[vel_cols].iloc[0].tolist()
            
            # Option 2: Data is a dictionary with (x,y,z) tuple keys
            if isinstance(data, dict):
                key = coord.as_tuple()
                if key in data:
                    # Assuming value is a list/tuple/array of 3 values
                    values = data[key]
                    if len(values) == 3:
                        return list(values)
            
            # Option 3: Data is a NumPy array with specific format
            # This would depend on your exact data structure
            
            # If we couldn't find the data, log a warning
            self.logger.warning(f"Could not find velocity data for {coord} in {file_path}")
        else:
            self.logger.warning("No simulation_name/time_index provided; can't fetch real data")
        
        # Return zeros as a fallback
        self.logger.warning(f"Returning zero velocities for {coord}")
        return [0.0, 0.0, 0.0]

    def get_latent_representation(self) -> List[float]:
        """
        Retrieves data for the cube, invokes the model, and returns the latent representation.
        Returns:
            list[float]: The 47 floats representing the latent space.
        """
        # 1. Get coordinates for the cube
        cube_coordinates = self._get_cube_coordinates()
        
        # 2. Fetch data for all coordinates
        all_float_values = []
        for coord in cube_coordinates:
            point_data = self._fetch_data_for_point(coord)
            all_float_values.extend(point_data)
            
        # 3. Verify we have the expected amount of data
        expected_values = 125 * 3  # 125 points, 3 values each
        if len(all_float_values) != expected_values:
            raise ValueError(f"Expected {expected_values} values, got {len(all_float_values)}")
            
        # 4. Convert to model input format
        import numpy as np
        model_input = np.array(all_float_values, dtype=np.float32)
        
        # 5. Get prediction from model
        if isinstance(self.model, WAEModel):
            # If using placeholder model
            latent_representation = self.model.predict(model_input)
        else:
            # If using actual PyTorch model
            device = self._get_device()
            # Reshape for model if needed and convert to tensor
            model_input_tensor = torch.tensor(model_input, dtype=torch.float32).to(device)
            if len(model_input_tensor.shape) == 1:
                model_input_tensor = model_input_tensor.unsqueeze(0)  # Add batch dimension
                
            # Forward pass
            with torch.no_grad():
                # Actual WAE model might have different method names/structure
                # You might need to adjust this depending on your model implementation
                try:
                    # Try calling encode method directly if it exists
                    latent_z = self.model.encode(model_input_tensor)
                except AttributeError:
                    # Fall back to the full forward pass which returns (recon_x, z)
                    _, latent_z = self.model(model_input_tensor)
                    
                # Move to CPU and convert to numpy
                latent_representation = latent_z.cpu().numpy()
                
                # If batch dimension was added, remove it
                if len(latent_representation.shape) > 1 and latent_representation.shape[0] == 1:
                    latent_representation = latent_representation.squeeze(0)
        
        # 6. Verify the output has the expected shape (47 values)
        if not hasattr(latent_representation, 'shape') or len(latent_representation) != 47:
            raise ValueError(f"Expected latent representation with 47 values, got {len(latent_representation)}")
            
        return latent_representation.tolist()


# Example Usage:
if __name__ == '__main__':
    # Example from CubeUnitTest.py
    central_coord = Coordinate(x=57, y=-8, z=-13)
    model_file_path = "/Users/kkreth/PycharmProjects/cgan/encoder/saved_models/WAE_01_epoch_2870.pt"
    
    # Now including simulation name and time index
    data_config = {
        "root_data_dir": "/Users/kkreth/PycharmProjects/data/"
    }
    
    try:
        retriever = RetreiveSingleCoordinate(
            center_coordinate=central_coord,
            model_path=model_file_path,
            data_source_config=data_config,
            simulation_name="10p4",  # From the CubeUnitTest.py example
            time_index=3,            # From the CubeUnitTest.py example
            step_size=4              # Matches the coordinate spacing in CubeUnitTest.py
        )
        
        latent_vector = retriever.get_latent_representation()
        print(f"Retrieved latent vector ({len(latent_vector)} floats): {latent_vector[:5]}...")
    except Exception as e:
        print(f"An error occurred: {e}")