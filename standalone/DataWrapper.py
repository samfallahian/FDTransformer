import numpy as np

class DataWrapper:
    def __init__(self):
        # Initialize coordinate arrays (125 points each for x, y, z)
        self._x_coords = np.zeros(125, dtype=np.int32)
        self._y_coords = np.zeros(125, dtype=np.int32)
        self._z_coords = np.zeros(125, dtype=np.int32)
        
        # Initialize 47 latent variables
        self._latents = np.zeros(47, dtype=np.float32)
        
        # Initialize distance separately
        self._distance = np.int32(0)

    # X coordinate getters and setters
    @property
    def x_coordinates(self) -> np.ndarray:
        return self._x_coords

    @x_coordinates.setter
    def x_coordinates(self, values: np.ndarray) -> None:
        if len(values) != 125:
            raise ValueError("Must provide 125 x coordinates")
        self._x_coords = values.astype(np.int32)

    # Y coordinate getters and setters
    @property
    def y_coordinates(self) -> np.ndarray:
        return self._y_coords

    @y_coordinates.setter
    def y_coordinates(self, values: np.ndarray) -> None:
        if len(values) != 125:
            raise ValueError("Must provide 125 y coordinates")
        self._y_coords = values.astype(np.int32)

    # Z coordinate getters and setters
    @property
    def z_coordinates(self) -> np.ndarray:
        return self._z_coords

    @z_coordinates.setter
    def z_coordinates(self, values: np.ndarray) -> None:
        if len(values) != 125:
            raise ValueError("Must provide 125 z coordinates")
        self._z_coords = values.astype(np.int32)

    # Latent variables getter and setter
    @property
    def latents(self) -> np.ndarray:
        return self._latents

    @latents.setter
    def latents(self, values: np.ndarray) -> None:
        if len(values) != 47:
            raise ValueError("Must provide 47 latent values")
        self._latents = values.astype(np.float32)

    # Distance property
    @property
    def distance(self) -> int:
        return self._distance

    @distance.setter
    def distance(self, value: int) -> None:
        self._distance = np.int32(value)

    # Generate individual coordinate getters and setters
    def __generate_coordinate_properties(self):
        # Generate x_1 through x_125
        for i in range(125):
            index = i
            prop_name = f'x_{i+1}'
            
            def getter(self, index=index):
                return self._x_coords[index]
                
            def setter(self, value, index=index):
                self._x_coords[index] = value
                
            setattr(DataWrapper, prop_name, property(getter, setter))

        # Generate y_1 through y_125
        for i in range(125):
            index = i
            prop_name = f'y_{i+1}'
            
            def getter(self, index=index):
                return self._y_coords[index]
                
            def setter(self, value, index=index):
                self._y_coords[index] = value
                
            setattr(DataWrapper, prop_name, property(getter, setter))

        # Generate z_1 through z_125
        for i in range(125):
            index = i
            prop_name = f'z_{i+1}'
            
            def getter(self, index=index):
                return self._z_coords[index]
                
            def setter(self, value, index=index):
                self._z_coords[index] = value
                
            setattr(DataWrapper, prop_name, property(getter, setter))

    # Generate individual latent getters and setters
    def __generate_latent_properties(self):
        for i in range(47):
            index = i
            prop_name = f'LATENT_{i+1}'
            
            def getter(self, index=index):
                return self._latents[index]
                
            def setter(self, value, index=index):
                self._latents[index] = value
                
            setattr(DataWrapper, prop_name, property(getter, setter))

# Generate all properties
DataWrapper._DataWrapper__generate_coordinate_properties(None)
DataWrapper._DataWrapper__generate_latent_properties(None)