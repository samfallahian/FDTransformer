from Ordered_001_Initialize import HostPreferences
import os
import json

class CleanFilesProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent class (HostPreferences). Check your configuration.")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' is set but contains None value. A valid path must be provided.")

    def process_file(self, file_path):
        # To be implemented based on specific requirements
        pass

    def run(self):
        print(f"\nPath Configuration:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")
        print(f"Metadata Location: {self.metadata_location}")
        
        # Verify metadata location exists and is readable
        if not os.path.exists(self.metadata_location):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_location}")
        if not os.access(self.metadata_location, os.R_OK):
            raise PermissionError(f"Metadata file is not readable: {self.metadata_location}")
        
        # Ensure output directory exists and is writable
        try:
            os.makedirs(self.output_directory, exist_ok=True)
            if not os.access(self.output_directory, os.W_OK):
                raise PermissionError(f"Directory {self.output_directory} is not writable")
            print(f"Verified output directory exists and is writable: {self.output_directory}")
        except Exception as e:
            raise RuntimeError(f"Failed to setup output directory: {str(e)}")

        # Additional implementation to be added based on specific requirements

if __name__ == "__main__":
    processor = CleanFilesProcessor()
    processor.run()