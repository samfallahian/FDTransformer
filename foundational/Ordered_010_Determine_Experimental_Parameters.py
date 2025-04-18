from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import torch
import json
from concurrent.futures import ThreadPoolExecutor

class MinimalProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        # Set default metadata location if not set by parent
        if not hasattr(self, 'metadata_location') or self.metadata_location is None:
            self.metadata_location = os.path.join(self.output_directory, 'metadata.json')

    def process_file(self, file_path):
        # Try different methods to read the pickle file
        try:
            # First try without compression
            with open(file_path, 'rb') as f:
                df = pd.read_pickle(f)
        except Exception as e:
            try:
                # If that fails, try with gzip compression
                with open(file_path, 'rb') as f:
                    df = pd.read_pickle(f, compression="gzip")
            except Exception as e2:
                print(f"Error reading file {file_path}: {str(e2)}")
                return None
    
        # Initialize a dictionary to store meta-data
        metadata_dict = {}
    
        # Define the columns of interest
        columns_of_interest = ['x', 'y', 'z', 'time', 'distance']
    
        for col in columns_of_interest:
            if col not in df.columns:
                continue
                
            # Convert column to PyTorch tensor
            tensor = torch.from_numpy(df[col].to_numpy())
    
            # Remove duplicates and sort the tensor
            tensor = torch.sort(torch.unique(tensor)).values
    
            # Store min, max, 4th min, 4th max
            metadata_dict[col + '_min'] = float(tensor[0])
            metadata_dict[col + '_max'] = float(tensor[-1])
            metadata_dict[col + '_4th_min'] = float(tensor[3]) if tensor.shape[0] > 3 else float('nan')
            metadata_dict[col + '_4th_max'] = float(tensor[-4]) if tensor.shape[0] > 3 else float('nan')
    
            # Enumerate the sorted tensor and convert to list
            metadata_dict[col + '_enumerated'] = list(map(float, tensor.tolist()))
    
        return metadata_dict

    def run(self):
        print(f"Working with paths:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.metadata_location), exist_ok=True)

        # Create list of absolute paths for each .pkl file
        file_paths = [os.path.join(self.raw_input, file) 
                     for file in os.listdir(self.raw_input) 
                     if file.endswith('.pkl')]
    
        # Initialize a dictionary to hold metadata for all files
        all_files_metadata = {}
    
        # Use multithreading to process files in parallel
        with ThreadPoolExecutor() as executor:
            # Process each file
            for file_path, metadata in zip(file_paths, executor.map(self.process_file, file_paths)):
                if metadata is not None:  # Only add successful results
                    # Use filename as key for metadata
                    all_files_metadata[os.path.basename(file_path)] = metadata
    
        # Write all metadata to a JSON file
        with open(self.metadata_location, 'w') as f:
            json.dump(all_files_metadata, f, indent=4)

if __name__ == "__main__":
    processor = MinimalProcessor()
    processor.run()