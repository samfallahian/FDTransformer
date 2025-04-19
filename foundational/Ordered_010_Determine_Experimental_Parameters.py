from Ordered_001_Initialize import HostPreferences
import os
import pandas as pd
import torch
import json
from concurrent.futures import ThreadPoolExecutor

class MinimalProcessor(HostPreferences):
    def __init__(self, filename="experiment.preferences"):
        super().__init__(filename)
        if not hasattr(self, 'metadata_location'):
            raise AttributeError(
                "'metadata_location' is required but not set in the parent class (HostPreferences). Check your configuration.")
        if self.metadata_location is None:
            raise ValueError("'metadata_location' is set but contains None value. A valid path must be provided.")

    def process_file(self, file_path):
        # Try different methods to read the pickle file
        for compression in [None, 'zip', 'gzip']:
            try:
                with open(file_path, 'rb') as f:
                    if compression:
                        df = pd.read_pickle(f, compression=compression)
                    else:
                        df = pd.read_pickle(f)
                    break  # If successful, break the loop
            except Exception as e:
                if compression == 'gzip':  # If we've tried all methods
                    print(f"Error reading file {file_path}: {str(e)}")
                    return None
                continue
    
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
        print(f"\nPath Configuration:")
        print(f"Input: {self.raw_input}")
        print(f"Output: {self.output_directory}")
        print(f"Metadata Location: {self.metadata_location}")
        
        # Verify metadata_location path
        metadata_dir = os.path.dirname(self.metadata_location)
        print(f"Metadata directory will be: {metadata_dir}")
        
        # Ensure output directory exists and is writable
        try:
            os.makedirs(metadata_dir, exist_ok=True)
            if not os.access(metadata_dir, os.W_OK):
                raise PermissionError(f"Directory {metadata_dir} is not writable")
            print(f"Verified metadata directory exists and is writable: {metadata_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to setup metadata directory: {str(e)}")

        # Rest of the run method...
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.metadata_location), exist_ok=True)

        # Create list of absolute paths for each .pkl file
        file_paths = [os.path.join(self.raw_input, file) 
                     for file in os.listdir(self.raw_input) 
                     if file.endswith('.pkl')]
    
        print(f"Found {len(file_paths)} .pkl files to process")
        
        # Initialize a dictionary to hold metadata for all files
        all_files_metadata = {}
        processed_count = 0
        error_count = 0
        
        # Use multithreading to process files in parallel
        with ThreadPoolExecutor() as executor:
            # Process each file
            for file_path, metadata in zip(file_paths, executor.map(self.process_file, file_paths)):
                try:
                    if metadata is None:
                        error_count += 1
                        print(f"WARNING: No metadata returned for {os.path.basename(file_path)}")
                        continue
                    
                    if not isinstance(metadata, dict):
                        error_count += 1
                        print(f"ERROR: Expected dictionary metadata for {os.path.basename(file_path)}, got {type(metadata)}")
                        continue
                    
                    # Use filename as key for metadata
                    filename = os.path.basename(file_path)
                    all_files_metadata[filename] = metadata
                    processed_count += 1
                    print(f"Successfully processed: {filename}")
                    
                except Exception as e:
                    error_count += 1
                    print(f"ERROR: Failed to handle metadata for {os.path.basename(file_path)}: {str(e)}")
    
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(file_paths)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Errors: {error_count}")
        
        if not all_files_metadata:
            raise RuntimeError("No files were successfully processed! Check the errors above.")
        
        # Write all metadata to a JSON file
        print(f"\nWriting metadata for {len(all_files_metadata)} files to {self.metadata_location}")
        with open(self.metadata_location, 'w') as f:
            json.dump(all_files_metadata, f, indent=4)
        print("Metadata write complete")

if __name__ == "__main__":
    processor = MinimalProcessor()
    processor.run()