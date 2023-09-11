import torch
import os  # <-- Import os for path operation
import sys
sys.path.append("/home/kkreth_umassd_edu/cgan/standalone/")
from HybridAutoencoder import HybrdidAutoencoder
import gzip
import shutil

# Check for command line arguments
if len(sys.argv) < 3:
    print("Usage: python your_script_name.py <path_to_file> <saved_state_path>")
    sys.exit(1)

# Retrieve the file path and saved state path from the command line arguments
file_path = sys.argv[1]
saved_state_path = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
loaded_data = torch.load(file_path)

# Instantiate the model first and move it to the device
model = HybrdidAutoencoder().to(device)

# Load the saved state into the model
checkpoint = torch.load(saved_state_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Create an empty list to store the new data
new_data = []

# Show me what this model looks like
print(dir(model))

# Now to iterate through them all
for pair in loaded_data:
    coordinates = pair['coordinates']
    velocity = pair['velocity']
    velocity = velocity.to(torch.float32)
    velocity = velocity.permute(0, 2, 1)
    answer = model.encode(velocity.to(device))

    # Append a tuple containing the three values to the list
    new_data.append({'coordinates': coordinates, 'velocity': velocity, 'answer': answer})

# Derive the path to save the new .torch file
base_directory = os.path.dirname(file_path)  # <-- Extract directory of file_path

# Extract the prefix from the original file name
prefix = os.path.basename(file_path).split('_')[0]

# Use the extracted prefix to create the new file name
new_file_name = prefix + "_tensor_for_transformer.torch"
new_file_path = os.path.join(base_directory, new_file_name)  # <-- Join directory with new file name

# Save the new data to a file
torch.save(new_data, new_file_path)

# Gzip the saved .torch file
with open(new_file_path, 'rb') as f_in:
    with gzip.open(new_file_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(new_file_path)  # Remove the original .torch file after gzipping

# Print a message to indicate that the file has been saved and compressed
print("New gzipped file has been saved to:", new_file_path + '.gz')