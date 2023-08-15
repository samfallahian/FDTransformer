import torch
from ConvolutionalAutoencoder import ConvolutionalAutoencoder

# Specify the path to the saved .torch file
file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1_exhaustive_tensors.torch"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Changed 'mps' to 'cpu'

# Load the data
loaded_data = torch.load(file_path)

# Prepare the model
saved_state_path = "/Users/kkreth/PycharmProjects/cgan/standalone/saved_models/checkpoint_300.pth"

# Instantiate the model first and move it to the device
model = ConvolutionalAutoencoder().to(device)

# Load the saved state into the model
checkpoint = torch.load(saved_state_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Create an empty list to store the new data
new_data = []

# Now to iterate through them all
for pair in loaded_data:
    coordinates = pair['coordinates']
    velocity = pair['velocity']
    velocity = velocity.to(torch.float32)
    velocity = velocity.permute(0, 2, 1)
    answer = model.encode(velocity.to(device))

    # Append a tuple containing the three values to the list
    new_data.append({'coordinates': coordinates, 'velocity': velocity, 'answer': answer})

# Specify the path to save the new .torch file
new_file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1_new_tensors.torch"

# Save the new data to a file
torch.save(new_data, new_file_path)

# Print a message to indicate that the file has been saved
print("New file has been saved to:", new_file_path)
