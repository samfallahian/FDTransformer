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
model = ConvolutionalAutoencoder().to(device)  # Moved this line up

# Load the saved state into the model
checkpoint = torch.load(saved_state_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Access the first pair of coordinates and velocities
first_pair = loaded_data[0]
coordinates = first_pair['coordinates']
velocity = first_pair['velocity']

# Print the coordinates and velocities
print("Coordinates:", coordinates)
print("Velocity Tensor:", velocity)

# Now to iterate through them all
for pair in loaded_data:
    coordinates = pair['coordinates']
    velocity = pair['velocity']
    #velocity = velocity.squeeze(0)  # Removes size 1 dimensions
    #velocity = velocity.transpose(0, 1)  # Transposes the tensor
    velocity = velocity.to(torch.float32)  # Converts to float32
    # Reshape the tensor from (1, 125, 3) to (1, 3, 125)
    velocity = velocity.permute(0, 2, 1)
    answer = model.encode(velocity.to(device))  # Moves tensor to the same device as model
    print(answer)
