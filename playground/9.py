import torch

# Create a tensor to hold the coordinates
coordinates = torch.tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ...], dtype=torch.float32)

# Create a tensor to hold the features
features = torch.tensor([[f1_1, f1_2, f1_3], [f2_1, f2_2, f2_3], [f3_1, f3_2, f3_3], ...], dtype=torch.float32)

# Concatenate the coordinates and features
data = torch.cat((coordinates, features),dim=-1)
