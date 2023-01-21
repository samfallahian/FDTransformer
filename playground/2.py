import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(3, 256)
        self.layer2 = nn.Linear(256+3, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, 6)
        self.physics_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, label):
        # Generate physics parameter
        physics_params = self.physics_net(label)
        # concatenate with the noise and label
        x = torch.cat([x, physics_params, label], 1)
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        x = nn.LeakyReLU(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        return x, physics_params
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(9, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 1)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        x = nn.LeakyReLU(self.layer3(x))
        x = torch.sigmoid(self.layer4(x))
        return x

# Define the loss function and optimizers
criterion = nn.BCELoss()
d_optimizer = optim
