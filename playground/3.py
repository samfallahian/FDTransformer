class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(3+2, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, 6)
        self.physics_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x, label):
        # Generate physics parameter
        physics_params = self.physics_net(x)
        # concatenate with the noise and label
        x = torch.cat([x, physics_params], 1)
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        x = nn.LeakyReLU(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        return x
