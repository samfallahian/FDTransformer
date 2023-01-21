import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(6, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1024)
        self.layer4 = nn.Linear(1024, 6)

    def forward(self, data, labels):
        x = torch.cat([data, labels], 1)
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        x = nn.LeakyReLU(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(9, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 1)

        def forward(self, data, labels):
        x = torch.cat([data, labels], 1)
        x = nn.LeakyReLU(self.layer1(x))
        x = nn.LeakyReLU(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

# Training loop

# Instantiate the generator and discriminator
gen = Generator()
disc = Discriminator()

# Define loss function and optimizers
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(gen.parameters(), lr=0.001)
disc_optimizer = optim.Adam(disc.parameters(), lr=0.001)

for epoch in range(200):
    # Generate fake data
    noise = torch.randn(batch_size, 3)
    fake_data = gen(data[torch.randint(0, len(data), (batch_size,))], labels[torch.randint(0, len(labels), (batch_size,))])

    # Compute loss for discriminator on real data
    disc_real_output = disc(data, labels)
    disc_real_loss = criterion(disc_real_output, torch.ones(batch_size, 1))
    
    # Compute loss for discriminator on fake data
    disc_fake_output = disc(fake_data, labels[torch.randint(0, len(labels), (batch_size,))])
    disc_fake_loss = criterion(disc_fake_output, torch.zeros(batch_size, 1))
    
    # Compute total loss for discriminator and perform backprop
    disc_loss = disc_real_loss + disc_fake_loss
    disc_optimizer.zero_grad()
        disc_loss.backward()
    disc_optimizer.step()

    # Generate fake data again and compute loss for generator
    noise = torch.randn(batch_size, 3)
    fake_data = gen(data[torch.randint(0, len(data), (batch_size,))], labels[torch.randint(0, len(labels), (batch_size,))])
    disc_fake_output = disc(fake_data, labels[torch.randint(0, len(labels), (batch_size,))])
    gen_loss = criterion(disc_fake_output, torch.ones(batch_size, 1))

    # Perform backprop for generator
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    # Print losses
    if epoch % 20 == 0:
        print("Epoch: %d, Discriminator loss: %.4f, Generator loss: %.4f" % (epoch, disc_loss.item(), gen_loss.item()))

