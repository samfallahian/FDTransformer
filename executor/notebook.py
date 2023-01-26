import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing

df = pd.read_pickle("dataset/test_500.pkl", compression="zip")

data = df.to_numpy()[:,:11]
labels = df.to_numpy()[:,11:]

scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
data = scalar.fit_transform(data)

num_epochs = 50
batch_size = 32
class_labels = df["label"].unique()
num_classes = len(class_labels)

class MyDataset(Dataset):
    def __init__(self, data,labels):
        # self.data = torch.tensor(data).float()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        # self.labels = torch.tensor(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] , self.labels[idx]


# Create instances of the dataset
dataset = MyDataset(data, labels)
print(type(dataset))
# Create a dataloader with a batch size of 32 and shuffle the data
train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(type(train_data))

# Define the generator and discriminator networks
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(12, 64)  # added 1 more input for the label
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 512)
        self.output = nn.Linear(512, 11)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(11, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x

# Create instances of the generator and discriminator
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)


# Train the CGAN
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_data):
        # Generate fake data with labels
        noise = torch.randn(batch_size, 11)
        if data.size()[0] < batch_size:
            continue
        fake_data = generator(torch.cat((noise, label), 1)) # concatenate noise and labels as input to generato

        # Train the discriminator
        disc_optimizer.zero_grad()
        real_loss = criterion(discriminator(data), torch.ones(batch_size, 1))
        fake_loss = criterion(discriminator(fake_data), torch.zeros(batch_size, 1))
        disc_loss = real_loss + fake_loss
        disc_loss.backward(retain_graph=True)
        disc_optimizer.step()

        # Train the generator
        gen_optimizer.zero_grad()
        gen_loss = criterion(discriminator(fake_data), torch.ones(batch_size, 1))
        gen_loss.backward(retain_graph=True)
        """retain_graph tells the autograd engine to retain the intermediate values of the graph,
        instead of freeing them, so that they can be used in the next backward pass."""
        gen_optimizer.step()

    # predictions = discriminator(data)
    # print("Predictions for real data: ", predictions, "with label: ", label)
    # predictions = discriminator(fake_data)
    # print("Predictions for fake data: ", predictions, "with label: ", label)
    print(f"Epoch {epoch + 1}: ")
    print(f"Generator Loss: {gen_loss}")
    print(f"Discriminator Loss : {disc_loss}")