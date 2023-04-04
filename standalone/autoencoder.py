import pandas as pd
import array
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#Put in the above, specifically to deal with :
# FutureWarning: The input object of type 'Tensor' is an array-like implementing one of the corresponding protocols (`__array__`, `__array_interface__` or `__array_struct__`); but not a sequence (or 0-D). In the future, this object will be coerced as if it was first converted using `np.array(obj)`. To retain the old behaviour, you have to either modify the type 'Tensor', or assign to an empty array created with `np.empty(correct_shape, dtype=object)`.data = torch.from_numpy(np.asarray(data))

#Read in the normalized df
#df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6_with_normalized_values_100000_sample.pkl.gz')
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6_with_normalized_values.pkl.gz')



# Define the size of the latent representation
latent_size = 10


# Define the encoder architecture
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(125 * 64, latent_size)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return x


# Define the decoder architecture
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 125 * 64)
        self.unflatten = nn.Unflatten(1, (64, 125))
        self.conv1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.unflatten(x)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return x


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the training loop
def train(model, data_loader, num_epochs, learning_rate, device):
    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the device
    model = model.to(device)

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in data_loader:
            # Move the data to the device
            #Clearly pretty dangerous
            #TODO Find out what a list was even here to begin with
            data = data[0].to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, data)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Update the running loss
            running_loss += loss.item()
        # Print the epoch loss
        epoch_loss = running_loss / len(data_loader)
        print("Epoch {} loss: {:.4f}".format(epoch + 1, epoch_loss))
        # Save the model every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.save("model_epoch{}.pt".format(epoch + 1))

# Define a sample input tensor



#A helper function we will use later
def within_range_check (x,y,z):
    if (x < x_lower or x  > x_upper or y < y_lower or y > y_upper or z < z_lower or z > z_upper):
        return False
    else:
        return True

#Here is where we will store the coordinates that we can use "safely", this will also serve
#As a coordinate list that we can leverage
x_array_coordinates = []
y_array_coordinates = []
z_array_coordinates = []

#Easiest to have upper and lower bounds on our x,y, and z dimensions, so we have to store them somewhere
x_lower = float()
x_upper = float()
y_lower = float()
y_upper = float()
z_lower = float()
z_upper = float()


#Code copied from "find_potential_areas_to_autoencode"....sloppy I know
alldimensions = ['x', 'y', 'z']
#Now to find the limits for all
for i in alldimensions:
    # sort the column in ascending order and select the 6th value
    print('the below is for ' + i + ' iterable')
    ascending_value_sorted = df[i].sort_values()
    ascending_set = set(ascending_value_sorted)
    #ascending_set = ascending_set.update(ascending_value_sorted)
    #print(ascending_set)
    # Sorting the elements of the set and converting them to a list
    ascending_list = sorted(ascending_set)
    ascending_value = ascending_list[3]
    print(ascending_value)  # output: -125
    x_ascending = ascending_value

    # sort the column in descending order and select the 6th value
    descending_value = ascending_list[-4]
    print(descending_value)  # output: 125
    x_descending = descending_value
    print(ascending_list)
    if (i == 'x'):
        x_array_coordinates = ascending_list
        x_lower = ascending_value
        x_upper = descending_value
    if(i == 'y'):
        y_array_coordinates = ascending_list
        y_lower = ascending_value
        y_upper = descending_value
    else:
        z_array_coordinates = ascending_list
        z_lower = ascending_value
        z_upper = descending_value


#Just proving it works...sort of :)
print(y_array_coordinates)
print(y_lower)
print(z_upper)



#Have to remember...to always look for the values at the same TIME...hopefully I do that below!
#This should keep the row(s) in question unique, if NOT done, this will be a disaster
#The plan is to grab some number of "unique" entries in the dataframe, though these could be duplicate
#That should be unlikely. The hope is that sampling will be sufficient, so the target will be roughly 30% or so
#Of all data points. IF this is not nonsense, then the convergence should show that through the loss function and
#differences in saved models should be trivial on the (future) transformer model (that will use some number of
#x, y, and z, though probably not in the same proportions


#Some unit testing...
import array
x = -97
y = -52
z = -17
print(within_range_check(x,y,z))


#Some unit testing...this time for NOT OK to use...
x = -121
y = -52
z = -17
print(within_range_check(x,y,z))


df_singleton = df.sample()
print(df_singleton)
myx = df_singleton.x.iloc[0]
myy = df_singleton.y.iloc[0]
myz = df_singleton.z.iloc[0]
#print(myx)
print(within_range_check(myx,myy,myz))


#OK, pretty simple, take in a "one row" dataframe (we should check that)
#First are we in range?
#Next, pull all datapoints for the same "time" (which we take from the "master df")
#Now in this smaller slice, we look through the types we created above, namely x_array_coordinates etc.
#We go "two" (e.g. +/-) coordinates in all three dimensions
#This we will then assemble as a cohesive 5x5x5 (with respect to x, y, and z) dataframe
#We already have tensors available in this dataframe...so feeding these to an encoder through iteration should be possible
import numpy as np

def atomic_auto(df_reference):
    myx = df_reference.x.iloc[0]
    myy = df_reference.y.iloc[0]
    myz = df_reference.z.iloc[0]
    #We need to know what timestep
    '''
    
    '''

    #We can continue only if the range works
    if (within_range_check(myx,myy,myz)):
        #First gather two lower (x for now)
        indices_x = np.where(x_array_coordinates == myx)[0]
        indices_y = np.where(y_array_coordinates == myy)[0]
        indices_z = np.where(z_array_coordinates == myz)[0]
        #Now to define the ranges we will tollerate/use
        lower_x = indices_x - 2
        upper_x = indices_x + 2
        lower_y = indices_y - 2
        upper_y = indices_y + 2
        lower_z = indices_z - 2
        upper_z = indices_z + 2
        #print(lower_x.dtype)
        #Now to get a scalar...because otherwise it doesn't work!
        lower_x = lower_x[0]
        upper_x = upper_x[0]
        lower_y = lower_y[0]
        upper_y = upper_y[0]
        lower_z = lower_z[0]
        upper_z = upper_z[0]
        '''
        Now the actual 
        '''
        val_lx = x_array_coordinates[lower_x]
        val_ux = x_array_coordinates[upper_x]
        val_ly = y_array_coordinates[lower_y]
        val_uy = y_array_coordinates[upper_y]
        val_lz = z_array_coordinates[lower_z]
        val_uz = z_array_coordinates[upper_z]
        #print(valX)
        range_X = (val_lx, val_ux)
        range_Y = (val_ly, val_uy)
        range_Z = (val_lz, val_uz)
        #print(range_X)
        # Use boolean indexing to select rows with values within the specified ranges
        mask = (df['x'].between(*range_X) & df['y'].between(*range_Y) & df['z'].between(*range_Z) ) # & df['B'].between(*range_B) & df['C'].between(*range_C))
        #Apparently this has to be against the entire df or we have an indexing issue...
        result_partial = df[mask]
        #So now we take the time slice again...
        mask_time = df_reference['time'].iloc[0]
        result = result_partial[result_partial['time'] == mask_time]

        return result

    else:
        #print("Values were " + myx + ' ' + myy + ' ' + myz)
        # Raise an exception with an error message containing the float values
        raise ValueError("Values were: myx={} and myy={} and myz={}".format(myx, myy, myz))




# Initialize counters for iterations with and without errors
with_error = 0
without_error = 0

# Create a new instance of the autoencoder
autoencoder = Autoencoder()

# Set up the try block loop
while True:
    try:
        # Get a random row from the DataFrame
        # This shows as an interpreter error...but works. Bug?
        df_unitTest = df.sample()
        # Being arbritrary here and selecting only the first 'time' step
        myTime = df_unitTest.time.iloc[0]
        mask_time = (df['time'] == myTime)
        relativeTimedf = df[mask_time]
        result = atomic_auto(df_unitTest)

        input_tensor = torch.tensor(result[['vx_norm','vy_norm','vz_norm']].values)
        print("input tensor shape:")
        print(input_tensor.shape)
        #input_tensor = input_tensor.reshape(1, 375)
        print("NEW input tensor shape:")
        print(input_tensor.shape)
        dataset = TensorDataset(input_tensor)

        data_loader = DataLoader(dataset, batch_size=1)



        # Train one epoch of the autoencoder
        train(autoencoder, data_loader, num_epochs=1, learning_rate=0.001, device='mps')

        print("result df length is: ")
        print(len(result))
        print(result)
        # If the value is even, increment the counter for iterations without errors
        without_error += 1
    except ValueError as e:
        # If a ValueError is raised, increment the counter for iterations with errors and print the error message
        with_error += 1
        print(e)
    # Break out of the loop after a certain number of iterations
    if with_error + without_error == 10:
        break

# Print the final counts of iterations with and without errors
print("Iterations with error: {}".format(with_error))
print("Iterations without error: {}".format(without_error))





'''
#This shows as an interpreter error...but works. Bug?
df_unitTest = df.sample()
#Being arbritrary here and selecting only the first 'time' step
myTime = df_unitTest.time.iloc[0]
mask_time = (df['time'] == myTime)
relativeTimedf = df[mask_time]
result = atomic_auto(relativeTimedf)
print("result df length is: ")
print(len(result))
print(result)
'''


