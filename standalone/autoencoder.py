import pandas as pd
import array
import numpy as np

#Read in the normalized df
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6_with_normalized_values_100000_sample.pkl.gz')


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
        #print(indices_x.dtype)
        #x_one_below = x_array_coordinates[indices_x -1]
        #x_two_below = x_array_coordinates[indices_x -2]
        lower_x = indices_x - 2
        upper_x = indices_x + 2
        #print(lower_x.dtype)
        #Now to get a scalar...because otherwise it doesn't work!
        lower_x = lower_x[0]
        upper_x = upper_x[0]
        '''
        print(lower_x)
        lower_x = 56
        print(lower_x)
        '''
        val_lx = x_array_coordinates[lower_x]
        val_ux = x_array_coordinates[upper_x]
        #print(valX)
        range_X = (val_lx, val_ux)
        print(range_X)
        # Use boolean indexing to select rows with values within the specified ranges
        mask = (df['x'].between(*range_X)) # & df['B'].between(*range_B) & df['C'].between(*range_C))
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



#This shows as an interpreter error...but works. Bug?
df_unitTest = df.sample()
#Being arbritrary here and selecting only the first 'time' step
myTime = df_unitTest.time.iloc[0]
mask_time = (df['time'] == myTime)
relativeTimedf = df[mask_time]
result = atomic_auto(relativeTimedf)



