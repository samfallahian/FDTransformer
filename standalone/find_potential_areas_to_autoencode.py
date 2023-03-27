#OK so here, we will enumerate the dimensions in x,y, and z coordinates
#Then if we "shrink" the space by taking the (for now hard-coded) 11 spaces "in" from all directions
#Then we should have the "safe" starting points, where there are always 10 spaces from any arbitrary random spot
#This way we can encode a 10x10x10 matrix (over and over and over)
#Most likely we will choose randomly a large number (given that these occur over time)
#Which for the encoding phase...we can patently ignore, because the encoding is "stateless"
#Order and consistency (in time) will only become important once we are using a transformer model to understand the
#language of velocity permutations


#First read in the data
import pandas as pd
import numpy as np
### PTV DATA
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6' + ".pkl", compression="zip")


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

#The output from the above was
'''
-105.0
109.0
-64.0
71.0
-13.0
18.0
'''


#So, if we take cubes of 5x5x5 (2 each way from a random point), even in the smallest dimension (z for some reason is MUCH smaller)
#This is a dozen for the z direction (not as many as I would hope...but this is by far the minimum axis)
#This allows us to avoid using any "border" dimensions that may have issues with surface tension
#Making them behave differently
#For now...and with this code, we don't concern ourselves with the perturbabion (we might in the future)...but
#Will do that seperately
#Obvi not including time...so we will have many snapshots of each of these "dozen" z coordinate
