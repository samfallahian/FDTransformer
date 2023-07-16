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
the below is for x iterable
-113.0
113.0
[-125.0, -121.0, -117.0, -113.0, -109.0, -105.0, -101.0, -97.0, -93.0, -89.0, -85.0, -81.0, -78.0, -74.0, -70.0, -66.0, -62.0, -58.0, -54.0, -50.0, -46.0, -42.0, -38.0, -34.0, -30.0, -26.0, -22.0, -18.0, -14.0, -10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0, 53.0, 57.0, 61.0, 65.0, 69.0, 73.0, 77.0, 81.0, 85.0, 89.0, 93.0, 97.0, 101.0, 105.0, 109.0, 113.0, 117.0, 121.0, 125.0]
the below is for y iterable
-72.0
75.0
[-83.0, -80.0, -76.0, -72.0, -68.0, -64.0, -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -28.0, -24.0, -20.0, -16.0, -12.0, -8.0, -4.0, -0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 23.0, 27.0, 31.0, 35.0, 39.0, 43.0, 47.0, 51.0, 55.0, 59.0, 63.0, 67.0, 71.0, 75.0, 79.0, 83.0, 87.0]
the below is for z iterable
-21.0
22.0
[-33.0, -29.0, -25.0, -21.0, -17.0, -13.0, -9.0, -5.0, -1.0, 3.0, 7.0, 11.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0]
'''


#So, if we take cubes of 5x5x5 (2 each way from a random point), even in the smallest dimension (z for some reason is MUCH smaller)
#This is a dozen for the z direction (not as many as I would hope...but this is by far the minimum axis)
#This allows us to avoid using any "border" dimensions that may have issues with surface tension
#Making them behave differently
#For now...and with this code, we don't concern ourselves with the perturbabion (we might in the future)...but
#Will do that seperately
#Obvi not including time...so we will have many snapshots of each of these "dozen" z coordinate
