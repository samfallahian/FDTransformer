import os
import pickle

from CoordinateAnalyzer import CoordinateAnalyzer
from standalone import TransformLatent

'''
OK, at least for the file: /Users/kkreth/PycharmProjects/cgan/dataset/3p6
We know the "limits" to use where we can pick random rows (as a seed)
Then test if those rows are within bounds (using these as hard-coded for now)
Take all the 125 x 3 values and put them in a single row?
'''

'''
So, what do we need exactly?

1) Pick a random row in a dataframe
    a) Create a copy of the df that only has the same timestamp (this will drop >90% from consideration)
2) Use that as a potential starting point
3) If it falls in the accepted range, continue
4) Grab all 125 rows
5) These should be sorted in x, y, and z (I'm paranoid)
6) Should be able to then create one row of (normalized using TransformLatent) values that are 125 * 3 = 375 large
7) Once we have one million of these...we can try an auto-encoder

Worth noting that this isn't completely arbitrary...this is ~375,000,000 values:
The memory consumption of Python floats can vary depending on the implementation and platform. However, we can make a rough estimation based on the assumption that each float requires 24 bytes of memory in CPython (the reference implementation of Python).

To calculate the memory usage for 375,000,000 floats, we multiply the number of floats by the size of each float:

375,000,000 floats * 24 bytes/float = 9,000,000,000 bytes

Converting this to a more human-readable format:

9,000,000,000 bytes = 8.4375 gigabytes (GB)

So, approximately 8.4375 GB of memory would be required to store 375,000,000 Python floats in CPython. 

So...all of this could be cached and run from memory, which at this stage is paramount.

Previous experiments showed this:
/Users/kkreth/PycharmProjects/cgan_venvPY/bin/python /Users/kkreth/PycharmProjects/cgan/standalone/find_potential_areas_to_autoencode.py 
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

#TODO Make these not hard-coded if this works at all
x_bound_lower = -113
x_bound_upper = 113
y_bound_lower = -72
y_bound_upper = 75
z_bound_lower = -21
z_bound_upper = 22


#First read in the data
import pandas as pd
import numpy as np
### PTV DATA

# Check if pickle file exists
pickle_file = 'df_subset.pickle'
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        df_subset = pickle.load(f)
else:
    exit()


randomSingleton = df_subset.sample()
# Subset the DataFrame based on the target values
target_x = -85
target_y = -68
target_z = -17
randomSingleton = df_subset[(df_subset['x'] == target_x) & (df_subset['y'] == target_y) & (df_subset['z'] == target_z)]
randomTimeHappenedToB = randomSingleton.time



randomx = randomSingleton.x
randomy = randomSingleton.y
randomz = randomSingleton.z
arandomx = randomx.iloc[0]
arandomy = randomy.iloc[0]
arandomz = randomz.iloc[0]


analyzer = CoordinateAnalyzer(df_subset)
result = analyzer.get_nearest_values(arandomx, arandomy, arandomz)
#also_result = analyzer.get_all_combinations()

print("The size of the resulting array:")
print(len(result))

result_vxVYvz = result.loc[:, ['vx', 'vy', 'vz']]

converter = TransformLatent.FloatConverter()
converted_values = converter.convert(result_vxVYvz)



