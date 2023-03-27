import pandas as pd
from sklearn import preprocessing
from utils import helpers
import numpy as np



        ### PTV DATA
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/3p6' + ".pkl", compression="zip")
        # labels = df.drop(df.columns.difference(["label"]), axis=1).to_numpy()
        # data = df.drop(["label"], axis=1).to_numpy()

vxdescribe = df['vx'].describe()
vydescribe = df['vy'].describe()
vzdescribe = df['vz'].describe()

print(df.head(200))

print(vxdescribe)
print(vydescribe)
print(vzdescribe)

print('number of uniques by vx, vy, and vz')
v_uniq =np.unique(df[['vx', 'vy', 'vz']], axis=0)
print(v_uniq)
print('of size')
print(v_uniq.size)


print('number of uniques by vx, vy, and vz with pressure values as well')
v_and_p_uniq = np.unique(df[['vx', 'vy', 'vz', 'px', 'py', 'pz']], axis=0)
print(v_and_p_uniq)
print('of size')
print(v_and_p_uniq.size)

print('another way of computing this for v and p values gives this result')
cols = ['vx', 'vy', 'vz', 'px', 'py', 'pz']
unique_l = pd.concat([df[col] for col in cols]).unique()