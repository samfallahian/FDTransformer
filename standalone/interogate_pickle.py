import pandas as pd

# Usage example
filename = '/Users/kkreth/PycharmProjects/cgan/dataset/3p6_with_normalized_values.pkl.gz'
df = pd.read_pickle(filename,  compression="gzip")


df.to_csv('/Users/kkreth/PycharmProjects/data/DL-PTV/3p6.csv', index=False)

# Now you can use the DataFrame
print(df.head(9))




