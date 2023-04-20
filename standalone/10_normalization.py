import pandas as pd
import torch

#For now a simple single file
df = pd.read_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/6p6' + ".pkl", compression="zip")

#Now let's drop what I know I won't need
df = df.drop(columns=['px','py','pz','distance'])
df_subset = df[['vx','vy','vz']]


#First some housekeeping
df_subset.vx = df_subset.vx.astype('float32')
df_subset.vy = df_subset.vy.astype('float32')
df_subset.vz = df_subset.vz.astype('float32')

# Example tensor on CUDA device
tensor_cuda = torch.tensor(df_subset.values, device='mps')

# Calculate the minimum and maximum values of the tensor
min_val = torch.min(tensor_cuda)
max_val = torch.max(tensor_cuda)

# Perform min-max normalization using tensor operations
tensor_normalized = (tensor_cuda - min_val) / (max_val - min_val)

# Move the normalized tensor back to CPU device and print it
tensor_cpu = tensor_normalized.cpu()
print(tensor_cpu)

size = tensor_cpu.numel()
print(size)
print(tensor_cpu.shape)


mint = tensor_cpu.min()
print(mint)

maxt = tensor_cpu.max()
print(maxt)

x_norm = tensor_cpu.numpy()[:,0]
print(x_norm)

y_norm = tensor_cpu.numpy()[:,1]
print(y_norm)

z_norm = tensor_cpu.numpy()[:,2]
print(z_norm)

#With NOT a lot of testing (other than the above) this looks corect.
#Now to place these values in the dataframe next to their non-normalized cousins.

df['vx_norm'] = x_norm
df['vy_norm'] = y_norm
df['vz_norm'] = z_norm

print(df.describe())

print(df.dtypes)

# Save the DataFrame to disk with pickle and gzip compression
df.to_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/6p6_with_normalized_values.pkl.gz', compression='gzip', protocol=5, storage_options=None)
df_100000 = df.sample(100000)
df_100000.to_pickle('/Users/kkreth/PycharmProjects/cgan/dataset/6p6_with_normalized_values_100000_sample.pkl.gz', compression='gzip', protocol=5, storage_options=None)