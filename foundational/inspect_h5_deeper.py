import h5py

def count_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        global dataset_count
        dataset_count += 1

dataset_count = 0

with h5py.File('/home/kkreth_umassd_edu/DL-PTV/3p6/tensor_5.hdf', 'r') as f:
    f.visititems(count_datasets)

print(f"Number of datasets: {dataset_count}")
