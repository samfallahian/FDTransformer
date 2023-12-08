import h5py

def inspect_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # Recursive function to print details of groups and datasets
        def inspect_group(group, prefix=''):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"Dataset Name: {prefix}/{key}")
                    print(f"Shape: {item.shape}")
                    print(f"Data Type: {item.dtype}")
                    if item.attrs:
                        print("Attributes:")
                        for attr_name, attr_value in item.attrs.items():
                            print(f"  {attr_name}: {attr_value}")
                    # For debug, print the first 5 elements of the dataset
                    if item.size > 0:
                        print("First 5 data points (if available):")
                        print(item[:5])
                    print('-' * 50)
                elif isinstance(item, h5py.Group):
                    inspect_group(item, prefix=f"{prefix}/{key}")

        inspect_group(f)

if __name__ == "__main__":
    hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.hd5'  # Change this to your file's path
    hdf5_file_path = '/home/kkreth_umassd_edu/DL-PTV/3p6/tensor_5.hdf'  # Change this to your file's path
    hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/10p4/14.hdf'
    hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/11p4.hd5'
    inspect_hdf5_file(hdf5_file_path)
