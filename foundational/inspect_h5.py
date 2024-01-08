import h5py

def inspect_hdf5_file(file_path, max_items=50):
    with h5py.File(file_path, 'r') as f:
        # Recursive function to print details of groups and datasets
        def inspect_group(group, prefix='', item_count=0):
            for key in group:
                if item_count >= max_items:
                    return  # Exit the loop once the maximum number of items has been printed
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    print(f"Dataset Name: {prefix}/{key}")
                    print(f"Shape: {item.shape}")
                    print(f"Data Type: {item.dtype}")
                    if item.attrs:
                        print("Attributes:")
                        for attr_name, attr_value in item.attrs.items():
                            print(f"  {attr_name}: {attr_value}")
                    # For debug, print the first 1 element of the dataset
                    if item.size > 0:
                        print("First 1 data point (if available):")
                        print(item[:1])
                    print('-' * 50)
                    item_count += 1
                elif isinstance(item, h5py.Group):
                    item_count = inspect_group(item, prefix=f"{prefix}/{key}", item_count=item_count)
            return item_count

        inspect_group(f)

if __name__ == "__main__":
    hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.hd5'  # Change this to your file's path
    #hdf5_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_external_links.hd5'
    # You can also specify the maximum number of items to print, e.g., inspect_hdf5_file(hdf5_file_path, max_items=50)
    inspect_hdf5_file(hdf5_file_path)
