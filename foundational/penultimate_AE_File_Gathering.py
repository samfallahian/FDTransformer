import h5py
import os


def merge_hdf5_files(directory_path, output_file):
    """
    Combine all HDF5 files in a given directory into one file.
    """
    with h5py.File(output_file, 'w') as f_out:
        for filename in os.listdir(directory_path):
            if filename.endswith('.hd5'):
                file_path = os.path.join(directory_path, filename)
                with h5py.File(file_path, 'r') as f_in:
                    for key in f_in.keys():
                        # Avoid overwriting existing keys
                        if key not in f_out:
                            f_in.copy(key, f_out)


def print_first_entry_and_statistics(output_file):
    """
    Print the first entry and statistics about the combined HDF5 file.
    """
    with h5py.File(output_file, 'r') as f:
        # Print the first entry
        first_key = list(f.keys())[0]
        print(f"First entry for key '{first_key}':\n{f[first_key][0]}")

        # Print statistics
        print("\nStatistics about the combined HDF5 file:")
        total_datasets = 0
        total_entries = 0
        for key in f.keys():
            total_datasets += 1
            dataset_size = len(f[key])
            total_entries += dataset_size
            print(f"  - Dataset '{key}' contains {dataset_size} entries.")
        print(f"\nTotal number of datasets: {total_datasets}")
        print(f"Total number of entries: {total_entries}")


# Main execution
directory_path = '/home/kkreth_umassd_edu/DL-PTV/_combined/'
output_file = '/home/kkreth_umassd_edu/DL-PTV/combined.hd5'
merge_hdf5_files(directory_path, output_file)
print_first_entry_and_statistics(output_file)
