import os
import sys
import h5py


def merge_hdf(input_directory, output_file):
    if not os.path.exists(input_directory):
        print(f"{input_directory} does not exist. Exiting without creating an output file.")
        sys.exit()

    files = os.listdir(input_directory)
    matched_files = []

    with h5py.File(output_file, 'w') as output_hdf:
        for filename in files:
            filepath = os.path.join(input_directory, filename)
            print(f"Checking {filepath}")

            if filename.startswith("tensor") and (filename.endswith('.h5') or filename.endswith('.hd5') or filename.endswith('.hdf')):
                print(f"Matched {filepath}")
                matched_files.append(filepath)

                with h5py.File(filepath, 'r') as input_hdf:
                    # Copy all datasets and attributes from the input file to the output file
                    input_hdf.visititems(lambda name, obj: copy_obj_to_file(name, obj, output_hdf))

    if not matched_files:
        print(f"No matching HDF files found in {input_directory}. Exiting without creating an output file.")
        sys.exit()


def copy_obj_to_file(name, obj, output_hdf):
    """Copy an object from the input HDF5 file to the output HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        # Copy the dataset to the output file
        output_hdf.create_dataset(name, data=obj[...])

        # Copy attributes from the input file's root to the dataset
        for attr_name, attr_value in obj.file.attrs.items():
            output_hdf[name].attrs[attr_name] = attr_value


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_hdf.py <input_directory> <output_file>")
        sys.exit()

    input_directory = sys.argv[1]
    output_file = sys.argv[2]

    merge_hdf(input_directory, output_file)
