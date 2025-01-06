import os
import sys
import h5py
import glob
import concurrent.futures

MAX_THREADS = 10


def process_file(filepath, output_hdf, verbose):
    if verbose:
        print(f"Processing {filepath}")

    with h5py.File(filepath, 'r') as input_hdf:
        # Copy all datasets and attributes from the input file to the output file
        input_hdf.visititems(lambda name, obj: copy_obj_to_file(name, obj, output_hdf, verbose))


def merge_hdf(input_directory, output_file, verbose=False):
    if not os.path.exists(input_directory):
        print(f"{input_directory} does not exist. Exiting without creating an output file.")
        sys.exit()

    # Efficiently filter files using glob
    matched_files = glob.glob(os.path.join(input_directory, "tens*.[hH][dD][fF5]*"))

    if not matched_files:
        print(f"No matching HDF files found in {input_directory}. Exiting without creating an output file.")
        sys.exit()

    with h5py.File(output_file, 'w') as output_hdf:
        # Use ThreadPoolExecutor to process multiple files simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(process_file, filepath, output_hdf, verbose) for filepath in matched_files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")


def copy_obj_to_file(name, obj, output_hdf, verbose):
    """Copy an object from the input HDF5 file to the output HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        # Copy the dataset to the output file in chunks
        chunk_size = min(1_000_000, obj.size)  # Example chunk size. Adjust as needed.
        for start_idx in range(0, obj.size, chunk_size):
            end_idx = start_idx + chunk_size
            output_hdf.require_dataset(name, shape=obj.shape, dtype=obj.dtype)[start_idx:end_idx] = obj[start_idx:end_idx]

        # Copy attributes from the input file's root to the dataset
        for attr_name, attr_value in obj.file.attrs.items():
            output_hdf[name].attrs[attr_name] = attr_value


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_hdf.py <input_directory> <output_file> <verbose>")
        sys.exit()

    input_directory = sys.argv[1]
    output_file = sys.argv[2]
    verbose = sys.argv[3].lower() == 'true'

    merge_hdf(input_directory, output_file, verbose)
