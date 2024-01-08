import pickle
import dask.array as da

def load_and_print_first_element(file_path):
    # Unpickle the Dask array
    with open(file_path, 'rb') as f:
        dask_array = pickle.load(f)

    # Check if the loaded object is a Dask array
    if not isinstance(dask_array, da.Array):
        print("Loaded object is not a Dask array.")
        return

    # Convert the Dask array to a NumPy array
    numpy_array = dask_array.compute()

    # Print the first element
    if numpy_array.size > 0:
        print("First element:", numpy_array[0])
    else:
        print("Array is empty.")

# Path to your pickled Dask array
pickle_file_path = '/Users/kkreth/PycharmProjects/data/DL-PTV/combined_external_links.dask'

# Load and print the first element
load_and_print_first_element(pickle_file_path)
