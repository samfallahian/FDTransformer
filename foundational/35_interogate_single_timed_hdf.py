import h5py
import pandas as pd

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)


def print_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[...]
            df = pd.DataFrame(data)

            # If columns attribute is available in the dataset, set DataFrame columns
            if 'columns' in obj.attrs:
                df.columns = [str(col) for col in obj.attrs['columns']]

            print(f"Dataset Name: {name}")
            print(df.head(25))
            print()

        except Exception as e:
            print(f"Could not print dataset {name} due to {str(e)}")


def explore_hdf5(file_name):
    with h5py.File(file_name, 'r') as file:
        file.visititems(print_dataset)


file_name = '/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1.hdf'
explore_hdf5(file_name)
