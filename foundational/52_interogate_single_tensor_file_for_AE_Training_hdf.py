import h5py
import pandas as pd

# Set pandas display option to show all columns
pd.set_option('display.max_columns', None)


def print_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            data = obj[...]
            reshaped_data = data.reshape(-1, data.shape[-1])  # Reshape the data to 2D
            df = pd.DataFrame(reshaped_data)

            # If columns attribute is available in the dataset, set DataFrame columns
            if 'columns' in obj.attrs:
                df.columns = [str(col) for col in obj.attrs['columns']]

            print(f"Dataset Name: {name}")
            # Print surrogate_coordinate attributes if available
            for key in obj.attrs.keys():
                if 'surrogate_coordinate_' in key:
                    print(f"{key}: {obj.attrs[key]}")

            print(df.head(125))
            print()

        except Exception as e:
            print(f"Could not print dataset {name} due to {str(e)}")


def explore_hdf5(file_name):
    with h5py.File(file_name, 'r') as file:
        file.visititems(print_dataset)


file_name = '/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1_tensors.hdf'
file_name = '/home/kkreth_umassd_edu/DL-PTV/3p6/tensor_111.hdf'
#file_name = '/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/111.hdf'
explore_hdf5(file_name)


'''
Dataset Name: 3p6_111_97.0_8.0_-17.0
           vx        vy        vz
0    0.487013  0.426407  0.422078
1    0.484848  0.437229  0.422078
2    0.482684  0.445887  0.419913
3    0.480519  0.452381  0.415584
4    0.478355  0.452381  0.411255
..        ...       ...       ...
120  0.515152  0.437229  0.428571
121  0.512987  0.443723  0.430736
122  0.508658  0.450216  0.432900
123  0.502165  0.452381  0.430736
124  0.495671  0.454545  0.424242
'''