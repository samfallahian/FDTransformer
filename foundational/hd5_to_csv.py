import h5py
import csv
import numpy as np
import os


def stream_hdf5_to_csv(input_hdf5_file, output_csv_file):
    print(f'Reading HDF5 file: {input_hdf5_file}')
    with h5py.File(input_hdf5_file, 'r') as hdf5_file:
        print(f'Creating CSV file: {output_csv_file}')
        with open(output_csv_file, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            header = ["Dataset Name"]
            for coordinate in range(1, 125):
                header.extend([
                    f"Coordinate_{coordinate}_X",
                    f"Coordinate_{coordinate}_Y",
                    f"Coordinate_{coordinate}_Z"
                ])
            csv_writer.writerow(header)

            for dataset_name in hdf5_file:
                print(f'\nProcessing dataset {dataset_name}')
                dataset = hdf5_file[dataset_name]
                row_data = [dataset_name]
                for coordinate in dataset:
                    flattened_coord = coordinate.flatten().tolist()
                    if len(flattened_coord) != 3:
                        print(f"Error at dataset {dataset_name}: Coordinate {coordinate} flattened to unexpected size {len(flattened_coord)}")
                    else:
                        row_data.extend(flattened_coord)
                csv_writer.writerow(row_data)
                print(f'Finished writing data for {dataset_name}')

    print('Finished writing CSV file')


if __name__ == "__main__":
    # Input directory, adjust this to your needs
    input_directory = "/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/"

    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if file is a .hd5 file
        if filename.endswith(".hd5"):
            hdf5_file_path = os.path.join(input_directory, filename)
            csv_file_name = filename.rsplit('.', 1)[0] + '.csv'  # Name of the .csv file is prefix of the .hd5 file
            output_csv_file = os.path.join(input_directory, csv_file_name)
            stream_hdf5_to_csv(hdf5_file_path, output_csv_file)
