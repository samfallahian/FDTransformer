import h5py
import csv
import numpy as np


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
    hdf5_file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.hd5"
    output_csv_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/_combined/4p6.csv"
    stream_hdf5_to_csv(hdf5_file_path, output_csv_file)
