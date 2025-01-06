import os
import h5py

# Directory containing the HDF5 files
input_directory = "/Users/kkreth/PycharmProjects/data/DL-PTV/_combined"

# Name of the super file with external links
output_file = "/Users/kkreth/PycharmProjects/data/DL-PTV/combined_external_links.hd5"

# Create an HDF5 file for the super file
with h5py.File(output_file, "w") as superfile:
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            if filename.endswith(".h5"):
                file_path = os.path.join(root, filename)
                # Create an external link to each HDF5 file in the super file
                superfile.create_external_link("/", filename, file_path)

print("Combined external links file created:", output_file)
