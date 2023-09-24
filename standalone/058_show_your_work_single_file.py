import torch
import gzip
import io

# Specify the path to the saved new .torch.gz file
new_file_path = "/home/kkreth_umassd_edu/DL-PTV/11p4/11_tensor_for_transformer.torch.gz"
new_file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/4p4_sample/1000_tensor_for_transformer.torch.gz"

# Unzip and load the new data from file
with gzip.open(new_file_path, 'rb') as f:
    buffer = io.BytesIO(f.read())
loaded_new_data = torch.load(buffer)

# Check if there are at least 5 items in the loaded data
if len(loaded_new_data) < 5:
    print("There are less than 5 items in the loaded data.")
else:
    # Print out the first 5 items
    for i in range(5):
        print("Item", i + 1)
        print("Coordinates:", loaded_new_data[i]['coordinates'])
        print("Velocity:", loaded_new_data[i]['velocity'])
        print("Answer:", loaded_new_data[i]['answer'])
        print('---' * 20)
