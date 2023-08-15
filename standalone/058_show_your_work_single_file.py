import torch

# Specify the path to the saved new .torch file
new_file_path = "/Users/kkreth/PycharmProjects/data/DL-PTV/3p6/1_new_tensors.torch"

# Load the new data from file
loaded_new_data = torch.load(new_file_path)

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
