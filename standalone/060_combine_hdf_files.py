import os
import torch

# The directory containing your hdf files
directory = '/Users/kkreth/PycharmProjects/data/DL-PTV/'

def get_all_files(directory):
    files = []
    for root, dirs, file_names in os.walk(directory):
        for file_name in file_names:
            if file_name.endswith('tensors.hdf'):  # or h5
                files.append(os.path.join(root, file_name))
    return files

def combine_files(files):
    # Load and concatenate the tensors
    tensors = []
    for file_name in files:
        try:
            loaded_object = torch.load(file_name)

            # Debugging: print out the type of the loaded object
            #print(f'Loaded object from file {file_name} is of type: {type(loaded_object)}')

            if isinstance(loaded_object, torch.Tensor):
                tensors.append(loaded_object)
            elif isinstance(loaded_object, list) and all(isinstance(item, torch.Tensor) for item in loaded_object):
                # If the loaded object is a list of tensors, add each tensor to our list
                tensors.extend(loaded_object)
            else:
                print(f'Loaded object from file {file_name} is not a tensor or list of tensors. Skipping this file.')

        except Exception as e:
            print(f'Error reading file: {file_name}, Error: {str(e)}')
            continue

    combined = torch.cat(tensors, dim=0)  # adjust the axis as needed

    return combined



def save_to_hdf(tensor, output_file):
    torch.save(tensor, output_file)

output_file = '/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.hdf'  # replace with your output file

files = get_all_files(directory)
combined_tensor = combine_files(files)
save_to_hdf(combined_tensor, output_file)
