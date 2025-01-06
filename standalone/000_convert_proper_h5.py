import h5py
import torch
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

DATA_PATH = "/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data.hdf"
NEW_FILE = "/Users/kkreth/PycharmProjects/data/DL-PTV-TrainingData/AE_training_data_corrected.hdf"

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

data_list = torch.load(DATA_PATH).to(device)
total_tensors = len(data_list)

# Global last logged percentage
last_logged_percentage = -0.1

def store_tensors(start, end, data_subset, dset):
    global last_logged_percentage

    for i in range(start, end):
        tensor = data_subset[i - start]
        tensor_cpu = tensor.to("cpu")
        dset[i] = tensor_cpu.numpy()

        # Calculate completion percentage using 'end'
        percentage = (end / total_tensors) * 100

        # If the change is more than or equal to 0.1% from the last logged value, log it
        if percentage - last_logged_percentage >= 0.1:
            logging.info(f"{end} / {total_tensors} ({percentage:.1f}%)")
            last_logged_percentage = percentage

with h5py.File(NEW_FILE, 'w') as f:
    dset = f.create_dataset('my_data', (len(data_list),) + data_list[0].shape, dtype='f', compression="lzf")

    # Number of threads
    n_threads = 100
    chunk_size = len(data_list) // n_threads

    # Use ThreadPoolExecutor to parallelize the task
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(0, len(data_list), chunk_size):
            start = i
            end = i + chunk_size if i + chunk_size < len(data_list) else len(data_list)
            futures.append(executor.submit(store_tensors, start, end, data_list[start:end], dset))

        # Waiting for all futures to finish
        for future in futures:
            future.result()
