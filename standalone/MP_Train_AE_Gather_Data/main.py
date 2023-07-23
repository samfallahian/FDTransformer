import json
import os
from multiprocessing import Pool, cpu_count
from file_processing import process_file
import multiprocessing as mp

# Set multiprocessing start method to 'fork'
mp.set_start_method('fork')

# read in the meta-data file
with open("/home/kkreth_umassd_edu/Umass_experiments.txt", 'r') as f:
    experiment_dict = json.load(f)

# Get the list of .pkl files in the directory
file_list = [filename for filename in os.listdir('/home/kkreth_umassd_edu/DL-PTV') if filename.endswith('.pkl')]

# Create a multiprocessing Pool
pool = Pool(processes=cpu_count())

# Process the files in parallel
pool.starmap(process_file, [(filename, experiment_dict) for filename in file_list])

# Close the multiprocessing Pool
pool.close()
pool.join()
