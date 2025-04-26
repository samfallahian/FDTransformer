#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=16gb
#SBATCH --job-name=6p6
#SBATCH -t 21:55:00  # Job time limit
#SBATCH --array=1-1200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/kkreth_umassd_edu/DL-PTV/logs/%x.%A.%a.out

source /home/kkreth_umassd_edu/_virtual_python_3/bin/activate

# Set the directory path
export PYTHONPATH="${PYTHONPATH}:/home/kkreth_umassd_edu/cgan/2025/:/home/kkreth_umassd_edu/cgan/:/home/kkreth_umassd_edu/cgan/configs/"
cd /home/kkreth_umassd_edu/cgan/2025/

# Set the directory path using job name
directory_path="/home/kkreth_umassd_edu/DL-PTV/all_data_broken_down_1200_each_directory/${SLURM_JOB_NAME}"

# Calculate the file number from the task ID
file_number=$SLURM_ARRAY_TASK_ID

# Construct the input file path
file_1="${directory_path}/${file_number}.pkl"


# Pass the input file as an argument to the Python script
/home/kkreth_umassd_edu/_virtual_python_3/bin/python /home/kkreth_umassd_edu/cgan/2025/Ordered_100_iterateThrough.py \
    "$file_1"

