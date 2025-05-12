#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=8gb
#SBATCH --job-name=latent_space
#SBATCH -t 21:55:00  # Job time limit
#SBATCH --array=1-1200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /home/kkreth_umassd_edu/DL-PTV/logs/%x.%A.%a.out

source /home/kkreth_umassd_edu/_virtual_python_3/bin/activate

# Set the directory path
export PYTHONPATH="${PYTHONPATH}:/home/kkreth_umassd_edu/cgan/2025/:/home/kkreth_umassd_edu/cgan/:/home/kkreth_umassd_edu/cgan/configs/"
cd /home/kkreth_umassd_edu/cgan/

# Define all experiment names directly in the script
declare -a experiments=(
    "10p4"
    "11p4"
    "3p6"
    "4p4"
    "4p6"
    "5p2"
    "6p4"
    "6p6"
    "7p2"
    "7p8"
    "8p4"
)

# File number directly from SLURM_ARRAY_TASK_ID (1-1200)
file_number=$SLURM_ARRAY_TASK_ID

# Process all experiments for this file number
for EXPERIMENT_NAME in "${experiments[@]}"; do
    echo "Processing experiment: $EXPERIMENT_NAME, file: $file_number"

    # Run the Python script with the proper file path
    /home/kkreth_umassd_edu/_virtual_python_3/bin/python /home/kkreth_umassd_edu/cgan/Ordered_300_GenerateLatentSpace.py \
        --picklefile "/work/pi_bseyedaghazadeh_umassd_edu/DL-PTV/all_data_broken_down_1200_each_directory/${EXPERIMENT_NAME}/$file_number.pkl"
done