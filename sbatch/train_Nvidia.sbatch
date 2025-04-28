#!/bin/bash
#SBATCH --job-name=WAE_01
#SBATCH --partition=gpu                # Choose a correct partition, e.g., gpu, gpu-preempt, superpod-a100, etc.
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 21:55:00  # Job time limit
#SBATCH --mem=16G                      # Set as needed
#SBATCH -o /home/kkreth_umassd_edu/DL-PTV/logs/%x.%A.%a.out

# Your job commands here
module load cuda/12.6   # Example, load CUDA if needed


source /home/kkreth_umassd_edu/_virtual_python_3/bin/activate

# Set the directory path
export PYTHONPATH="${PYTHONPATH}:/home/kkreth_umassd_edu/cgan/:/home/kkreth_umassd_edu/cgan/encoder/saved_models:/home/kkreth_umassd_edu/cgan/configs/"
cd /home/kkreth_umassd_edu/cgan/


# Pass the input file as an argument to the Python script
/home/kkreth_umassd_edu/_virtual_python_3/bin/python /home/kkreth_umassd_edu/cgan/encoder/train_WAE_01.py \
    --resume_checkpoint /home/kkreth_umassd_edu/cgan/encoder/saved_models/WAE_From_OSX.pt

