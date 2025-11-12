#!/usr/bin/env zsh

# This script schedules four SLURM jobs with 2-hour intervals.

#echo "Waiting 2 hours for first submission..."
#sleep 7200
echo "Submitting populate_all_cubes_6p6.sbatch"
sbatch /home/kkreth_umassd_edu/cgan/2025/sbatch/populate_all_cubes_6p6.sbatch

echo "Waiting 2 hours for next submission..."
sleep 9200
echo "Submitting populate_all_cubes_7p2.sbatch"
sbatch /home/kkreth_umassd_edu/cgan/2025/sbatch/populate_all_cubes_7p2.sbatch

echo "Waiting 2 hours for next submission..."
sleep 9200
echo "Submitting populate_all_cubes_7p8.sbatch"
sbatch /home/kkreth_umassd_edu/cgan/2025/sbatch/populate_all_cubes_7p8.sbatch

echo "Waiting 2 hours for final submission..."
sleep 9200
echo "Submitting populate_all_cubes_8p4.sbatch"
sbatch /home/kkreth_umassd_edu/cgan/2025/sbatch/populate_all_cubes_8p4.sbatch

echo "All jobs submitted with 2-hour intervals!"