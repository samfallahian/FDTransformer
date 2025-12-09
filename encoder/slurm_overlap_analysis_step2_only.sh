#!/bin/bash
#SBATCH --job-name=overlap_step2
#SBATCH --output=/home/kkreth_umassd_edu/cgan/logs/overlap_step2_%A_%a.out
#SBATCH --error=/home/kkreth_umassd_edu/cgan/logs/overlap_step2_%A_%a.err
#SBATCH --array=1-1200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# This script runs ONLY Steps 2 & 3 (skips Step 1 which already completed)
# Usage: sbatch encoder/slurm_overlap_analysis_step2_only.sh

# IMPORTANT: Set project directory FIRST
PROJECT_DIR="/home/kkreth_umassd_edu/cgan"

# Change to project directory immediately
cd "$PROJECT_DIR" || {
    echo "FATAL ERROR: Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

# Configuration
DATASET="7p2"
TIME=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Current directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Processing dataset: $DATASET"
echo "Processing time step: $TIME"
echo "Start time: $(date)"
echo "=========================================="

# Activate virtual environment
echo "Activating virtual environment..."
source /home/kkreth_umassd_edu/_virtual_python_3/bin/activate

# Step 2: Analyze overlapping velocities
echo "[$(date)] Step 2/2: Running analysis_overlapping_velocities.py..."
python encoder/analysis_overlapping_velocities.py --dataset $DATASET --time $TIME
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: analysis_overlapping_velocities.py failed for time $TIME"
    exit 1
fi
echo "[$(date)] Step 2/2 completed successfully"

# Step 3: Mutual information analysis
echo "[$(date)] Step 3/3: Running analysis_overlapping_velocities_mutual_information.py..."
python encoder/analysis_overlapping_velocities_mutual_information.py --dataset $DATASET --time $TIME
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: analysis_overlapping_velocities_mutual_information.py failed for time $TIME"
    exit 2
fi
echo "[$(date)] Step 3/3 completed successfully"

echo "=========================================="
echo "All steps completed successfully for time $TIME"
echo "End time: $(date)"
echo "=========================================="

exit 0
