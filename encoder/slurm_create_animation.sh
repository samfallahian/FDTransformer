#!/bin/bash
#SBATCH --job-name=create_animation
#SBATCH --output=/home/kkreth_umassd_edu/cgan/logs/animation_%j.out
#SBATCH --error=/home/kkreth_umassd_edu/cgan/logs/animation_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# This SLURM script creates the animated HTML from all processed time steps
# Run this AFTER all array jobs complete
# Usage: sbatch encoder/slurm_create_animation.sh

# IMPORTANT: Set project directory FIRST, before anything else
PROJECT_DIR="/home/kkreth_umassd_edu/cgan"

# Change to project directory immediately
cd "$PROJECT_DIR" || {
    echo "FATAL ERROR: Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

# Configuration
DATASET="7p2"
START_TIME=1
END_TIME=1200

echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Current directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs || {
    echo "ERROR: Cannot create logs directory in $PROJECT_DIR"
    exit 1
}

# Activate virtual environment based on hostname
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"umass.edu"* ]] || [[ "$HOSTNAME" == *"login"* ]]; then
    # UMass cluster
    echo "Activating virtual environment for UMass cluster..."
    source /home/kkreth_umassd_edu/_virtual_python_3/bin/activate
else
    # Local or other environment - uncomment and modify as needed
    # source ~/miniconda3/etc/profile.d/conda.sh
    # conda activate your_environment_name
    echo "Running on: $HOSTNAME (no virtual environment activated)"
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Creating animated HTML for dataset: $DATASET"
echo "Time range: $START_TIME to $END_TIME"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Create animation using the skip-processing flag (data already exists)
# Downsample factor: 1 = full resolution (no downsampling)
# Only generating first 100 frames to reduce file size
python encoder/overlap_analysis_entire_experiment.py \
    --dataset $DATASET \
    --start $START_TIME \
    --end $END_TIME \
    --skip-processing \
    --downsample 1

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Failed to create animation"
    exit 1
fi

echo "=========================================="
echo "Animation created successfully!"
echo "End time: $(date)"
echo "=========================================="

exit 0
