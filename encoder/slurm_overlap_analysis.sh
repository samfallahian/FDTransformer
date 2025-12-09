#!/bin/bash
#SBATCH --job-name=overlap_analysis
#SBATCH --output=/home/kkreth_umassd_edu/cgan/logs/overlap_%A_%a.out
#SBATCH --error=/home/kkreth_umassd_edu/cgan/logs/overlap_%A_%a.err
#SBATCH --array=1-1200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# This SLURM script processes one time step per array job
# Usage: sbatch encoder/slurm_overlap_analysis.sh
# The array index (1-1200) is automatically used as the time step
#
# For testing individual time steps directly:
# ./slurm_overlap_analysis.sh 100  (processes time step 100)

# IMPORTANT: Set project directory FIRST, before anything else
PROJECT_DIR="/home/kkreth_umassd_edu/cgan"

# Change to project directory immediately
cd "$PROJECT_DIR" || {
    echo "FATAL ERROR: Cannot change to project directory: $PROJECT_DIR"
    exit 1
}

# Configuration
DATASET="7p2"

# If run directly (not via sbatch), use command line argument
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    if [ -z "$1" ]; then
        echo "ERROR: When running directly (not via sbatch), you must provide a time step as argument"
        echo "Usage: $0 <time_step>"
        echo "Example: $0 100"
        exit 1
    fi
    TIME=$1
    echo "Running in TEST MODE (not submitted via SLURM)"
else
    TIME=$SLURM_ARRAY_TASK_ID
fi

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

# Load any required modules (uncomment and modify as needed)
# module load cuda/11.8
# module load python/3.10

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset: $DATASET"
echo "Processing time step: $TIME"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Step 1: Generate decoded velocity analysis
echo "[$(date)] Step 1/3: Running generate_decoded_velocity_analysis.py..."
python encoder/generate_decoded_velocity_analysis.py --dataset $DATASET --time $TIME
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: generate_decoded_velocity_analysis.py failed for time $TIME"
    exit 1
fi
echo "[$(date)] Step 1/3 completed successfully"

# Step 2: Analyze overlapping velocities
echo "[$(date)] Step 2/3: Running analysis_overlapping_velocities.py..."
python encoder/analysis_overlapping_velocities.py --dataset $DATASET --time $TIME
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: analysis_overlapping_velocities.py failed for time $TIME"
    exit 2
fi
echo "[$(date)] Step 2/3 completed successfully"

# Step 3: Mutual information analysis
echo "[$(date)] Step 3/3: Running analysis_overlapping_velocities_mutual_information.py..."
python encoder/analysis_overlapping_velocities_mutual_information.py --dataset $DATASET --time $TIME
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: analysis_overlapping_velocities_mutual_information.py failed for time $TIME"
    exit 3
fi
echo "[$(date)] Step 3/3 completed successfully"

echo "=========================================="
echo "All steps completed successfully for time $TIME"
echo "End time: $(date)"
echo "=========================================="

exit 0
