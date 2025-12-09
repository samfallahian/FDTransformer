    # SLURM Job Submission Guide for Overlap Analysis

## Overview

This guide explains how to run the complete overlap analysis pipeline (1200 time steps) on a SLURM cluster using array jobs for parallel processing.

## Files

1. **`slurm_overlap_analysis.sh`** - Main array job that processes individual time steps (1-1200)
2. **`slurm_create_animation.sh`** - Post-processing job that creates the animated HTML
3. **`overlap_analysis_entire_experiment.py`** - Python driver script (used by animation job)

## Step-by-Step Instructions

### Step 1: Prepare the Environment

Before submitting jobs, make sure:

1. Create the logs directory:
   ```bash
   mkdir -p logs
   ```

2. Edit `slurm_overlap_analysis.sh` to configure:
   - Your conda environment (if using one)
   - Required modules (CUDA, Python, etc.)
   - Partition name (currently set to `gpu`)
   - Memory/time requirements based on your cluster
   - Project directory path

3. Make scripts executable:
   ```bash
   chmod +x encoder/slurm_overlap_analysis.sh
   chmod +x encoder/slurm_create_animation.sh
   ```

### Step 2: Submit the Array Job

Submit 1200 parallel jobs (one per time step):

```bash
sbatch encoder/slurm_overlap_analysis.sh
```

This will create 1200 jobs with IDs like `12345_1`, `12345_2`, ..., `12345_1200`.

### Step 3: Monitor Progress

Check job status:
```bash
# View all your jobs
squeue -u $USER

# Count running jobs
squeue -u $USER -t RUNNING | wc -l

# Count pending jobs
squeue -u $USER -t PENDING | wc -l

# Check specific job array
squeue -j <JOB_ID>

# View detailed job info
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed
```

Check for failures:
```bash
# Find failed jobs (non-zero exit codes)
sacct -j <JOB_ID> --format=JobID,State,ExitCode | grep -v "0:0"

# Check error logs
tail logs/overlap_*_*.err

# Find which time steps failed
grep -l "ERROR" logs/overlap_*.err | sed 's/.*_\([0-9]*\)\.err/\1/'
```

### Step 4: Resubmit Failed Jobs (if needed)

If some jobs fail, resubmit specific array indices:

```bash
# Resubmit specific time steps (e.g., 5, 23, 147)
sbatch --array=5,23,147 encoder/slurm_overlap_analysis.sh

# Or resubmit a range
sbatch --array=100-150 encoder/slurm_overlap_analysis.sh
```

### Step 5: Create Animation

After ALL array jobs complete successfully, create the animated HTML:

```bash
sbatch encoder/slurm_create_animation.sh
```

Or run it directly if you prefer:
```bash
python encoder/overlap_analysis_entire_experiment.py \
    --dataset 7p2 \
    --start 1 \
    --end 1200 \
    --skip-processing
```

### Step 6: Download Results

The final output will be:
```
encoder/velocity_overlap_analysis/velocity_rmse_animation_7p2_complete.html
```

Download it to your local machine to view in a browser.

## Resource Requirements

Each time step job requires:
- **Memory**: 16GB (adjust based on your data size)
- **Time**: 2 hours (adjust based on testing)
- **GPU**: 1 GPU (for model inference)
- **CPUs**: 1 core

The animation job requires:
- **Memory**: 32GB (loading all RMSE data)
- **Time**: 1 hour
- **CPUs**: 4 cores (no GPU needed)

## Customization

### Different Dataset or Time Range

Edit the configuration variables in the scripts:

```bash
# In slurm_overlap_analysis.sh
DATASET="7p2"

# In slurm_create_animation.sh
DATASET="7p2"
START_TIME=1
END_TIME=1200

# Or submit with different array range:
sbatch --array=1-600 encoder/slurm_overlap_analysis.sh  # Only first 600 steps
```

### Different Partition or Resources

Edit the `#SBATCH` directives in the scripts:

```bash
#SBATCH --partition=your_partition_name
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1  # Specific GPU type
```

### Using Conda Environment

Uncomment and modify these lines in the scripts:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate your_environment_name
```

## Troubleshooting

### Job Immediately Fails

Check the error log:
```bash
tail -50 logs/overlap_<JOB_ID>_<TASK_ID>.err
```

Common issues:
- Python environment not activated
- Missing dependencies
- Wrong file paths
- Insufficient memory

### Job Times Out

Increase the time limit:
```bash
#SBATCH --time=04:00:00  # 4 hours instead of 2
```

### Out of Memory

Increase memory allocation:
```bash
#SBATCH --mem=32G  # 32GB instead of 16GB
```

### Missing Input Files

Check that time step files exist:
```bash
ls /Users/kkreth/PycharmProjects/data/all_data_ready_for_training/7p2/*.pkl | wc -l
```

Should show 1200 files (1.pkl through 1200.pkl).

## Estimated Runtime

- **Single time step**: ~10-30 minutes (depends on data size and GPU)
- **All 1200 steps in parallel**: 2-3 hours (if cluster has capacity)
- **Animation creation**: 20-30 minutes

Total wall-clock time: ~3-4 hours (if cluster has enough resources)

## Output Files

For each time step `t`, the following files are created:

1. **Decoded velocity data**:
   - `/Users/kkreth/PycharmProjects/data/overlap_analysis/7p2/df_decoded_velocity_<t>.pkl.gz`
   - `/Users/kkreth/PycharmProjects/data/overlap_analysis/7p2/df_position_mapping_<t>.pkl.gz`

2. **Analysis results**:
   - `encoder/velocity_overlap_analysis/rmse_per_position_7p2_<t>.csv`
   - `encoder/velocity_overlap_analysis/rmse_statistics_7p2_<t>.txt`
   - `encoder/velocity_overlap_analysis/velocity_3d_interactive_7p2_<t>.html`

3. **Mutual information analysis**:
   - `encoder/velocity_overlap_analysis/mi_correlation_report_7p2_<t>.txt`
   - `encoder/velocity_overlap_analysis/rmse_vs_sample_count_7p2_<t>.png`
   - `encoder/velocity_overlap_analysis/rmse_binned_analysis_7p2_<t>.png`

4. **Final animation**:
   - `encoder/velocity_overlap_analysis/velocity_rmse_animation_7p2_complete.html`

## Advanced: Job Dependencies

To automatically run animation after array completes:

```bash
# Submit array job and capture job ID
JOB_ID=$(sbatch --parsable encoder/slurm_overlap_analysis.sh)

# Submit animation job that waits for array to complete
sbatch --dependency=afterok:$JOB_ID encoder/slurm_create_animation.sh
```

Or wait for array to complete successfully:
```bash
JOB_ID=$(sbatch --parsable encoder/slurm_overlap_analysis.sh)
sbatch --dependency=aftercorr:$JOB_ID encoder/slurm_create_animation.sh
```
