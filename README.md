# PTV Project

### Requirements:
`pip install -r requirements.txt`



### Directory names:
`3p6-1` and `3p6-2` ---->  `0.05232 m/s`

`4p4-1` and `4p4-2` ---->  `0.06528 m/s`

`4p6-1` and `4p6-2` ---->  `0.06852 m/s`

`5p2-1` and `5p2-2` ---->  `0.07824 m/s`

`6p4-1` and `6p4-2` ---->  `0.09768 m/s`

`6p6-1` and `6p6-2` ---->  `0.10092 m/s`

`7p2-1` and `7p2-2` ---->  `0.11064 m/s`

`7p8-1` and `7p8-2` ---->  `0.12036 m/s`

`8p4-1` and `8p4-2` ---->  `0.13008 m/s`

`10p4-1` and `10p4-2` ---->  `0.16248 m/s`

`11p4-1` and `11p4-2` ---->  `0.17868 m/s`


## Submit a job to cluster (slurm)


### Usefull command for cluster  
`module avail`&nbsp;&nbsp;&nbsp;##List available modules  
`module load`&nbsp;&nbsp;&nbsp;##Load module named  
`module unload`&nbsp;&nbsp;&nbsp;##Unload module named  
`module whatis`&nbsp;&nbsp;&nbsp;##Give description of module  
`module list`&nbsp;&nbsp;&nbsp;##List modules that are loaded in your environment  
`module purge`&nbsp;&nbsp;&nbsp;##Unload all currently loaded modules from your environment  
`module display`&nbsp;&nbsp;&nbsp;##Give the rules for module  

To submit a job to the GPU partition:  

`#SBATCH --partition=GPU`&nbsp;&nbsp;&nbsp;# (Submits job to the GPU partition)  
To request 1 node, 8 CPU cores, and 4 GPUs, you would use the following syntax:
`#SBATCH --nodes=1`  
`#SBATCH --ntasks-per-node=8`  
`#SBATCH --gres=gpu:4`  
Request a particular type of GPU
You can specify a specific type of GPU, by model name. Currently we have 3 types of NVIDIA GPUs to choose from:  
* Titan V
* Titan RTX
* Tesla V100s  

You can specify the GPU model by modifying the "gres" directive, like so:  
`#SBATCH --gres=gpu:TitanV:4`&nbsp;&nbsp;&nbsp;#  will reserve 4 Titan V GPUs (8 Titan Vs is the max per node)  
`#SBATCH --gres=gpu:TitanRTX:2`&nbsp;&nbsp;&nbsp;#  (will reserve 2 Titan RTX GPUs (4 Titan RTXs is the max per node)  
`#SBATCH --gres=gpu:V100S:1`&nbsp;&nbsp;&nbsp; #  (will reserve 1 Tesla V100s GPU (4 Tesla V100s is the max per node)  
Submitting a Job  
Once you are satisfied with the contents of your submit script, save it, then submit it to the Slurm Workload Manager. Here are some helpful commands to do so:  
`sbatch submit-script.slurm`&nbsp;&nbsp;&nbsp;Submit Your Job  
`sbatch submit-script.slurm`&nbsp;&nbsp;&nbsp;Check the Queue  
`scontrol show job -d [job-id]`&nbsp;&nbsp;&nbsp;Show a Job's Detail  
`scancel [job-id]`&nbsp;&nbsp;&nbsp;  Cancel a Job  