#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --array=0-189
#SBATCH --time=00:10:00         # Run time of 15 minutes
#SBATCH --job-name=rats_test    # Sets the job name

### Program Code
# Load dataset
python ./dataLoader.py --dataset-idx ${SLURM_ARRAY_TASK_ID}

# Run tests
python ./time_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID}
python ./memory_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID}

# Create time vs memory plots
python ./time_memory_plots.py --dataset-idx ${SLURM_ARRAY_TASK_ID}