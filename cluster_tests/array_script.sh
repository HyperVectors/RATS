#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --array=100-149
#SBATCH --output=stdout.txt
#SBATCH --time=00:05:00         # Run time of 15 minutes
#SBATCH --job-name=rats_test    # Sets the job name

### Program Code
source ./env/bin/activate
# Load dataset
python ./dataLoader.py --dataset-idx ${SLURM_ARRAY_TASK_ID}

# Run tests
python ./time_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID}
python ./memory_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID}

# Create time vs memory plots
python ./time_memory_plots.py --dataset-idx ${SLURM_ARRAY_TASK_ID}

rm -rf ./data/${SLURM_ARRAY_TASK_ID}/
