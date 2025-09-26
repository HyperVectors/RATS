#!/usr/bin/zsh

### Job Parameters
#SBATCH --array=0-99
#SBATCH --partition c23ml
#SBATCH --output=stdout.txt
#SBATCH --time=00:05:00         # Run time of 15 minutes
#SBATCH --job-name=rats_test    # Sets the job name

### Program Code
source ./env/bin/activate

mkdir -p results/${SLURM_ARRAY_TASK_ID}/

# Load dataset
python ./dataLoader.py --dataset-idx ${SLURM_ARRAY_TASK_ID} >> ./results/${SLURM_ARRAY_TASK_ID}/dataLoader.log 2>&1

# Run tests
python ./time_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID} >> ./results/${SLURM_ARRAY_TASK_ID}/time.log 2>&1
python ./memory_benchmarking.py --dataset-idx ${SLURM_ARRAY_TASK_ID} >> ./results/${SLURM_ARRAY_TASK_ID}/memory.log 2>&1

# Create time vs memory plots
python ./time_memory_plots.py --dataset-idx ${SLURM_ARRAY_TASK_ID} >> ./results/${SLURM_ARRAY_TASK_ID}/plots.log 2>&1

rm -rf ./data/${SLURM_ARRAY_TASK_ID}/
