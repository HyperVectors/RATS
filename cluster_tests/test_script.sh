#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:10:00         # Run time of 15 minutes
#SBATCH --job-name=rats_test    # Sets the job name

### Program Code
# Load dataset
python ./dataLoader.py --dataset-idx 1

# Run tests
python ./time_benchmarking.py --dataset-idx 1
python ./memory_benchmarking.py --dataset-idx 1

# Create time vs memory plots
python ./time_memory_plots.py --dataset-idx 1