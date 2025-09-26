#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:10:00         # Run time of 15 minutes
#SBATCH --job-name=rats_test    # Sets the job name
#SBATCH --output output.txt

### Program Code
source ./env/bin/activate

mkdir -p results/144/

# Load dataset
python ./dataLoader.py --dataset-idx 144

# Run tests
python ./time_benchmarking.py --dataset-idx 144
python ./memory_benchmarking.py --dataset-idx 144

# Create time vs memory plots
python ./time_memory_plots.py --dataset-idx 144

rm -rf ./data/144
