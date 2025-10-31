import pandas as pd
import glob
import os
import json
from aeon.datasets import get_dataset_meta_data

# Path to results folders
base_path = "results"

# Time Box Plot:
csv_files = glob.glob(os.path.join(base_path, "*", "*_time_benchmark.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found. Check that 'results/*/*_time_benchmarking.csv' exists.")

table = ""

min_speedup = ("", float('inf'))
max_speedup = ("", float('-inf'))
avg_speedup = 0

min_mem_reduction = ("", float('inf'))
max_mem_reduction = ("", float('-inf'))
avg_mem_reduction = 0

count = 0

all_metadata = get_dataset_meta_data()
all_metadata = all_metadata[all_metadata["Channels"] == 1]

for i in range(150):
    if not os.path.exists(f"{base_path}/{i}/{i}_time_benchmark.csv"):
        print(f"Skipping {i}, no time csv")
        continue
    
    name = all_metadata.iloc[i]["Dataset"]
    with open(f"{base_path}/{i}/{i}_meta.json", 'r') as file:
        metadata = json.load(file)
        #print(metadata["problemname"])
    #print(name)
    
    n_samples = all_metadata.iloc[i]["TrainSize"] + all_metadata.iloc[i]["TestSize"]
    sample_length = all_metadata.iloc[i]["Length"]
    #print(f" n_samples: {n_samples}, sample_length: {sample_length}")
    
    df_time = pd.read_csv(f"{base_path}/{i}/{i}_time_benchmark.csv")
    df_memory = pd.read_csv(f"{base_path}/{i}/{i}_memory_benchmark.csv")
    

    # Drop unwanted augmenters
    df_time = df_time[~df_time["Augmenter"].isin(["fft", "ifft", "AmplitudePhasePerturbation", "Repeat", "Scaling", "Rotation", "FrequencyMask", "Permutate", "RandomTimeWarpAugmenter", "compare_within_tolerance"])]
    df_memory = df_memory[~df_memory["Augmenter"].isin(["fft", "ifft", "AmplitudePhasePerturbation", "Repeat", "Scaling", "Rotation", "FrequencyMask", "Permutate", "RandomTimeWarpAugmenter", "compare_within_tolerance"])]
    
    pipeline_time_RATS = df_time.loc[df_time["Augmenter"] == "Pipeline"]["RATSpy_time_sec"].values[0]
    pipeline_time_tsaug = df_time.loc[df_time["Augmenter"] == "Pipeline"]["tsaug_time_sec"].values[0]
    
    speedup = pipeline_time_tsaug / pipeline_time_RATS
    if speedup < min_speedup[1]:
        min_speedup = (name, speedup)
    if speedup > max_speedup[1]:
        max_speedup = (name, speedup)
    avg_speedup += speedup
    
    pipeline_memory_RATS = df_memory.loc[df_memory["Augmenter"] == "Pipeline"]["RATSpy_peak_mem_MB"].values[0]
    pipeline_memory_tsaug = df_memory.loc[df_memory["Augmenter"] == "Pipeline"]["tsaug_peak_mem_MB"].values[0]
    
    mem_reduction = pipeline_memory_RATS / pipeline_memory_tsaug
    if mem_reduction < min_mem_reduction[1]:
        min_mem_reduction = (name, mem_reduction)
    if mem_reduction > max_mem_reduction[1]:
        max_mem_reduction = (name, mem_reduction)
    avg_mem_reduction += mem_reduction
    
    table += f"{name} & {n_samples} & {sample_length} & {round(pipeline_time_RATS, 4)} & {round(pipeline_memory_RATS, 1)} & {round(pipeline_time_tsaug, 4)} & {round(pipeline_memory_tsaug, 1)} \\\\ \n"
    count += 1

avg_speedup /= count
avg_mem_reduction /= count

print(f"Speedup: min {min_speedup}, max {max_speedup}, avg {avg_speedup}")
print(f"Memory Reduction: min {min_mem_reduction}, max {max_mem_reduction}, avg {avg_mem_reduction}")
print(count)
#print(table)
