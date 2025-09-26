import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Path to results folders
base_path = "results"

# Time Box Plot:
csv_files = glob.glob(os.path.join(base_path, "*", "*_time_benchmark.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found. Check that 'results/*/*_time_benchmarking.csv' exists.")

all_data = []

for file in csv_files:
    df = pd.read_csv(file)

    # Drop unwanted augmenters
    df = df[~df["Augmenter"].isin(["fft", "ifft", "AmplitudePhasePerturbation", "Repeat", "Scaling", "Rotation", "FrequencyMask", "Permutate", "RandomTimeWarpAugmenter", "Pipeline", "compare_within_tolerance"])]

    # Convert from wide to long format
    tidy = df.melt(
        id_vars=["Augmenter"],
        value_vars=["RATSpy_time_sec", "tsaug_time_sec"],
        var_name="Set",
        value_name="Values"
    )

    # Clean up Set names (remove "_time_sec")
    tidy["Set"] = tidy["Set"].str.replace("_time_sec", "", regex=False)

    all_data.append(tidy)

# Combine all datasets into one DataFrame
final_df = pd.concat(all_data, ignore_index=True)

print("Aggregated DataFrame shape:", final_df.shape)
print(final_df.head(10))

# ---- Plot aggregated results ----
plt.figure(figsize=(14, 6))
sns.boxplot(x="Augmenter", y="Values", hue="Set", data=final_df)
plt.xticks(rotation=45, ha="right")
plt.yscale("log")
plt.title("Aggregated Benchmarking Time per Augmenter (All Datasets)")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(base_path, "time_box_plot.png"))




# Memory Box Plot
csv_files = glob.glob(os.path.join(base_path, "*", "*_memory_benchmark.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found. Check that 'results/*/*_memory_benchmarking.csv' exists.")

all_data = []

for file in csv_files:
    df = pd.read_csv(file)

    # Drop unwanted augmenters
    df = df[~df["Augmenter"].isin(["fft", "ifft", "AmplitudePhasePerturbation", "Repeat", "Scaling", "Rotation", "FrequencyMask", "Permutate", "RandomTimeWarpAugmenter", "Pipeline", "compare_within_tolerance"])]

    # Convert from wide to long format
    tidy = df.melt(
        id_vars=["Augmenter"],
        value_vars=["RATSpy_peak_mem_MB", "tsaug_peak_mem_MB"],
        var_name="Set",
        value_name="Values"
    )

    # Clean up Set names (remove "_time_sec")
    tidy["Set"] = tidy["Set"].str.replace("_peak_mem_MB", "", regex=False)

    all_data.append(tidy)

# Combine all datasets into one DataFrame
final_df = pd.concat(all_data, ignore_index=True)

print("Aggregated DataFrame shape:", final_df.shape)
print(final_df.head(10))

# ---- Plot aggregated results ----
plt.figure(figsize=(14, 6))
sns.boxplot(x="Augmenter", y="Values", hue="Set", data=final_df)
plt.xticks(rotation=45, ha="right")
plt.yscale("log")
plt.title("Aggregated Benchmarking Memory Usage per Augmenter (All Datasets)")
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(base_path, "memory_box_plot.png"))