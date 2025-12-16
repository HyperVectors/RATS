import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

plt.rcParams.update({'font.size': 16})

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
        value_name="Running Time (ms)"
    )
    
    tidy["Running Time (ms)"] *= 1000

    # Clean up Set names (remove "_time_sec")
    tidy["Set"] = tidy["Set"].str.replace("_time_sec", "", regex=False)

    all_data.append(tidy)

# Combine all datasets into one DataFrame
final_df = pd.concat(all_data, ignore_index=True)

print("Aggregated DataFrame shape:", final_df.shape)
print(final_df.head(10))

# Your plotting code
plt.figure(figsize=(14, 6))
ax = sns.boxplot(x="Augmenter", y="Running Time (ms)", hue="Set", data=final_df)
plt.xticks(rotation=45, ha="right")
plt.yscale("log")
# plt.title("Aggregated Benchmarking Runtime per Augmenter (All Datasets)")
plt.tight_layout()

# --- Extract the legend handles and labels BEFORE removing it ---
handles, labels = ax.get_legend_handles_labels()

# --- Remove the legend from the main figure ---
ax.legend_.remove()

# --- Save the main figure WITHOUT the legend ---
plt.savefig(os.path.join(base_path, "time_box_plot.eps"), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(base_path, "time_box_plot.pdf"), format='pdf', bbox_inches='tight')

# --- Create a new figure just for the legend ---
fig_legend = plt.figure(figsize=(4, 2))  # Adjust size as needed
fig_legend.legend(handles, labels, loc='center', frameon=True, ncol=2)
fig_legend.tight_layout()

# --- Save the legend separately ---
fig_legend.savefig(os.path.join(base_path, "box_plot_legend.pdf"), format='pdf', bbox_inches='tight')
fig_legend.savefig(os.path.join(base_path, "box_plot_legend.eps"), format='eps', bbox_inches='tight')

plt.close(fig_legend)
plt.close()


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
        value_name="Peak Memory (MB)"
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
ax = sns.boxplot(x="Augmenter", y="Peak Memory (MB)", hue="Set", data=final_df)
# sns.boxplot(x="Augmenter", y="Peak Memory (MB)", hue="Set", data=final_df)
plt.xticks(rotation=45, ha="right")
plt.yscale("log")
plt.ylim(100, 10000)
#plt.title("Aggregated Benchmarking Memory Usage per Augmenter (All Datasets)")
plt.tight_layout()

ax.legend_.remove()
# plt.show()
plt.savefig(os.path.join(base_path, "memory_box_plot.eps"), format='eps')
plt.savefig(os.path.join(base_path, "memory_box_plot.pdf"), format='pdf')