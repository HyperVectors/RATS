import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Get dataset name from CLI argument, default to 'Car'
dataset_name = sys.argv[1] if len(sys.argv) > 1 else "Car"

# Derive file paths
data_dir = os.path.join("..", "data", dataset_name)
original_file = os.path.join(data_dir, f"{dataset_name}.csv")
augmented_file = os.path.join(data_dir, f"{dataset_name}_ifft.csv")

# Load datasets
df_orig = pd.read_csv(original_file)
df_aug = pd.read_csv(augmented_file)

df_orig = df_orig.iloc[:,:-1]
df_aug = df_aug.iloc[:,:-1]

# minimum number of columns
min_cols = min(df_orig.shape[1], df_aug.shape[1])

# Plot the first time series (up to min_cols)
plt.figure(figsize=(12, 6))
plt.plot(df_orig.iloc[0, :], label='Original', alpha=1.0)
plt.plot(df_aug.iloc[0, :], label='Augmented', alpha=0.5)
#plt.title(f'Comparison of Original vs Augmented Time Series (First Sample) - {dataset_name}')
#plt.xlabel('Time Step')
plt.xticks([], [])
#plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(data_dir, "comparison_first_sample.png"))

# Compute mean and std up to min_cols
mean_orig = df_orig.iloc[:, :min_cols].mean(axis=0)
std_orig = df_orig.iloc[:, :min_cols].std(axis=0)
mean_aug = df_aug.iloc[:, :min_cols].mean(axis=0)
std_aug = df_aug.iloc[:, :min_cols].std(axis=0)

plt.figure(figsize=(12, 6))
plt.plot(mean_orig, label='Original Mean')
plt.fill_between(range(len(mean_orig)), mean_orig-std_orig, mean_orig+std_orig, alpha=0.2)
plt.plot(mean_aug, label='Augmented Mean')
plt.fill_between(range(len(mean_aug)), mean_aug-std_aug, mean_aug+std_aug, alpha=0.2)
plt.title(f'Mean and Std of Original vs Augmented - {dataset_name}')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(data_dir, "mean_std_comparison.png"))


n = 10  # Plot every nth sample

# Heatmap for original data
plt.figure(figsize=(12, 6))
sns.heatmap(df_orig.iloc[::n, :min_cols], cmap="viridis", cbar=True)
plt.title(f'Original Data Heatmap (every {n}th sample) - {dataset_name}')
plt.xlabel('Time Step')
plt.ylabel('Sample')
plt.savefig(os.path.join(data_dir, f"original_heatmap.png"))

# Heatmap for augmented data
plt.figure(figsize=(12, 6))
sns.heatmap(df_aug.iloc[::n, :min_cols], cmap="viridis", cbar=True)
plt.title(f'Augmented Data Heatmap (every {n}th sample) - {dataset_name}')
plt.xlabel('Time Step')
plt.ylabel('Sample')
plt.savefig(os.path.join(data_dir, f"augmented_heatmap.png"))