import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import ratspy as rp
import importlib

plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 24,      # Axis label font size
    'xtick.labelsize': 20,     # X-axis tick label size
    'ytick.labelsize': 20,     # Y-axis tick label size
    'legend.fontsize': 22,     # Legend font size
    'figure.titlesize': 30     # Figure title size
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_rp_kwargs, load_data

# dataset name
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
)
args = parser.parse_args()
dataset_name = args.dataset

csv_path = f"../../../data/{dataset_name}/{dataset_name}.csv"
yaml_path = "../augmenter_configs.yaml"
sample_plot_dir = "results/qualitative_plots/sample_comparison"
meanstd_plot_dir = "results/qualitative_plots/mean_std_comparison"
tsaug_plot_dir = "results/qualitative_plots/tsaug_RATSpy_sample_comparison"
os.makedirs(tsaug_plot_dir, exist_ok=True)
os.makedirs(sample_plot_dir, exist_ok=True)
os.makedirs(meanstd_plot_dir, exist_ok=True)

# Loading data
X, y = load_data(csv_path)

# Loading augmenters from YAML
with open(yaml_path, "r") as f:
    augmenters = yaml.safe_load(f)

for aug in augmenters:
    aug_name = aug.get("name")
    rp_kwargs = fix_rp_kwargs(aug_name, aug.get("rp_kwargs") or {})
    if not rp_kwargs:
        continue
    if not hasattr(rp, aug_name):
        continue

    # Instantiating augmenter
    rp_aug_class = getattr(rp, aug_name)
    rp_aug = rp_aug_class(**rp_kwargs)

    # Applying augmentation
    ds = rp.Dataset(X.copy(), y.copy())
    rp_aug.augment_batch(ds, parallel=True)
    aug_X = ds.features

    # Plotting first sample comparison
    orig_sample = X[0]
    aug_sample = aug_X[0]
    plt.figure(figsize=(16, 10))  # Significantly increased size
    plt.plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
    plt.plot(aug_sample, label="Augmented", alpha=0.7, linewidth=3)
    plt.title(f"{dataset_name} dataset : {aug_name} - First Sample Comparison", fontsize=36, pad=30)
    plt.xlabel("Time Step", fontsize=28)
    plt.ylabel("Value", fontsize=28)
    x_ticks = range(0, len(orig_sample), 100)
    plt.xticks(x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(f"{sample_plot_dir}/{aug_name}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting mean and std deviation comparison
    orig_mean = X.mean(axis=0)
    orig_std = X.std(axis=0)
    aug_mean = aug_X.mean(axis=0)
    aug_std = aug_X.std(axis=0)

    plt.figure(figsize=(16, 10)) 
    plt.plot(orig_mean, label="Original Mean", linewidth=3)
    plt.fill_between(
        range(len(orig_mean)), orig_mean - orig_std, orig_mean + orig_std, alpha=0.2
    )
    plt.plot(aug_mean, label="Augmented Mean", linewidth=3)
    plt.fill_between(
        range(len(aug_mean)), aug_mean - aug_std, aug_mean + aug_std, alpha=0.2
    )
    plt.title(f"{dataset_name} dataset : {aug_name} - Mean and Std Comparison", fontsize=36, pad=30)
    plt.xlabel("Time Step", fontsize=28)
    plt.ylabel("Value", fontsize=28)
    x_ticks = range(0, len(orig_mean), 100)
    plt.xticks(x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(f"{meanstd_plot_dir}/{dataset_name}_{aug_name}_mean_std_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting first sample comparison for tsaug if available
    tsaug_class_name = aug.get("tsaug_class")
    tsaug_kwargs = aug.get("tsaug_kwargs") or {}
    if tsaug_class_name:
        try:
            tsaug_mod = importlib.import_module("tsaug")
            tsaug_class = getattr(tsaug_mod, tsaug_class_name)
            if tsaug_class_name == "Convolve" and isinstance(
                tsaug_kwargs.get("window", None), list
            ):
                tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])
            orig_sample = X[0]
            tsaug_aug_sample = tsaug_class(**tsaug_kwargs).augment(orig_sample)
            if hasattr(tsaug_aug_sample, "shape") and len(tsaug_aug_sample.shape) > 1:
                tsaug_aug_sample = tsaug_aug_sample[0]

            # Side-by-side plot
            fig, axs = plt.subplots(1, 2, figsize=(24, 10), sharey=True) 
            # RATSpy
            axs[0].plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
            axs[0].plot(aug_sample, label="RATSpy Augmented", alpha=0.7, linewidth=3)
            axs[0].set_title(f"{dataset_name} dataset : {aug_name} (RATSpy)", fontsize=36, pad=20)
            axs[0].set_xlabel("Time Step", fontsize=28)
            axs[0].set_ylabel("Value", fontsize=28)
            x_ticks = range(0, len(orig_sample), 100)
            axs[0].set_xticks(x_ticks)
            axs[0].tick_params(axis='both', which='major', labelsize=20)
            axs[0].legend(fontsize=22)
            # tsaug
            axs[1].plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
            axs[1].plot(tsaug_aug_sample, label="tsaug Augmented", alpha=0.7, linewidth=3)
            axs[1].set_title(f"{dataset_name} dataset : {aug_name} (tsaug)", fontsize=36, pad=20)
            axs[1].set_xlabel("Time Step", fontsize=28)
            axs[1].set_ylabel("Value", fontsize=28)
            axs[1].set_xticks(x_ticks)
            axs[1].tick_params(axis='both', which='major', labelsize=20)
            axs[1].legend(fontsize=22)
            plt.tight_layout()
            plt.savefig(f"{tsaug_plot_dir}/{dataset_name}_{aug_name}_vs_RATSpy.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Skipping tsaug plot for {aug_name}: {e}")
