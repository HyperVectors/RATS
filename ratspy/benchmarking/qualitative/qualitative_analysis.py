import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml
import ratspy as rp
import importlib

plt.rcParams.update({
    'font.size': 24,           
    'axes.titlesize': 30,      
    'axes.labelsize': 24,      
    'xtick.labelsize': 45,     
    'ytick.labelsize': 45,     
    'legend.fontsize': 26,     
    'figure.titlesize': 32     
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

# Directory for standalone legend strip
legend_dir = "results/qualitative_plots/legend"
os.makedirs(legend_dir, exist_ok=True)
legend_saved = False


def save_rotation_legend_strip() -> None:
    fig = plt.figure(figsize=(16, 2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="C0", lw=3, label="Original"),
        Line2D([0], [0], color="C1", lw=3, label="Augmented"),
    ]
    ax.legend(handles=handles, loc="center", ncol=2, fontsize=50, frameon=True)
    fig.savefig(os.path.join(legend_dir, "rotation_legend_strip.eps"), bbox_inches="tight")
    fig.savefig(os.path.join(legend_dir, "rotation_legend_strip.pdf"), bbox_inches="tight")
    plt.close(fig)

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
    plt.figure(figsize=(16, 12))
    plt.plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
    plt.plot(aug_sample, label="Augmented", alpha=0.7, linewidth=3)
    plt.title(f"{aug_name}", fontsize=54, pad=30)
    plt.xlabel("Time Step", fontsize=50)
    plt.ylabel("Value", fontsize=50)
    x_ticks = range(0, len(orig_sample), 100)
    plt.xticks(x_ticks, fontsize=45)
    plt.yticks(fontsize=45)
    # For rotation, omit legend on the plot; generate standalone legend once
    if aug_name == "Rotation" and not legend_saved:
        save_rotation_legend_strip()
        legend_saved = True
    plt.tight_layout()
    plt.savefig(f"{sample_plot_dir}/{aug_name}_comparison.eps")
    plt.savefig(f"{sample_plot_dir}/{aug_name}_comparison.pdf")
    plt.close()

    # Plotting mean and std deviation comparison
    orig_mean = X.mean(axis=0)
    orig_std = X.std(axis=0)
    aug_mean = aug_X.mean(axis=0)
    aug_std = aug_X.std(axis=0)

    plt.figure(figsize=(16, 12)) 
    plt.plot(orig_mean, label="Original Mean", linewidth=3)
    plt.fill_between(
        range(len(orig_mean)), orig_mean - orig_std, orig_mean + orig_std, alpha=0.2
    )
    plt.plot(aug_mean, label="Augmented Mean", linewidth=3)
    plt.fill_between(
        range(len(aug_mean)), aug_mean - aug_std, aug_mean + aug_std, alpha=0.2
    )
    plt.title(f"{dataset_name} dataset : {aug_name} - Mean and Std Comparison", fontsize=45, pad=30)
    plt.xlabel("Time Step", fontsize=45)
    plt.ylabel("Value", fontsize=45)
    x_ticks = range(0, len(orig_mean), 100)
    plt.xticks(x_ticks, fontsize=45)
    plt.yticks(fontsize=45)
    # For rotation, omit legend on the plot; generate standalone legend once
    if aug_name == "Rotation" and not legend_saved:
        save_rotation_legend_strip()
        legend_saved = True
    plt.tick_params(axis='both', which='major', labelsize=45)
    plt.tight_layout()
    plt.savefig(f"{meanstd_plot_dir}/{dataset_name}_{aug_name}_mean_std_comparison.eps")
    plt.savefig(f"{meanstd_plot_dir}/{dataset_name}_{aug_name}_mean_std_comparison.pdf")
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
            fig, axs = plt.subplots(1, 2, figsize=(24, 12), sharey=True) 
            # RATSpy
            axs[0].plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
            axs[0].plot(aug_sample, label="RATSpy Augmented", alpha=0.7, linewidth=3)
            axs[0].set_title(f"{aug_name} (RATSpy)", fontsize=45, pad=20)
            axs[0].set_xlabel("Time Step", fontsize=45)
            axs[0].set_ylabel("Value", fontsize=45)
            x_ticks = range(0, len(orig_sample), 100)
            axs[0].set_xticks(x_ticks)
            axs[0].tick_params(axis='both', which='major', labelsize=45)
            # For rotation, omit legend on the subplots; generate standalone legend once
            if aug_name == "Rotation" and not legend_saved:
                save_rotation_legend_strip()
                legend_saved = True
            # tsaug
            axs[1].plot(orig_sample, label="Original", alpha=1.0, linewidth=3)
            axs[1].plot(tsaug_aug_sample, label="tsaug Augmented", alpha=0.7, linewidth=3)
            axs[1].set_title(f"{aug_name} (tsaug)", fontsize=45, pad=20)
            axs[1].set_xlabel("Time Step", fontsize=45)
            # axs[1].set_ylabel("Value", fontsize=45)
            axs[1].set_xticks(x_ticks)
            axs[1].tick_params(axis='both', which='major', labelsize=45)
            fig.tight_layout()
            plt.savefig(f"{tsaug_plot_dir}/{dataset_name}_{aug_name}_vs_RATSpy.eps")
            plt.savefig(f"{tsaug_plot_dir}/{dataset_name}_{aug_name}_vs_RATSpy.pdf")
            plt.close()
        except Exception as e:
            print(f"Skipping tsaug plot for {aug_name}: {e}")
