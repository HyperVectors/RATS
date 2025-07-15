import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import pyfraug as pf
import importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_pf_kwargs, load_data

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
tsaug_plot_dir = "results/qualitative_plots/tsaug_pyfraug_sample_comparison"
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
    pf_kwargs = fix_pf_kwargs(aug_name, aug.get("pf_kwargs") or {})
    if not pf_kwargs:
        continue
    if not hasattr(pf, aug_name):
        continue

    # Instantiating augmenter
    pf_aug_class = getattr(pf, aug_name)
    pf_aug = pf_aug_class(**pf_kwargs)

    # Applying augmentation
    ds = pf.Dataset(X.copy(), y.copy())
    pf_aug.augment_batch(ds, parallel=True)
    aug_X = ds.features

    # Plotting first sample comparison
    orig_sample = X[0]
    aug_sample = aug_X[0]
    plt.figure(figsize=(12, 6))
    plt.plot(orig_sample, label="Original", alpha=1.0)
    plt.plot(aug_sample, label="Augmented", alpha=0.7)
    plt.title(f"{dataset_name} dataset : {aug_name} - First Sample Comparison")
    plt.xticks([], [])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{sample_plot_dir}/{dataset_name}_{aug_name}_comparison.png")
    plt.close()

    # Plotting mean and std deviation comparison
    orig_mean = X.mean(axis=0)
    orig_std = X.std(axis=0)
    aug_mean = aug_X.mean(axis=0)
    aug_std = aug_X.std(axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(orig_mean, label="Original Mean")
    plt.fill_between(
        range(len(orig_mean)), orig_mean - orig_std, orig_mean + orig_std, alpha=0.2
    )
    plt.plot(aug_mean, label="Augmented Mean")
    plt.fill_between(
        range(len(aug_mean)), aug_mean - aug_std, aug_mean + aug_std, alpha=0.2
    )
    plt.title(f"{dataset_name} dataset : {aug_name} - Mean and Std Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{meanstd_plot_dir}/{dataset_name}_{aug_name}_mean_std_comparison.png")
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
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            # PyFraug
            axs[0].plot(orig_sample, label="Original", alpha=1.0)
            axs[0].plot(aug_sample, label="PyFraug Augmented", alpha=0.7)
            axs[0].set_title(f"{dataset_name}: {aug_name} (PyFraug)")
            axs[0].legend()
            axs[0].set_xticks([])
            # tsaug
            axs[1].plot(orig_sample, label="Original", alpha=1.0)
            axs[1].plot(tsaug_aug_sample, label="tsaug Augmented", alpha=0.7)
            axs[1].set_title(f"{dataset_name}: {aug_name} (tsaug)")
            axs[1].legend()
            axs[1].set_xticks([])
            plt.tight_layout()
            plt.savefig(f"{tsaug_plot_dir}/{dataset_name}_{aug_name}_vs_pyfraug.png")
            plt.close()
        except Exception as e:
            print(f"Skipping tsaug plot for {aug_name}: {e}")
