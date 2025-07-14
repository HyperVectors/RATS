import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import pyfraug as pf
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_pf_kwargs

csv_path = "../../../data/Car/Car.csv"
yaml_path = "../augmenters.yaml"
plot_dir = "results/qualitative_plots"
os.makedirs(plot_dir, exist_ok=True)

# Loading data
df = pd.read_csv(csv_path)
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].astype(str).tolist()

# Loading augmenters from YAML
with open(yaml_path, "r") as f:
    augmenters = yaml.safe_load(f)

for aug in augmenters:
    aug_name = aug.get("name")
    pf_kwargs = pf_kwargs = fix_pf_kwargs(aug_name, aug.get("pf_kwargs") or {})
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
    plt.title(f"{aug_name} - First Sample Comparison")
    plt.xticks([], [])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{aug_name}_comparison.png")
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
    plt.title(f"{aug_name} - Mean and Std Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{aug_name}_mean_std_comparison.png")
    plt.close()
