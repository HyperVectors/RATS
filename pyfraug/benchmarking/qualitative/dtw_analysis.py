import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyfraug as pf
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from utils import fix_pf_kwargs, load_data
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Benchmark PyFraug and tsaug augmenters and transforms."
)
parser.add_argument(
    "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
)
args = parser.parse_args()
dataset_name = args.dataset

csv_path = f"../../../examples/{dataset_name}/{dataset_name}.csv"
print(f"Loading data from {csv_path}")

with open("../augmenters.yaml", "r") as f:
    AUGMENTERS = yaml.safe_load(f)


x, y = load_data(csv_path)

for aug in tqdm(AUGMENTERS, desc="Augmenters"):
    dataset = pf.Dataset(x, y)
    original_dataset = pf.Dataset(
        x.copy(), y.copy()
    )  # Load original dataset for comparison as a reference

    aug_name = aug["name"]
    pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})

    pf_aug_class = getattr(pf, aug_name)
    pf_aug = pf_aug_class(**pf_kwargs)
    pf_aug.augment_batch(dataset, parallel=True)
    aug_X = dataset.features

    # Computing dtw of the first sample with reference
    orig_sample = x[0]
    aug_sample = aug_X[0]
    dtw_distance, optimum_path = pf.QualityBenchmarking.compute_dtw(
        orig_sample, aug_sample
    )

    # Plotting the DTW path
    plt.figure(figsize=(12, 6))
    plt.plot(orig_sample, label="Original", alpha=1.0, color="black")
    plt.plot(aug_sample, label="Augmented", alpha=0.7, color="orange")
    for i, j in optimum_path:
        plt.plot([i, j], [orig_sample[i], aug_sample[j]], alpha=0.5)
    plt.title(
        f"{dataset_name} dataset : {aug_name} - DTW Alignment : Distance = {dtw_distance:.2f}"
    )
    plt.show()
