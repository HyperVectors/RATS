"""
This script benchmarks RATSpy augmenters by applying them to a dataset
and computes the Dynamic Time Warping (DTW) distance between the original
and augmented samples. It visualizes the DTW alignment path for each augmenter.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ratspy as rp
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from utils import fix_rp_kwargs, load_data
import yaml
from tqdm import tqdm
import pathlib

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 55,
    'axes.labelsize': 50,
    'xtick.labelsize': 45,
    'ytick.labelsize': 45,
    'legend.fontsize': 30,
    'figure.titlesize': 55
})


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RATSpy and tsaug augmenters and transforms."
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default="Car",
        help="Dataset name (default: Car)",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default="./dtw_plots",
        help="Directory to save DTW plots",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    csv_path = pathlib.Path(f"../../../data/{dataset_name}/{dataset_name}.csv")
    print(f"Loading data from {csv_path}")

    with open("../augmenter_configs.yaml", "r") as f:
        AUGMENTERS = yaml.safe_load(f)

    x, y = load_data(csv_path)

    for aug in tqdm(AUGMENTERS, desc="Augmenters"):
        dataset = rp.Dataset(x, y)
        original_dataset = rp.Dataset(
            x.copy(), y.copy()
        )  # Load original dataset for comparison as a reference

        aug_name = aug["name"]
        rp_kwargs = fix_rp_kwargs(aug_name, aug["rp_kwargs"] or {})

        rp_aug_class = getattr(rp, aug_name)
        rp_aug = rp_aug_class(**rp_kwargs)
        rp_aug.augment_batch(dataset, parallel=True)
        aug_X = dataset.features

        # Computing dtw of the first sample with reference
        orig_sample = x[0]
        aug_sample = aug_X[0]
        dtw_distance, optimum_path = rp.QualityBenchmarking.compute_dtw(
            orig_sample, aug_sample
        )

        # Plotting the DTW path
        save_dir = pathlib.Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(24, 12))
        plt.plot(orig_sample, label="Original", alpha=1.0, color="blue")
        plt.plot(aug_sample, label="Augmented", alpha=0.7, color="orange")
        for i, j in optimum_path:
            plt.plot([i, j], [orig_sample[i], aug_sample[j]], alpha=0.5)
        plt.title(
            f"{aug_name} - Similarity = {(1 - abs(dtw_distance))*100:.2f}%",
            fontsize=55, pad=30
        )
        plt.xlabel("Time", fontsize=50)
        plt.ylabel("Value", fontsize=50)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        plt.tick_params(axis='both', which='major', labelsize=45)
        # No legend
        plt.tight_layout()
        plt.savefig(save_dir / f"dtw_{dataset_name}_{aug_name}.eps", bbox_inches='tight')
        plt.savefig(save_dir / f"dtw_{dataset_name}_{aug_name}.pdf", bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    sys.exit(main())
