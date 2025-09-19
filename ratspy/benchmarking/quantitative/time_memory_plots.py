"""
This script generates time and memory benchmark plots for RATSpy and tsaug augmenters. It uses the csv
results from the `time_benchmarking.py` and `memory_benchmarking.py` scripts to create these plots
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Entry point for the script. It parses command line arguments, loads the time and memory benchmark CSV files,
    and generates plots comparing RATSpy and tsaug augmenters.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
    )
    parser.add_argument(
        "--time_csv",
        type=str,
        required=True,
        help="Path to the time benchmark CSV file (default: ./results/Car_time_benchmark.csv)",
    )
    parser.add_argument(
        "--memory_csv",
        type=str,
        required=True,
        help="Path to the memory benchmark CSV file (default: ./results/Car_memory_benchmark.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/plots",
        help="Directory to save the plots (default: ./results/plots)",
    )
    parser.add_argument(
        "--remove_cols",
        nargs="+",
        default=["Pipeline", "FFT", "IFFT", "Compare"],
        help="List of augmenter names to remove from the plots (default: Pipeline FFT IFFT Compare)",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    time_csv = args.time_csv
    mem_csv = args.memory_csv

    time_df = pd.read_csv(time_csv)
    mem_df = pd.read_csv(mem_csv)

    # Removing pipeline/fft/ifft/compare rows for per-augmenter plots
    remove_cols: list = args.remove_cols
    time_augs = time_df[~time_df["Augmenter"].isin(remove_cols)]
    mem_augs = mem_df[~mem_df["Augmenter"].isin(remove_cols)]

    # RATSpy vs tsaug time
    plt.figure(figsize=(12, 5))
    plt.bar(
        time_augs["Augmenter"],
        time_augs["RATSpy_time_sec"],
        width=0.4,
        label="RATSpy",
        align="center",
    )
    plt.bar(
        time_augs["Augmenter"],
        time_augs["tsaug_time_sec"],
        width=0.4,
        label="tsaug",
        align="edge",
    )
    plt.ylabel("Time (seconds)")
    plt.title(f"{dataset_name} Dataset Augmentation Time: RATSpy vs tsaug")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_time_benchmark_bar.png")
    plt.close()

    # RATSpy vs tsaug memory
    plt.figure(figsize=(12, 5))
    plt.bar(
        mem_augs["Augmenter"],
        mem_augs["RATSpy_peak_mem_MB"],
        width=0.4,
        label="RATSpy",
        align="center",
    )
    plt.bar(
        mem_augs["Augmenter"],
        mem_augs["tsaug_peak_mem_MB"],
        width=0.4,
        label="tsaug",
        align="edge",
    )
    plt.ylabel("Peak Memory (MB)")
    plt.title(f"{dataset_name} Dataset Augmentation Peak Memory: RATSpy vs tsaug")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_memory_benchmark_bar.png")
    plt.close()

    # Time vs Memory (only augmenters present in both)
    both = set(time_augs["Augmenter"]).intersection(set(mem_augs["Augmenter"]))
    both = [
        a
        for a in both
        if not (
            pd.isna(time_augs.set_index("Augmenter").loc[a, "tsaug_time_sec"])
            or pd.isna(mem_augs.set_index("Augmenter").loc[a, "tsaug_peak_mem_MB"])
            or pd.isna(mem_augs.set_index("Augmenter").loc[a, "tsaug_peak_mem_MB"])
        )
    ]

    plt.figure(figsize=(8, 6))
    for lib, color in [("RATSpy", "tab:blue"), ("tsaug", "tab:orange")]:
        x = []
        y = []
        labels = []
        for aug in both:
            if lib == "RATSpy":
                t = time_augs.set_index("Augmenter").loc[aug, "RATSpy_time_sec"]
                m = mem_augs.set_index("Augmenter").loc[aug, "RATSpy_peak_mem_MB"]
            else:
                t = time_augs.set_index("Augmenter").loc[aug, "tsaug_time_sec"]
                m = mem_augs.set_index("Augmenter").loc[aug, "tsaug_peak_mem_MB"]
            if not pd.isna(t) and not pd.isna(m):
                x.append(t)
                y.append(m)
                labels.append(aug)
        plt.scatter(x, y, label=lib, color=color)
        for xi, yi, label in zip(x, y, labels):
            plt.annotate(label, (xi, yi), fontsize=8, alpha=0.7)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Peak Memory (MB)")
    plt.title(
        f"{dataset_name} Dataset Time vs Memory: RATSpy vs tsaug (per augmenter)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{dataset_name}_time_vs_memory_scatter.png")
    plt.close()

    print(
        f"Plots saved as results/plots/{dataset_name}_time_benchmark_bar.png, "
        f"results/plots/{dataset_name}_memory_benchmark_bar.png, and "
        f"results/plots/{dataset_name}_time_vs_memory_scatter.png"
    )


if __name__ == "__main__":
    sys.exit(main())
