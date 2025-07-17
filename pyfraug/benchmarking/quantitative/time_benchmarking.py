"""
This script creates time benchmarks for individual fraug augmenters as well as pipeline and compares them with tsaug.
It generates a CSV file with the results of the benchmarking.
"""

import pathlib
import sys
import os
import matplotlib.pyplot as plt

## To mount utils on to the default interpreter path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_pf_kwargs, load_data
from tqdm import tqdm
import argparse
from io import StringIO
import time
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf
import tsaug as ts


def run_individual_time_benchmarks(
    augmenters: list[dict], x: np.ndarray, y: np.ndarray
) -> list[dict]:
    """
    Run individual time benchmarks for PyFraug and tsaug augmenters.
    Args:
        augmenters (list): List of augmenter configurations.
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
    Returns:
        list[dict]: List of dictionaries containing time benchmark results for each augmenter.
    """
    results = []
    for aug in tqdm(
        augmenters,
        desc="Running individual augmenter benchmarks",
        total=len(augmenters),
    ):
        aug_name = aug["name"]
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge Case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        pf_aug_class = getattr(pf, aug_name)
        pf_aug = pf_aug_class(**pf_kwargs)

        ds_copy = pf.Dataset(x.copy(), y.copy())
        start = time.perf_counter()
        pf_aug.augment_batch(ds_copy, parallel=True)
        pf_time = time.perf_counter() - start

        if tsaug_class_name:
            tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
            x_copy = x.copy()
            start = time.perf_counter()
            tsaug_class(**tsaug_kwargs).augment(x_copy)
            tsaug_time = time.perf_counter() - start
        else:
            tsaug_time = None

        results.append(
            {
                "Augmenter": aug_name,
                "PyFraug_time_sec": pf_time,
                "tsaug_time_sec": tsaug_time,
            }
        )
        print(
            f"{aug_name}: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}"
        )

    return results


def run_freq_transformation_benchmarks(dataset: pf.Dataset) -> list[dict]:
    """
    Creates a benchmark for frequency transformations like FFT and IFFT.
    Args:
        dataset (pf.Dataset): The dataset to run the benchmarks on.
    Returns:
        list[dict]: List of dictionaries containing time benchmark results for frequency transformations.
    """

    results = []
    # FFT benchmarking
    print("Running FFT...")
    start = time.perf_counter()
    ds_freq = pf.Transforms.fft(dataset, parallel=True)
    fft_time = time.perf_counter() - start
    results.append(
        {"Augmenter": "fft", "PyFraug_time_sec": fft_time, "tsaug_time_sec": None}
    )
    print(f"fft: PyFraug {fft_time:.4f}s, tsaug N/A")

    print("Running IFFT...")
    start = time.perf_counter()
    ds_time = pf.Transforms.ifft(ds_freq, parallel=True)
    ifft_time = time.perf_counter() - start
    results.append(
        {"Augmenter": "ifft", "PyFraug_time_sec": ifft_time, "tsaug_time_sec": None}
    )
    print(f"ifft: PyFraug {ifft_time:.4f}s, tsaug N/A")

    print("Validating FFT and IFFT...")
    start = time.perf_counter()
    max_diff, all_within = pf.Transforms.compare_within_tolerance(
        dataset, ds_time, 1e-6
    )
    diff_time = time.perf_counter() - start
    print(
        f"compare_within_tolerance: PyFraug {diff_time:.4f}s, max_diff={max_diff}, all_within={all_within}"
    )

    return results


def run_pipeline_benchmarks(
    augmenters: list[dict], x: np.ndarray, y: np.ndarray
) -> list[dict]:
    """
    Run a benchmark for the augmentation pipeline against tsaug pipeline.
    Args:
        augmenters (list): List of augmenter configurations.
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
    Returns:
        list[dict]: List of dictionaries containing time benchmark results for the augmentation pipeline.
    """
    results = []
    print("Running Batch Pipeline_with_tsaug")
    pf_pipeline_tsaug = pf.AugmentationPipeline()
    tsaug_pipeline = None
    for aug in tqdm(augmenters, desc="Pipeline_with_tsaug"):
        aug_name = aug["name"]
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        pf_aug_class = getattr(pf, aug_name)
        if tsaug_class_name:
            pf_pipeline_tsaug = pf_pipeline_tsaug + pf_aug_class(**pf_kwargs)
            tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
            if tsaug_pipeline is None:
                tsaug_pipeline = tsaug_class(**tsaug_kwargs)
            else:
                tsaug_pipeline = tsaug_pipeline + tsaug_class(**tsaug_kwargs)

    ds_copy = pf.Dataset(x.copy(), y.copy())
    start = time.perf_counter()
    pf_pipeline_tsaug.augment_batch(ds_copy, parallel=True)
    pf_time = time.perf_counter() - start

    if tsaug_pipeline is not None:
        x_copy = x.copy()
        start = time.perf_counter()
        tsaug_pipeline.augment(x_copy)
        tsaug_time = time.perf_counter() - start
    else:
        tsaug_time = None

    results.append(
        {
            "Augmenter": "Pipeline",
            "PyFraug_time_sec": pf_time,
            "tsaug_time_sec": tsaug_time,
        }
    )
    print(
        f"Pipeline_with_tsaug: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}"
    )
    return results


def benchmark_time_dataset_size(
    augmenters, x, y, dataset_name, n_iterations: int
) -> str:
    """
    Benchmark the time taken by the augmentation pipeline with varying dataset sizes.
    Args:
        augmenters (list): List of augmenter configurations.
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
        dataset_name (str): Name of the dataset for saving results.
        n_iterations (int): Number of iterations to run the benchmark. At every iteration, the dataset size is doubled.
    Returns:
        str: Path to the CSV file containing time benchmarks for varying dataset sizes.
    """
    time_benchmarks = []
    dataset = pf.Dataset(x, y)

    for i in range(n_iterations):
        repeat_augmenter = pf.Repeat(times=2)
        result_list = run_pipeline_benchmarks(
            augmenters, dataset.features, dataset.labels
        )

        print(
            f"Results for dataset size {len(dataset.features)}: Pyfraug: {result_list[0]['PyFraug_time_sec']}, tsaug: {result_list[0]['tsaug_time_sec']}"
        )
        time_benchmarks.append(
            {
                "Dataset_size": len(dataset.features),
                "PyFraug_time_sec": result_list[0]["PyFraug_time_sec"],
                "tsaug_time_sec": result_list[0]["tsaug_time_sec"],
            }
        )

        repeat_augmenter.augment_batch(
            dataset,
            parallel=True,
        )

    df = pd.DataFrame(time_benchmarks)
    df.to_csv(f"./results/{dataset_name}_time_vs_size.csv", index=False)

    return f"./results/{dataset_name}_time_vs_size.csv"


def plot_time_vs_size(csv_path: str, dataset_name: str):
    """
    Plot the time taken by PyFraug and tsaug for varying dataset sizes.
    Args:
        df (pd.DataFrame): DataFrame containing time benchmarks.
        dataset_name (str): Name of the dataset for saving the plot.
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["Dataset_size"],
        df["PyFraug_time_sec"],
        label="PyFraug Time",
        color="black",
        marker="o",
    )

    plt.plot(
        df["Dataset_size"],
        df["tsaug_time_sec"],
        label="tsaug Time",
        color="orange",
        marker="x",
    )
    plt.xlabel("Dataset Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Time Benchmark for Varying Dataset Sizes")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/{dataset_name}_time_vs_size.png")

    print(f"Time vs Size plot saved to results/{dataset_name}_time_vs_size.png")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyFraug and tsaug augmenters and transforms."
    )
    parser.add_argument(
        "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
    )
    parser.add_argument(
        "--augmenter_configs",
        type=pathlib.Path,
        default="../augmenter_configs.yaml",
        help="Path to the YAML file containing augmenter configurations (default: ../augmenter_configs.yaml)",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of iterations for time vs size benchmark (default: 5)",
    )

    args = parser.parse_args()
    dataset_name = args.dataset

    csv_path = pathlib.Path(f"../../../examples/{dataset_name}/{dataset_name}.csv")
    print(f"Loading data from {csv_path}")

    # loading data
    x, y = load_data(csv_path)
    dataset = pf.Dataset(x, y)

    with open(args.augmenter_configs, "r") as f:
        AUGMENTERS = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    aug_results = run_individual_time_benchmarks(AUGMENTERS, x, y)

    aug_results.extend(run_freq_transformation_benchmarks(dataset))

    aug_results.extend(run_pipeline_benchmarks(AUGMENTERS, x, y))

    df = pd.DataFrame(aug_results)
    df.to_csv(f"./results/{dataset_name}_time_benchmark.csv", index=False)
    print(f"Benchmark results saved to results/{dataset_name}_time_benchmark.csv")

    save_file_path = benchmark_time_dataset_size(
        AUGMENTERS, x, y, dataset_name, args.n_iterations
    )
    print(f"Time vs Size results saved to {save_file_path}")

    plot_time_vs_size(save_file_path, dataset_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
