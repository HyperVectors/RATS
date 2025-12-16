"""
This script benchmarks the memory usage of RATSpy augmenters and transforms against tsaug.
It measures peak memory usage during augmentation and transformation processes.
It generates a CSV file with the results and plots for visual comparison."""

import os
import numpy as np
import pandas as pd
import yaml
import importlib
import ratspy as rp
import tsaug as ts
from tqdm import tqdm
from memory_profiler import memory_usage
import argparse
from io import StringIO
import sys
import pathlib
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,           
    'axes.titlesize': 28,      
    'axes.labelsize': 24,      
    'xtick.labelsize': 20,     
    'ytick.labelsize': 20,     
    'legend.fontsize': 22,     
    'figure.titlesize': 30     
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_rp_kwargs, load_data


def rp_aug_memory(
    aug_name: str, rp_kwargs: dict, x: np.ndarray, y: np.ndarray
) -> float:
    """
    Measure the peak memory usage of a RATSpy augmenter.
    Args:
        aug_name (str): Name of the RATSpy augmenter.
        rp_kwargs (dict): Keyword arguments for the RATSpy augmenter.
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
    Returns:
        float: Peak memory usage in MB.
    """
    rp_aug = getattr(rp, aug_name)(**rp_kwargs)
    ds_copy = rp.Dataset(x.copy(), y.copy())
    mem = memory_usage(
        (rp_aug.augment_batch, (ds_copy,), {"parallel": True}), max_usage=True
    )
    return mem


def tsaug_memory(tsaug_class_name: str, tsaug_kwargs: dict, x: np.ndarray) -> float:
    """
    Measure the peak memory usage of a tsaug augmenter.
    Args:
        tsaug_class_name (str): Name of the tsaug augmenter class.
        tsaug_kwargs (dict): Keyword arguments for the tsaug augmenter.
        x (np.ndarray): Input data features.

    Returns:
        float: Peak memory usage in MB.
    """
    tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
    x_copy = x.copy()
    mem = memory_usage((tsaug_class(**tsaug_kwargs).augment, (x_copy,)), max_usage=True)
    return mem


def run_individual_memory_benchmarks(
    x: np.ndarray, y: np.ndarray, augmenters: list[dict]
) -> list[dict]:
    """
    Run memory benchmarks for each augmenter in the provided list.
    Args:
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
        augmenters (list): List of augmenter configurations.

    Returns:
        list[dict]: List of dictionaries containing memory benchmark results.
    """

    results = []
    os.makedirs("results", exist_ok=True)

    # Benchmark each augmenter
    for aug in tqdm(augmenters, desc="Augmenters"):
        aug_name = aug["name"]
        rp_kwargs = fix_rp_kwargs(aug_name, aug["rp_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        rp_mem = rp_aug_memory(aug_name, rp_kwargs, x, y)
        if tsaug_class_name:
            tsaug_mem = tsaug_memory(tsaug_class_name, tsaug_kwargs, x)
        else:
            tsaug_mem = None

        results.append(
            {
                "Augmenter": aug_name,
                "RATSpy_peak_mem_MB": rp_mem,
                "tsaug_peak_mem_MB": tsaug_mem,
            }
        )
        print(
            f"{aug_name}: RATSpy {rp_mem:.2f} MB, tsaug {tsaug_mem if tsaug_mem is not None else 'N/A'} MB"
        )
    return results


def run_frequency_memory_benchmarks(x, y) -> list[dict]:
    """
    Run memory benchmarks for frequency-domain transformations.
    Args:
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
    Returns:
        list[dict]: List of dictionaries containing memory benchmark results for frequency-domain transformations.
    """

    results = []
    # FFT memory benchmarking
    print("Running FFT...")

    def fft_mem():
        ds = rp.Dataset(x.copy(), y.copy())
        rp.Transforms.fft(ds, parallel=True)

    fft_peak_mem = memory_usage(fft_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "fft",
            "RATSpy_peak_mem_MB": fft_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"fft: RATSpy {fft_peak_mem:.2f} MB, tsaug N/A")

    print("Running IFFT...")

    def ifft_mem():
        ds = rp.Dataset(x.copy(), y.copy())
        ds_freq = rp.Transforms.fft(ds, parallel=True)
        rp.Transforms.ifft(ds_freq, parallel=True)

    ifft_peak_mem = memory_usage(ifft_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "ifft",
            "RATSpy_peak_mem_MB": ifft_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"ifft: RATSpy {ifft_peak_mem:.2f} MB, tsaug N/A")

    print("Running compare_within_tolerance...")

    def compare_mem():
        ds = rp.Dataset(x.copy(), y.copy())
        ds_freq = rp.Transforms.fft(ds, parallel=True)
        ds_time = rp.Transforms.ifft(ds_freq, parallel=True)
        rp.Transforms.compare_within_tolerance(ds, ds_time, 1e-6)

    compare_peak_mem = memory_usage(compare_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "compare_within_tolerance",
            "RATSpy_peak_mem_MB": compare_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"compare_within_tolerance: RATSpy {compare_peak_mem:.2f} MB, tsaug N/A")
    return results


def run_pipeline_memory_benchmarks(
    x: np.ndarray, y: np.ndarray, augmenters: list[dict]
) -> list[dict]:
    """
    Run memory benchmarks for a pipeline that includes all augmenters with tsaug equivalents.
    Args:
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
        augmenters (list): List of augmenter configurations.
    Returns:
        list[dict]: List of dictionaries containing memory benchmark results for the pipeline.
    """
    # Pipeline with only augmenters that have a tsaug equivalent
    results = []
    print("Running Batch Pipeline")
    rp_pipeline_tsaug = rp.AugmentationPipeline()
    tsaug_pipeline = None
    for aug in tqdm(augmenters, desc="Pipeline_with_tsaug"):
        aug_name = aug["name"]
        rp_kwargs = fix_rp_kwargs(aug_name, aug["rp_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        if tsaug_class_name:
            rp_pipeline_tsaug = rp_pipeline_tsaug + getattr(rp, aug_name)(**rp_kwargs)
            tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
            if tsaug_pipeline is None:
                tsaug_pipeline = tsaug_class(**tsaug_kwargs)
            else:
                tsaug_pipeline = tsaug_pipeline + tsaug_class(**tsaug_kwargs)

    def rp_pipeline_tsaug_mem_false():
        ds_copy = rp.Dataset(x.copy(), y.copy())
        rp_pipeline_tsaug.augment_batch(ds_copy, parallel=True)

    memory_rp_pipeline_tsaug_false = memory_usage(
        rp_pipeline_tsaug_mem_false, max_usage=True
    )

    if tsaug_pipeline is not None:

        def tsaug_pipeline_mem():
            x_copy = x.copy()
            tsaug_pipeline.augment(x_copy)

        memory_tsaug_pipeline = memory_usage(tsaug_pipeline_mem, max_usage=True)
    else:
        memory_tsaug_pipeline = None

    results.append(
        {
            "Augmenter": "Pipeline",
            "RATSpy_peak_mem_MB": memory_rp_pipeline_tsaug_false,
            "tsaug_peak_mem_MB": memory_tsaug_pipeline,
        }
    )
    print(
        f"Pipeline_with_tsaug: RATSpy {memory_rp_pipeline_tsaug_false:.2f} MB, tsaug {memory_tsaug_pipeline if memory_tsaug_pipeline is not None else 'N/A'} MB"
    )
    return results


def run_memory_size_benchmarks(
    x: np.ndarray,
    y: np.ndarray,
    augmenters: list[dict],
    dataset_name: str,
    n_iterations: int,
    max_dataset_size: int = None,
) -> str:
    """
     Run memory usage benchmarks for RATSpy and compares them with tsaug for different dataset sizes.
    Args:
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
        augmenters (list): List of augmenter configurations.
        dataset_name (str): Name of the dataset for saving results.
        n_iterations (int): Number of iterations for the benchmark.
        max_dataset_size (int, optional): Maximum dataset size. If specified, stops when dataset size exceeds this value.
    """
    mem_benchmarks = []
    dataset = rp.Dataset(x, y)

    iteration = 0
    while True:
        if max_dataset_size is not None:
            if len(dataset.features) > max_dataset_size:
                print(f"Stopping: Dataset size {len(dataset.features)} exceeds max_dataset_size {max_dataset_size}")
                break
        else:
            if iteration >= n_iterations:
                break

        repeat_augmenter = rp.Repeat(times=2)
        result_list = run_pipeline_memory_benchmarks(
            dataset.features, dataset.labels, augmenters
        )

        print(
            f"Results for dataset size {len(dataset.features)}: RATSpy: {result_list[0]['RATSpy_peak_mem_MB']}, tsaug: {result_list[0]['tsaug_peak_mem_MB']}"
        )

        mem_benchmarks.append(
            {
                "Dataset Size": len(dataset.features),
                "RATSpy Peak Memory (MB)": result_list[0]["RATSpy_peak_mem_MB"],
                "tsaug Peak Memory (MB)": result_list[0]["tsaug_peak_mem_MB"],
            }
        )
        repeat_augmenter.augment_batch(dataset, parallel=True)
        
        iteration += 1

    df = pd.DataFrame(mem_benchmarks)
    
    # Use actual iterations in filename if max_dataset_size was used
    actual_iterations = len(mem_benchmarks)
    df.to_csv(
        f"./results/{dataset_name}_memory_vs_size_{actual_iterations}_iterations.csv",
        index=False,
    )

    return f"./results/{dataset_name}_memory_vs_size_{actual_iterations}_iterations.csv"


def plot_mem_vs_size(save_dir: str, dataset_name: str):
    """
    Plot the memory usage vs dataset size for RATSpy and tsaug augmenters.
    Args:
        save_dir (str): Directory where the CSV file is saved.
        dataset_name (str): Name of the dataset for saving the plot.
    """

    df = pd.read_csv(save_dir)
    plt.figure(figsize=(16, 10))  # Increased size
    plt.plot(
        df["Dataset Size"],
        df["RATSpy Peak Memory (MB)"],
        label="RATSpy",
        marker="o",
        linewidth=3,
        markersize=10,
    )
    plt.plot(
        df["Dataset Size"],
        df["tsaug Peak Memory (MB)"],
        label="tsaug",
        marker="x",
        linewidth=3,
        markersize=12,
    )
    plt.xlabel("Dataset Size", fontsize=28)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=28)
    plt.title(f"Memory Usage vs Dataset Size - {dataset_name}", fontsize=36, pad=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=22)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{dataset_name}_memory_vs_size.eps", format="eps", bbox_inches='tight')
    plt.savefig(f"results/{dataset_name}_memory_vs_size.pdf", format="pdf", bbox_inches='tight')
    plt.close()


def main():
    """
    Entry point for the memory benchmarking script.
    It parses command-line arguments, loads the dataset, runs the benchmarks,
    and saves the results to a CSV file."""
    parser = argparse.ArgumentParser(
        description="Benchmark RATSpy and tsaug augmenters and transforms (memory)."
    )
    parser.add_argument(
        "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="../../../data/Car/Car.csv",
        help="Path to the dataset CSV file (default: ../../../data/Car/Car.csv)",
    )
    parser.add_argument(
        "--augmenter_configs",
        type=str,
        default="../augmenter_configs.yaml",
        help="Path to the augmenter config YAML file (default: ../augmenter_configs.yaml)",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=3,
        help="Number of iterations for memory vs size benchmark (default: 3)",
    )
    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=None,
        help="Maximum dataset size for memory vs size benchmark. If specified, overrides n_iterations and stops when dataset size exceeds this value.",
    )

    args = parser.parse_args()
    dataset_name = args.dataset

    csv_path = pathlib.Path(args.data_path)
    print(f"Loading data from {csv_path}")

    x, y = load_data(csv_path)

    with open(args.augmenter_configs, "r") as f:
        AUGMENTERS = yaml.safe_load(f)

    aug_results = run_individual_memory_benchmarks(x, y, AUGMENTERS)

    aug_results.extend(run_frequency_memory_benchmarks(x, y))

    aug_results.extend(run_pipeline_memory_benchmarks(x, y, AUGMENTERS))

    # Save results
    df = pd.DataFrame(aug_results)
    df.to_csv(f"./results/{dataset_name}_memory_benchmark.csv", index=False)
    print(f"Benchmark results saved to results/{dataset_name}_memory_benchmark.csv")

    save_dir = run_memory_size_benchmarks(
        x, y, AUGMENTERS, dataset_name, args.n_iterations, args.max_dataset_size
    )

    print(f"Memory vs Size results saved to {save_dir}")

    plot_mem_vs_size(save_dir, dataset_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
