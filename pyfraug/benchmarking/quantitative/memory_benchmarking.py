"""
This script benchmarks the memory usage of PyFraug augmenters and transforms against tsaug.
It measures peak memory usage during augmentation and transformation processes.
It generates a CSV file with the results and plots for visual comparison."""

import os
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf
import tsaug as ts
from tqdm import tqdm
from memory_profiler import memory_usage
import argparse
from io import StringIO
import sys
import pathlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_pf_kwargs, load_data


def pf_aug_memory(
    aug_name: str, pf_kwargs: dict, x: np.ndarray, y: np.ndarray
) -> float:
    """
    Measure the peak memory usage of a PyFraug augmenter.
    Args:
        aug_name (str): Name of the PyFraug augmenter.
        pf_kwargs (dict): Keyword arguments for the PyFraug augmenter.
        x (np.ndarray): Input data features.
        y (np.ndarray): Input data labels.
    Returns:
        float: Peak memory usage in MB.
    """
    pf_aug = getattr(pf, aug_name)(**pf_kwargs)
    ds_copy = pf.Dataset(x.copy(), y.copy())
    mem = memory_usage(
        (pf_aug.augment_batch, (ds_copy,), {"parallel": True}), max_usage=True
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
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        pf_mem = pf_aug_memory(aug_name, pf_kwargs, x, y)
        if tsaug_class_name:
            tsaug_mem = tsaug_memory(tsaug_class_name, tsaug_kwargs, x)
        else:
            tsaug_mem = None

        results.append(
            {
                "Augmenter": aug_name,
                "PyFraug_peak_mem_MB": pf_mem,
                "tsaug_peak_mem_MB": tsaug_mem,
            }
        )
        print(
            f"{aug_name}: PyFraug {pf_mem:.2f} MB, tsaug {tsaug_mem if tsaug_mem is not None else 'N/A'} MB"
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
        ds = pf.Dataset(x.copy(), y.copy())
        pf.Transforms.fft(ds, parallel=True)

    fft_peak_mem = memory_usage(fft_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "fft",
            "PyFraug_peak_mem_MB": fft_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"fft: PyFraug {fft_peak_mem:.2f} MB, tsaug N/A")

    print("Running IFFT...")

    def ifft_mem():
        ds = pf.Dataset(x.copy(), y.copy())
        ds_freq = pf.Transforms.fft(ds, parallel=True)
        pf.Transforms.ifft(ds_freq, parallel=True)

    ifft_peak_mem = memory_usage(ifft_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "ifft",
            "PyFraug_peak_mem_MB": ifft_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"ifft: PyFraug {ifft_peak_mem:.2f} MB, tsaug N/A")

    print("Running compare_within_tolerance...")

    def compare_mem():
        ds = pf.Dataset(x.copy(), y.copy())
        ds_freq = pf.Transforms.fft(ds, parallel=True)
        ds_time = pf.Transforms.ifft(ds_freq, parallel=True)
        pf.Transforms.compare_within_tolerance(ds, ds_time, 1e-6)

    compare_peak_mem = memory_usage(compare_mem, max_usage=True)
    results.append(
        {
            "Augmenter": "compare_within_tolerance",
            "PyFraug_peak_mem_MB": compare_peak_mem,
            "tsaug_peak_mem_MB": None,
        }
    )
    print(f"compare_within_tolerance: PyFraug {compare_peak_mem:.2f} MB, tsaug N/A")
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
    pf_pipeline_tsaug = pf.AugmentationPipeline()
    tsaug_pipeline = None
    for aug in tqdm(augmenters, desc="Pipeline_with_tsaug"):
        aug_name = aug["name"]
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(
            tsaug_kwargs.get("window", None), list
        ):
            tsaug_kwargs["window"] = tuple(tsaug_kwargs["window"])

        if tsaug_class_name:
            pf_pipeline_tsaug = pf_pipeline_tsaug + getattr(pf, aug_name)(**pf_kwargs)
            tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
            if tsaug_pipeline is None:
                tsaug_pipeline = tsaug_class(**tsaug_kwargs)
            else:
                tsaug_pipeline = tsaug_pipeline + tsaug_class(**tsaug_kwargs)

    def pf_pipeline_tsaug_mem_false():
        ds_copy = pf.Dataset(x.copy(), y.copy())
        pf_pipeline_tsaug.augment_batch(ds_copy, parallel=True)

    memory_pf_pipeline_tsaug_false = memory_usage(
        pf_pipeline_tsaug_mem_false, max_usage=True
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
            "PyFraug_peak_mem_MB": memory_pf_pipeline_tsaug_false,
            "tsaug_peak_mem_MB": memory_tsaug_pipeline,
        }
    )
    print(
        f"Pipeline_with_tsaug: PyFraug {memory_pf_pipeline_tsaug_false:.2f} MB, tsaug {memory_tsaug_pipeline if memory_tsaug_pipeline is not None else 'N/A'} MB"
    )
    return results


def main():
    """
    Entry point for the memory benchmarking script.
    It parses command-line arguments, loads the dataset, runs the benchmarks,
    and saves the results to a CSV file."""
    parser = argparse.ArgumentParser(
        description="Benchmark PyFraug and tsaug augmenters and transforms (memory)."
    )
    parser.add_argument(
        "--dataset", type=str, default="Car", help="Dataset name (default: Car)"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="../../../examples/Car/Car.csv",
        help="Path to the dataset CSV file (default: ../../../examples/Car/Car.csv)",
    )
    parser.add_argument(
        "--augmenter_configs",
        type=str,
        default="../augmenter_configs.yaml",
        help="Path to the augmenter config YAML file (default: ../augmenter_configs.yaml)",
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    csv_path = pathlib.Path(f"../../../examples/{dataset_name}/{dataset_name}.csv")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
