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

parser = argparse.ArgumentParser(description="Benchmark PyFraug and tsaug augmenters and transforms (memory).")
parser.add_argument("--dataset", type=str, default="Car", help="Dataset name (default: Car)")
args = parser.parse_args()
dataset_name = args.dataset

csv_path = f"../../data/{dataset_name}/{dataset_name}.csv"
print(f"Loading data from {csv_path}")

data = []
with open(csv_path, "r") as f:
    for line in tqdm(f, desc="Loading CSV"):
        data.append(line)
data = pd.read_csv(StringIO("".join(data))).to_numpy()

x = data[:, :-1].astype(np.float64)
y = list(map(str, data[:, -1]))

def pf_aug_memory(aug_name, pf_kwargs):
    pf_aug = getattr(pf, aug_name)(**pf_kwargs)
    ds_copy = pf.Dataset(x.copy(), y.copy())
    mem = memory_usage((pf_aug.augment_batch, (ds_copy,), {'parallel': True}), max_usage=True)
    return mem

def tsaug_memory(tsaug_class_name, tsaug_kwargs):
    tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
    x_copy = x.copy()
    mem = memory_usage((tsaug_class(**tsaug_kwargs).augment, (x_copy,)), max_usage=True)
    return mem

if __name__ == "__main__":
    with open("augmenters.yaml", "r") as f:
        AUGMENTERS = yaml.safe_load(f)

    results = []
    os.makedirs("results", exist_ok=True)

    # Benchmark each augmenter
    for aug in tqdm(AUGMENTERS, desc="Augmenters"):
        aug_name = aug["name"]
        pf_kwargs = aug["pf_kwargs"] or {}
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        pf_mem = pf_aug_memory(aug_name, pf_kwargs)
        if tsaug_class_name:
            tsaug_mem = tsaug_memory(tsaug_class_name, tsaug_kwargs)
        else:
            tsaug_mem = None

        results.append({
            "Augmenter": aug_name,
            "PyFraug_peak_mem_MB": pf_mem,
            "tsaug_peak_mem_MB": tsaug_mem
        })
        print(f"{aug_name}: PyFraug {pf_mem:.2f} MB, tsaug {tsaug_mem if tsaug_mem is not None else 'N/A'} MB")

    # FFT memory benchmarking
    print("Running FFT...")
    def fft_mem():
        ds = pf.Dataset(x.copy(), y.copy())
        pf.Transforms.fft(ds, parallel=True)
    fft_peak_mem = memory_usage(fft_mem, max_usage=True)
    results.append({
        "Augmenter": "fft",
        "PyFraug_peak_mem_MB": fft_peak_mem,
        "tsaug_peak_mem_MB": None
    })
    print(f"fft: PyFraug {fft_peak_mem:.2f} MB, tsaug N/A")

    print("Running IFFT...")
    def ifft_mem():
        ds = pf.Dataset(x.copy(), y.copy())
        ds_freq = pf.Transforms.fft(ds, parallel=True)
        pf.Transforms.ifft(ds_freq, parallel=True)
    ifft_peak_mem = memory_usage(ifft_mem, max_usage=True)
    results.append({
        "Augmenter": "ifft",
        "PyFraug_peak_mem_MB": ifft_peak_mem,
        "tsaug_peak_mem_MB": None
    })
    print(f"ifft: PyFraug {ifft_peak_mem:.2f} MB, tsaug N/A")

    print("Running compare_within_tolerance...")
    def compare_mem():
        ds = pf.Dataset(x.copy(), y.copy())
        ds_freq = pf.Transforms.fft(ds, parallel=True)
        ds_time = pf.Transforms.ifft(ds_freq, parallel=True)
        pf.Transforms.compare_within_tolerance(ds, ds_time, 1e-6)
    compare_peak_mem = memory_usage(compare_mem, max_usage=True)
    results.append({
        "Augmenter": "compare_within_tolerance",
        "PyFraug_peak_mem_MB": compare_peak_mem,
        "tsaug_peak_mem_MB": None
    })
    print(f"compare_within_tolerance: PyFraug {compare_peak_mem:.2f} MB, tsaug N/A")

    # Pipeline with only augmenters that have a tsaug equivalent
    print("Running Pipeline_with_tsaug...")
    pf_pipeline_tsaug = pf.AugmentationPipeline()
    tsaug_pipeline = None
    for aug in tqdm(AUGMENTERS, desc="Pipeline_with_tsaug"):
        aug_name = aug["name"]
        pf_kwargs = aug["pf_kwargs"] or {}
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        if tsaug_class_name:
            pf_pipeline_tsaug = pf_pipeline_tsaug + getattr(pf, aug_name)(**pf_kwargs)
            tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
            if tsaug_pipeline is None:
                tsaug_pipeline = tsaug_class(**tsaug_kwargs)
            else:
                tsaug_pipeline = tsaug_pipeline + tsaug_class(**tsaug_kwargs)

    def pf_pipeline_tsaug_mem():
        ds_copy = pf.Dataset(x.copy(), y.copy())
        pf_pipeline_tsaug.augment_batch(ds_copy, parallel=True)
    memory_pf_pipeline_tsaug = memory_usage(pf_pipeline_tsaug_mem, max_usage=True)

    if tsaug_pipeline is not None:
        def tsaug_pipeline_mem():
            x_copy = x.copy()
            tsaug_pipeline.augment(x_copy)
        memory_tsaug_pipeline = memory_usage(tsaug_pipeline_mem, max_usage=True)
    else:
        memory_tsaug_pipeline = None

    results.append({
        "Augmenter": "Pipeline_with_tsaug",
        "PyFraug_peak_mem_MB": memory_pf_pipeline_tsaug,
        "tsaug_peak_mem_MB": memory_tsaug_pipeline
    })
    print(f"Pipeline_with_tsaug: PyFraug {memory_pf_pipeline_tsaug:.2f} MB, tsaug {memory_tsaug_pipeline if memory_tsaug_pipeline is not None else 'N/A'} MB")

    # Full pipeline with all augmenters
    print("Running Full_pipeline...")
    pf_full_pipeline = pf.AugmentationPipeline()
    for aug in tqdm(AUGMENTERS, desc="Full_pipeline"):
        aug_name = aug["name"]
        pf_kwargs = aug["pf_kwargs"] or {}
        pf_full_pipeline = pf_full_pipeline + getattr(pf, aug_name)(**pf_kwargs)

    def pf_full_pipeline_mem():
        ds_copy = pf.Dataset(x.copy(), y.copy())
        pf_full_pipeline.augment_batch(ds_copy, parallel=True)
    memory_pf_full_pipeline = memory_usage(pf_full_pipeline_mem, max_usage=True)

    results.append({
        "Augmenter": "Full_pipeline",
        "PyFraug_peak_mem_MB": memory_pf_full_pipeline,
        "tsaug_peak_mem_MB": None
    })
    print(f"Full_pipeline: PyFraug {memory_pf_full_pipeline:.2f} MB, tsaug N/A")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"./results/{dataset_name}_memory_benchmark.csv", index=False)
    print(f"Benchmark results saved to results/{dataset_name}_memory_benchmark.csv")
