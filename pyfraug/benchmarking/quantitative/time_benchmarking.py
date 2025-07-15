import os
import time
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf
import tsaug as ts
from tqdm import tqdm
import argparse
from io import StringIO
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_pf_kwargs

parser = argparse.ArgumentParser(description="Benchmark PyFraug and tsaug augmenters and transforms.")
parser.add_argument("--dataset", type=str, default="Car", help="Dataset name (default: Car)")
args = parser.parse_args()
dataset_name = args.dataset

csv_path = f"../../../data/{dataset_name}/{dataset_name}.csv"
print(f"Loading data from {csv_path}")

# loading data
data = []
with open(csv_path, "r") as f:
    for line in tqdm(f, desc="Loading CSV"):
        data.append(line)
data = pd.read_csv(StringIO("".join(data))).to_numpy()

x = data[:, :-1].astype(np.float64)
y = list(map(str, data[:, -1]))
dataset = pf.Dataset(x, y)

with open("../augmenters.yaml", "r") as f:
    AUGMENTERS = yaml.safe_load(f)

results = []
os.makedirs("results", exist_ok=True)

if __name__ == "__main__":

    # Benchmark each augmenter
    for aug in tqdm(AUGMENTERS, desc="Augmenters"):
        aug_name = aug["name"]
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        # Edge Case tsaug Convolve Gaussian window
        if tsaug_class_name == "Convolve" and isinstance(tsaug_kwargs.get("window", None), list):
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

        results.append({
            "Augmenter": aug_name,
            "PyFraug_time_sec": pf_time,
            "tsaug_time_sec": tsaug_time
        })
        print(f"{aug_name}: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}")

    # FFT benchmarking
    print("Running FFT...")
    start = time.perf_counter()
    ds_freq = pf.Transforms.fft(dataset, parallel=True)
    fft_time = time.perf_counter() - start
    results.append({
        "Augmenter": "fft",
        "PyFraug_time_sec": fft_time,
        "tsaug_time_sec": None
    })
    print(f"fft: PyFraug {fft_time:.4f}s, tsaug N/A")

    print("Running IFFT...")
    start = time.perf_counter()
    ds_time = pf.Transforms.ifft(ds_freq, parallel=True)
    ifft_time = time.perf_counter() - start
    results.append({
        "Augmenter": "ifft",
        "PyFraug_time_sec": ifft_time,
        "tsaug_time_sec": None
    })
    print(f"ifft: PyFraug {ifft_time:.4f}s, tsaug N/A")

    print("Validating FFT and IFFT...")
    start = time.perf_counter()
    max_diff, all_within = pf.Transforms.compare_within_tolerance(dataset, ds_time, 1e-6)
    diff_time = time.perf_counter() - start
    print(f"compare_within_tolerance: PyFraug {diff_time:.4f}s, max_diff={max_diff}, all_within={all_within}")

    
    
    print("Running Batch Pipeline_with_tsaug")
    pf_pipeline_tsaug = pf.AugmentationPipeline()
    tsaug_pipeline = None
    for aug in tqdm(AUGMENTERS, desc="Pipeline_with_tsaug"):
        aug_name = aug["name"]
        pf_kwargs = fix_pf_kwargs(aug_name, aug["pf_kwargs"] or {})
        tsaug_class_name = aug["tsaug_class"]
        tsaug_kwargs = aug["tsaug_kwargs"] or {}

        if tsaug_class_name == "Convolve" and isinstance(tsaug_kwargs.get("window", None), list):
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

    results.append({
        "Augmenter": "Pipeline",
        "PyFraug_time_sec": pf_time,
        "tsaug_time_sec": tsaug_time
    })
    print(f"Pipeline_with_tsaug: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}")

    # Saving results
    df = pd.DataFrame(results)
    df.to_csv(f"./results/{dataset_name}_time_benchmark.csv", index=False)
    print(f"Benchmark results saved to results/{dataset_name}_time_benchmark.csv")