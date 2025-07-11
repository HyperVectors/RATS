import os
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf
import tsaug as ts
from memory_profiler import memory_usage

data = pd.read_csv("../../data/Car/Car.csv").to_numpy()
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

    for aug in AUGMENTERS:
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

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("./results/memory_benchmark.csv", index=False)
    print("Benchmark results saved to results/memory_benchmark.csv")