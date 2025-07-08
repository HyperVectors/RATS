import os
import time
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf

data = pd.read_csv("../../data/Car/Car.csv").to_numpy()
x = data[:, :-1].astype(np.float64)
y = list(map(str, data[:, -1]))
dataset = pf.Dataset(x, y)

# Augmenters to benchmark: (PyFraug class, args, tsaug equivalent, tsaug args)
with open("augmenters.yaml", "r") as f:
    AUGMENTERS = yaml.safe_load(f)

results = []

os.makedirs("results", exist_ok=True)

for aug in AUGMENTERS:
    aug_name = aug["name"]
    pf_kwargs = aug["pf_kwargs"] or {}
    tsaug_class_name = aug["tsaug_class"]
    tsaug_kwargs = aug["tsaug_kwargs"] or {}

    pf_aug = getattr(pf, aug_name)(**pf_kwargs)
    pf_pipeline = pf.AugmentationPipeline() + pf_aug
    ds_copy = pf.Dataset(x.copy(), y.copy())
    start = time.time()
    pf_pipeline.augment_dataset(ds_copy, parallel=True)
    pf_time = time.time() - start

    if tsaug_class_name:
        tsaug_class = getattr(importlib.import_module("tsaug"), tsaug_class_name)
        x_copy = x.copy()
        start = time.time()
        tsaug_class(**tsaug_kwargs).augment(x_copy)
        tsaug_time = time.time() - start
    else:
        tsaug_time = None

    results.append({
        "Augmenter": aug_name,
        "PyFraug_time_sec": pf_time,
        "tsaug_time_sec": tsaug_time
    })
    print(f"{aug_name}: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}")

# Saving results
df = pd.DataFrame(results)
df.to_csv("./results/time_benchmark.csv", index=False)
print("Benchmark results saved to results/time_benchmark.csv")