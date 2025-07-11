import os
import time
import numpy as np
import pandas as pd
import yaml
import importlib
import pyfraug as pf
import tsaug as ts

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
    ds_copy = pf.Dataset(x.copy(), y.copy())
    start = time.time()
    pf_aug.augment_batch(ds_copy, parallel=True)
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

pf_pipeline = (pf.AugmentationPipeline()
                + pf.Repeat(5)
                + pf.Crop(400)
                + pf.Jittering(0.1)
                + pf.Quantize(50))
ds_copy = pf.Dataset(x.copy(), y.copy())
start = time.time()
pf_pipeline.augment_batch(ds_copy, parallel=True)
pf_time = time.time() - start

tsaug_pipeline = (ts.Crop(size=400) * 5
                + ts.AddNoise(scale=0.1)
                + ts.Quantize(n_levels=50))
x_copy = x.copy()
start = time.time()
tsaug_pipeline.augment(x_copy)
tsaug_time = time.time() - start

results.append({
    "Augmenter": "Pipeline",
    "PyFraug_time_sec": pf_time,
    "tsaug_time_sec": tsaug_time
})
print(f"Pipeline: PyFraug {pf_time:.4f}s, tsaug {tsaug_time if tsaug_time is not None else 'N/A'}")

# FFT benchmarking
start = time.time()
ds_freq = pf.Transforms.fft(dataset, parallel=True)
fft_time = time.time() - start

results.append({
    "Augmenter": "fft",
    "PyFraug_time_sec": fft_time,
    "tsaug_time_sec": None
})
print(f"fft: PyFraug {fft_time:.4f}s, tsaug N/A")

# IFFT benchmarking
start = time.time()
ds_time = pf.Transforms.ifft(ds_freq, parallel=True)
ifft_time = time.time() - start

results.append({
    "Augmenter": "ifft",
    "PyFraug_time_sec": ifft_time,
    "tsaug_time_sec": None
})
print(f"ifft: PyFraug {ifft_time:.4f}s, tsaug N/A")

# Validating FFT and IFFT
start = time.time()
max_diff, all_within = pf.Transforms.compare_within_tolerance(dataset, ds_time, 1e-6)
diff_time = time.time() - start

print(f"compare_within_tolerance: PyFraug {diff_time:.4f}s, max_diff={max_diff}, all_within={all_within}")

# Saving results
df = pd.DataFrame(results)
df.to_csv("./results/time_benchmark.csv", index=False)
print("Benchmark results saved to results/time_benchmark.csv")