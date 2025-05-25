import argparse
import sys
import pandas as pd
import json
from aeon.datasets import load_classification
import os

parser = argparse.ArgumentParser(
    description="Load a time series classification dataset.",
    usage="python dataLoader.py --dataset <DATASET_NAME>"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="GunPoint",
    help="Name of the time series dataset to load (default: GunPoint)"
)

try:
    args = parser.parse_args()
except Exception as e:
    print("Error:", e)
    print("Usage: python dataLoader.py --dataset <DATASET_NAME>")
    sys.exit(1)

try:
    X, y, meta_data = load_classification(
        name=args.dataset,
        split=None,
        return_metadata=True
    )
except Exception as e:
    print(f"Failed to load dataset '{args.dataset}': {e}")
    print("Usage: python dataLoader.py --dataset <DATASET_NAME>")
    sys.exit(1)

print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)

# Save as CSV

# directory with the name of the dataset if it doesn't exist


if not os.path.exists('../../data/' + args.dataset):
    os.makedirs('../../data/' + args.dataset)


if hasattr(X, "to_numpy"):
    X_np = X.to_numpy()
else:
    X_np = X

# If X_np is 3D , squeeze the 2nd dimension
if X_np.ndim == 3 and X_np.shape[1] == 1:
    X_np = X_np.squeeze(1)  # shape becomes (n_samples, n_timesteps)

n_timesteps = X_np.shape[1]
col_names = [f"t_{i}" for i in range(n_timesteps)]
df = pd.DataFrame(X_np, columns=col_names)
df["label"] = y
csv_filename = f"../../data/{args.dataset}/{args.dataset}.csv"
df.to_csv(csv_filename, index=False)
print(f"Saved dataset to {csv_filename}")

# Save metadata as JSON
meta_filename = f"../../data/{args.dataset}/{args.dataset}_meta.json"
with open(meta_filename, "w") as f:
    json.dump(meta_data, f, indent=4)
print(f"Saved metadata to {meta_filename}")