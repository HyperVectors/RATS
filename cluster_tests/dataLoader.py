import argparse
import sys
import pandas as pd
import json
from aeon.datasets import load_classification
import os

from aeon.datasets import get_dataset_meta_data

df = get_dataset_meta_data()
df = df[df["Channels"] == 1]
print(f"Total datasets available: {len(df)}")

parser = argparse.ArgumentParser(
    description="Load a time series classification dataset.",
    usage="python dataLoader.py --dataset-idx <DATASET_INDEX>",
)
parser.add_argument(
    "--dataset-idx",
    type=str,
    required=True,
    help="Index of the time series dataset to load",
)

try:
    args = parser.parse_args()
except Exception as e:
    print("Error:", e)
    print("Usage: python dataLoader.py --dataset <DATASET_NAME>")
    sys.exit(1)

try:
    X, y, meta_data = load_classification(
        name=df.iloc[int(args.dataset_idx)]["Dataset"], split=None, return_metadata=True
    )
except Exception as e:
    print(f"Failed to load dataset with index '{args.dataset_idx}': {e}")
    print("Usage: python dataLoader.py --dataset-idx <DATASET_INDEX>")
    sys.exit(1)

print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)

# Save as CSV

# directory with the name of the dataset if it doesn't exist


if not os.path.exists("./data/" + args.dataset_idx):
    os.makedirs("./data/" + args.dataset_idx, exist_ok=True)

if not os.path.exists("./results/" + args.dataset_idx):
    os.makedirs("./results/" + args.dataset_idx, exist_ok=True)

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
csv_filename = f"./data/{args.dataset_idx}/{args.dataset_idx}.csv"
df.to_csv(csv_filename, index=False)
print(f"Saved dataset to {csv_filename}")

# metadata as JSON
meta_filename = f"./results/{args.dataset_idx}/{args.dataset_idx}_meta.json"
with open(meta_filename, "w") as f:
    json.dump(meta_data, f, indent=4)
print(f"Saved metadata to {meta_filename}")
