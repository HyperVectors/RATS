import os
import sys
import json
import pandas as pd
import arff
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python arff_to_csv.py <ARFF_FILE>")
    sys.exit(1)

arff_path = sys.argv[1]
dataset_name = os.path.splitext(os.path.basename(arff_path))[0]
output_dir = os.path.join("../../data", dataset_name)
os.makedirs(output_dir, exist_ok=True)

print(f"Reading ARFF file: {arff_path}")
with open(arff_path, "r") as f:
    arff_data = arff.load(f)
print("ARFF file loaded.")

# Extract data and attribute names
data = arff_data["data"]
attributes = arff_data["attributes"]
print(f"Found {len(data)} rows and {len(attributes)} attributes.")

# Find label column (assume last attribute is label)
label_col = None
for i, (name, attr_type) in enumerate(attributes):
    if isinstance(attr_type, list) or attr_type == "STRING":
        label_col = i
        break
if label_col is None:
    label_col = len(attributes) - 1  # fallback: last column

print(f"Label column detected: {attributes[label_col][0]} (index {label_col})")

# Build DataFrame
print("Converting ARFF data to DataFrame...")
df = pd.DataFrame(
    tqdm(data, desc="Rows processed", unit="row"),
    columns=[a[0] for a in attributes]
)
labels = df.iloc[:, label_col]
features = df.drop(df.columns[label_col], axis=1)

# Save CSV (features + label as last column)
print("Saving CSV file...")
csv_df = features.copy()
csv_df["label"] = labels
csv_filename = os.path.join(output_dir, f"{dataset_name}.csv")
csv_df.to_csv(csv_filename, index=False)
print(f"Saved dataset to {csv_filename}")

# Save metadata as JSON
print("Saving metadata JSON...")
meta = {
    "n_samples": len(df),
    "n_timesteps": features.shape[1],
    "label_col": df.columns[label_col],
    "attribute_names": list(features.columns),
    "label_values": sorted(labels.unique())
}
meta_filename = os.path.join(output_dir, f"{dataset_name}_meta.json")
with open(meta_filename, "w") as f:
    json.dump(meta, f, indent=4)
print(f"Saved metadata to {meta_filename}")