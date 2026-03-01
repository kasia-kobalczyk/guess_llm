from datasets import load_from_disk
import os
import random
from datasets import concatenate_datasets
import numpy as np
import numpy as np
from collections import defaultdict
import random
from pathlib import Path
import argparse
import pandas as pd

argparse = argparse.ArgumentParser(description="Combine datasets")
argparse.add_argument(
    "--model_name",
    default="Llama-2-7b-hf",
    type=str,
)

args = argparse.parse_args()

project_root = str(Path(__file__).parent.parent.parent.parent)
DATA_PATH = os.path.join(project_root, "data")

# Step 1: Concatenate dataset and save
# ====================================
num_samples = 100
dss = []
for i in ["1.0", "10.0", "1000.0", "10000.0"]:
    dataset_name = "multi_func_scale_" + i
    dataset_path = os.path.join(
        DATA_PATH, dataset_name, "embedded_data", args.model_name
    )
    ds = load_from_disk(dataset_path)
    if "vert_stretch" in ds.features.keys():
        ds = ds.remove_columns(["vert_stretch", "displacement"])
    dss.append(ds)

combined_ds = concatenate_datasets(dss)

correct_indices = np.argwhere(np.array(combined_ds["error"]) == 0).flatten().tolist()
combined_ds = combined_ds.select(correct_indices)

# Get rid of the rows where not enough samples were generated
correct_indices = (
    np.argwhere(np.array([len(y) == num_samples for y in combined_ds["y_pred"]]))
    .flatten()
    .tolist()
)
combined_ds = combined_ds.select(correct_indices)

# Save the combined dataset
save_path = os.path.join(
    DATA_PATH, "multi_func_combined", "embedded_data", args.model_name
)
combined_ds.save_to_disk(save_path)

# Remove the original datasets to save space
for i in ["1.0", "10.0", "1000.0", "10000.0"]:
    dataset_name = "multi_func_scale_" + i
    dataset_path = os.path.join(
        DATA_PATH, dataset_name, "embedded_data", args.model_name
    )
    if os.path.exists(dataset_path):
        os.system(f"rm -rf {dataset_path}")


# # Step 2: Create balanced splits
# ==============================
def get_bins(ds, max_value=10000):
    # Extract y_test column
    y = np.abs(np.median(np.array(ds["y_pred"]), axis=1))
    y_mean = np.abs(np.mean(np.array(ds["y_pred"]), axis=1))
    y_greedy = np.abs(np.array(ds["y_greedy"])).squeeze(1)

    # Define log-spaced bins and digitize
    bins = np.logspace(np.log10(0.001), np.log10(10000), num=8)

    bin_ids = np.digitize(y, bins)

    # Group indices by bin
    bin_to_indices = dict(
        zip(list(range(0, len(bins))), [[] for _ in range(0, len(bins))])
    )
    for idx, b in enumerate(bin_ids):
        if (
            0 <= b <= len(bins)
            and y[idx] < max_value
            and y_mean[idx] < max_value
            and y_greedy[idx] < max_value
        ):  # ignore out-of-range
            bin_to_indices[b].append(idx)

    return bin_to_indices


for max_val in [1, 10, 1000, 10000]:
    bin_to_indices = get_bins(combined_ds, max_value=max_val)

    # Uniform sampling per bin
    samples_per_bin = 12000
    selected_indices = []
    for b, indices in bin_to_indices.items():
        if len(indices) >= samples_per_bin:
            selected = random.sample(indices, samples_per_bin)
        else:
            selected = indices
        selected_indices.extend(selected)

    # Create the balanced datasets
    balanced_dataset = combined_ds.select(selected_indices)

    _ = get_bins(balanced_dataset)

    # Save split files
    splits_df = pd.DataFrame(
        {
            "index": selected_indices,
            "split": np.random.choice(
                ["train", "val", "test"], p=[0.8, 0.1, 0.1], size=len(selected_indices)
            ),
        }
    )

    save_dir = os.path.join(DATA_PATH, "multi_func_combined", "splits", args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    splits_df.to_csv(os.path.join(save_dir, f"balanced_{max_val}.csv"), index=False)
