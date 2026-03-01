# Combine monash and darts
from guess_llm.datasets.generate_splits import load_dataset
from datasets import concatenate_datasets
from pathlib import Path
import os
import argparse

argparse = argparse.ArgumentParser(description="Combine datasets")
argparse.add_argument(
    "--model_name",
    default="Llama-2-7b-hf",
    type=str,
)
args = argparse.parse_args()
project_root = str(Path(__file__).parent.parent.parent.parent)
DATA_PATH = os.path.join(project_root, "data")

num_samples = 100

# Load datasets
monash = load_dataset("monash_max_n_series_50_v2", args.model_name, num_samples)
darts = load_dataset("darts_v1", args.model_name, num_samples)

# Add identifier column
monash = monash.add_column("data_subset", ["monash"] * len(monash))
darts = darts.add_column("data_subset", ["darts"] * len(darts))
darts = darts.add_column("y_test_new", [x[0] for x in darts["y_test"]])
darts = darts.remove_columns("y_test")
darts = darts.rename_column("y_test_new", "y_test")

# Define common schema
common_features = monash.features

# Cast darts to the same features:
for c in darts.features:
    if c not in monash.features:
        darts = darts.remove_columns(c)
darts = darts.cast(common_features)

# Concatenate
combined_ds = concatenate_datasets([monash, darts])
combined_ds.save_to_disk(
    os.path.join(DATA_PATH, "monash_darts_combined", "embedded_data", args.model_name)
)

# Remove the original datasets to save space
for dataset_name in ["monash_max_n_series_50_v2", "darts_v1"]:
    dataset_path = os.path.join(
        DATA_PATH, dataset_name, "embedded_data", args.model_name
    )
    if os.path.exists(dataset_path):
        os.system(f"rm -rf {dataset_path}")
