from torch.utils.data import Dataset
from datasets import load_from_disk
import numpy as np
import pandas as pd
from time import time


class GuessDatasetTorch(Dataset):
    def __init__(self, ds, hidden_states_list):
        self.hidden_states_list = hidden_states_list
        self.ds = ds

    def update_hidden_states(self, hidden_states_list):
        self.hidden_states_list = hidden_states_list

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return_dict = {
            "x": item["hidden_states"][self.hidden_states_list],
            "y": item["y_pred"],
            "y_greedy": item["y_greedy"],
            "index": idx,
        }
        return return_dict


def load_datasets(
    dataset_path,
    split_path,
    hidden_states_list,
    num_samples,
    splits=["train", "val"],
    load_plain=False,
    subsample_frac=None,
):
    # load the dataset
    ds = load_from_disk(dataset_path)

    start_time = time()
    # Get rid of the rows where generation failed
    if "error" in ds.features.keys():
        correct_indices = np.argwhere(np.array(ds["error"]) == 0).flatten().tolist()
        ds = ds.select(correct_indices)

    # Get rid of the rows where not enough samples were generated
    correct_indices = (
        np.argwhere(np.array([len(y) == num_samples for y in ds["y_pred"]]))
        .flatten()
        .tolist()
    )  # TO DO: this should be changed to selecting y[:num_samples]
    ds = ds.select(correct_indices)
    
    # Train-validation split
    split_df = pd.read_csv(split_path)

    dataset_dict = {}

    for split in splits:
        start_time = time()
        split_indices = split_df[split_df["split"] == split]["index"].tolist()
        dataset_dict[split] = ds.select(split_indices)
        if not load_plain:
            start_time = time()
            dataset_dict[split].set_format(
                type="torch", columns=["y_pred", "hidden_states", "y_test", "y_greedy"]
            )
            if "nll" in dataset_dict[split].features.keys():
                dataset_dict[split].set_format(type="torch", columns=["nll"])
            if subsample_frac is not None:
                dataset_dict[split] = (
                    dataset_dict[split]
                    .shuffle(seed=42)
                    .select(list(range(int(len(dataset_dict[split]) * subsample_frac))))
                )
            start_time = time()
            dataset_dict[split] = GuessDatasetTorch(
                dataset_dict[split], hidden_states_list
            )

    return dataset_dict
