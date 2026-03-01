from datasets import load_from_disk
import os
import random
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path

random.seed(42)
np.random.seed(42)

project_root = str(Path(__file__).parent.parent.parent.parent)
DATA_PATH = os.path.join(project_root, "data")


def load_dataset(dataset_name, model_name, num_samples):
    dataset_path = os.path.join(DATA_PATH, dataset_name, "embedded_data", model_name)
    ds = load_from_disk(dataset_path)
    correct_indices = np.argwhere(np.array(ds["error"]) == 0).flatten().tolist()
    ds = ds.select(correct_indices)
    # Get rid of the rows where not enough samples were generated
    correct_indices = (
        np.argwhere(np.array([len(y) == num_samples for y in ds["y_pred"]]))
        .flatten()
        .tolist()
    )
    ds = ds.select(correct_indices)
    return ds


def get_random_splits(df, save_dir, save_name="random_splits"):
    # Generate random splits
    splits_df = df[["index"]].copy()
    splits_df["split"] = np.random.choice(
        ["train", "val", "test"], p=[0.8, 0.1, 0.1], size=len(splits_df)
    )
    splits_df.sort_values(by="index", inplace=True)
    splits_df.reset_index(drop=True, inplace=True)
    splits_df.to_csv(os.path.join(save_dir, save_name + ".csv"), index=False)


def get_splits_by_context_length(
    df, save_dir, save_name="context_length_splits", train_min=10, train_max=20
):
    df = df.copy()
    df["n_context"] = df["train"].apply(lambda x: len(x))
    splits_df = df[["index", "n_context"]].copy()
    splits_df["split"] = "test"
    train_index = splits_df[
        (splits_df.n_context >= train_min) & (splits_df.n_context <= train_max)
    ].index
    N_train = len(train_index)
    splits_df.loc[train_index, "split"] = np.random.choice(
        ["train", "val"], p=[0.9, 0.1], size=N_train
    )
    splits_df.sort_values(by="index", inplace=True)
    splits_df.reset_index(drop=True, inplace=True)
    splits_df.to_csv(os.path.join(save_dir, save_name + ".csv"), index=False)


def get_kfold_splits_by_dataset(df, save_dir, n_folds=5, save_name="kfold_splits"):
    # Generate 5 train-test validation folds
    datasets = df.dataset.unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for i, split in enumerate(kf.split(datasets)):
        train_index, _ = split
        train_datasets = datasets[train_index]
        splits_df = df[["index", "dataset"]].copy()
        splits_df["split"] = "test"
        train_index = splits_df[splits_df.dataset.isin(train_datasets)].index
        N_train = len(train_index)
        splits_df.loc[train_index, "split"] = np.random.choice(
            ["train", "val"], p=[0.9, 0.1], size=N_train
        )
        splits_df.to_csv(os.path.join(save_dir, f"{save_name}_{i}.csv"), index=False)


def mean_filter(df, threshold):
    # Filter the dataframe based on the mean of the specified column
    df["y_mean"] = df["y_pred"].apply(lambda x: np.mean(x))
    df = df[df["y_mean"].abs() < threshold]
    return df


def median_filter(df, threshold):
    # Filter the dataframe based on the median of the specified column
    df["y_median"] = df["y_pred"].apply(lambda x: np.median(x))
    df = df[df["y_median"].abs() < threshold]
    return df


def greedy_value_filter(df, threshold):
    # Filter the dataframe based on the greedy value of the specified column
    df = df[df["y_greedy"].abs() < threshold]
    return df


def relative_median_filter(df):
    df["y_median"] = df["y_pred"].apply(lambda x: np.median(x))
    iqr = df["y_median"].quantile(0.75) - df["y_median"].quantile(0.25)
    df = df[
        (df["y_median"] < df["y_median"].quantile(0.75) + 1.5 * iqr)
        & (df["y_median"] > df["y_median"].quantile(0.25) - 1.5 * iqr)
    ]
    return df


def remove_bitcoin(df):
    df = df[df["dataset"] != "bitcoin"]
    return df


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="multi_func_scale_1.0",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Meta-Llama-3-8B",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
    )
    parser.add_argument("--filter_types", type=str, default="strong_filter")
    parser.add_argument("--save_name", type=str, default="strong_filter")
    parser.add_argument(
        "--split_type",
        type=str,
        default="random",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    num_samples = args.num_samples
    dataset_name = args.dataset_name
    ds = load_dataset(dataset_name, args.model_name, num_samples)
    df = ds.to_pandas()
    df = df.reset_index(drop=True)
    df["index"] = df.index

    splits_dir = os.path.join(DATA_PATH, dataset_name, "splits", args.model_name)
    if not os.path.exists(splits_dir):
        os.makedirs(splits_dir)

    save_dir = splits_dir

    filtered_df = df.copy()

    if dataset_name.startswith("multi_func"):
        dataset_scale = dataset_name.split("_")[-1]
        if dataset_scale == "combined":
            dataset_scale = 10000.0
        elif dataset_scale == "context":
            dataset_scale = 1.0
        else:
            dataset_scale = float(dataset_scale)
            if dataset_scale == 0:
                dataset_scale = 1.0

        if "mean_fixed" in args.filter_types:
            filtered_df = mean_filter(filtered_df, dataset_scale)
        if "median_fixed" in args.filter_types:
            filtered_df = median_filter(filtered_df, dataset_scale)
        if "greedy_fixed" in args.filter_types:
            filtered_df = greedy_value_filter(filtered_df, dataset_scale)
    else:
        print("Applying filters: ", args.filter_types)
        if "relative_median_filter" in args.filter_types:
            print("Applying relative median filter")
            filtered_df = relative_median_filter(filtered_df)
        if "remove_bitcoin" in args.filter_types:
            print("Removing bitcoin dataset")
            filtered_df = remove_bitcoin(filtered_df)

    if args.split_type == "random":
        get_random_splits(filtered_df, save_dir, save_name=args.save_name)
    elif args.split_type == "context_length":
        get_splits_by_context_length(filtered_df, save_dir, save_name=args.save_name)
    elif args.split_type == "kfold_by_dataset":
        get_kfold_splits_by_dataset(
            filtered_df, save_dir, n_folds=args.n_folds, save_name=args.save_name
        )
