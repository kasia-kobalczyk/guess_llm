import sys
import os
import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser
from pathlib import Path

from guess_llm.probe.models import *
from guess_llm.datasets.dataloaders import load_datasets
from guess_llm.utils.utils import batch_to_device

ROOT = str(Path(__file__).parent.parent.parent.parent)
# ROOT = "/mnt/pdata/knk25/guess_llm"
DATA_PATH = os.path.join(ROOT, "data")


def load_data(config, splits=["test"]):
    # Load the datasets
    if not hasattr(config.dataset, "model_name"):
        config.dataset.model_name = "Llama-2-7b-hf"
    dataset_path = os.path.join(
        DATA_PATH, config.dataset.name, "embedded_data", config.dataset.model_name
    )
    split_path = os.path.join(
        DATA_PATH,
        config.dataset.name,
        "splits",
        config.dataset.model_name,
        config.dataset.split_file,
    )
    hidden_states_list = config.dataset.hidden_states_list
    num_samples = config.dataset.num_samples

    return _load_data(
        dataset_path, split_path, hidden_states_list, num_samples, splits=splits
    )


def _load_data(
    dataset_path, split_path, hidden_states_list, num_samples, splits=["test"]
):
    # Load the datasets
    datasets = load_datasets(
        dataset_path=dataset_path,
        split_path=split_path,
        hidden_states_list=hidden_states_list,
        num_samples=num_samples,
        splits=splits,
    )

    # Load the dataset with all columns as pandas dataset (for meta data)
    meta_df = load_datasets(
        dataset_path=dataset_path,
        split_path=split_path,
        hidden_states_list=hidden_states_list,
        num_samples=num_samples,
        splits=splits,
        load_plain=True,
    )

    dataloaders = {}
    meta_dfs = {}

    for split in splits:
        # Create the dataloader
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=2048,
            num_workers=24,
            pin_memory=True,
            shuffle=False,
        )
        meta_dfs[split] = meta_df[split].to_pandas()

    return dataloaders, meta_dfs


def load_model(save_dir, config):
    checkpoint_save_path = os.path.join(ROOT, "saves", save_dir, "best_model.ckpt")
    if config.model.model_type == "mag_reg":
        model = LitMagnitudeRegressionPredictor.load_from_checkpoint(
            checkpoint_save_path, map_location=torch.device("cuda:0")
        ).model
    elif config.model.model_type == "quantile_mag_reg":
        model = LitQuantileConditionalPredictor.load_from_checkpoint(
            checkpoint_save_path
        ).model
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    return model


def load_config(save_dir):
    config_path = os.path.join(ROOT, "saves", save_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    config = OmegaConf.create(config)
    return config


def load_model_and_config(save_dir):
    config = load_config(save_dir)
    model = load_model(save_dir, config)
    return model, config


def eval_model(model, dataloader, device, model_type="gaussian"):
    # Run predictions
    batches = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = batch_to_device(batch, device=device)
            x, y, y_greedy = batch["x"], batch["y"], batch["y_greedy"]
            if model_type == "gaussian":
                mean, std = model(x)
                mean_orig, std_orig = model(x, rescale_to_orig=True)
                batch["pred_mean"] = mean
                batch["pred_std"] = std
                batch["pred_mean_orig"] = mean_orig
                batch["pred_std_orig"] = std_orig
                batch["MSE_loss"] = model.loss_function(
                    mean, std, y, regularise=False, mean_reduce=False, loss="MSE"
                )
                batch["MSE_std_loss"] = model.loss_function(
                    mean, std, y, regularise=False, mean_reduce=False, loss="MSE_std"
                )
                batch["KL_loss"] = model.loss_function(
                    mean, std, y, regularise=False, mean_reduce=False, loss="KL"
                )
            elif model_type in [
                "quantile",
                "quantile_mag_reg",
            ]:
                pred_quantiles = model.predict(x)
                batch["pred_quantiles"] = pred_quantiles
            if model_type in ["quantile_mag_reg"]:
                batch["pred_exp_quantiles"] = model.predict_expected(x)
            elif model_type == "greedy":
                if model.target == "mean":
                    pred_mean = model.predict(x)
                    batch["pred_mean"] = pred_mean
                    batch["mse"] = model.loss_function(
                        pred_mean, y_greedy, y, regularise=False, mean_reduce=False
                    )
                elif model.target == "median":
                    pred_median = model.predict(x)
                    batch["pred_median"] = pred_median
                    batch["mse"] = model.loss_function(
                        pred_median, y_greedy, y, regularise=False, mean_reduce=False
                    )
                elif model.target == "greedy":
                    pred_greedy = model.predict(x)
                    batch["pred_greedy"] = pred_greedy
                    batch["mse"] = model.loss_function(
                        pred_greedy, y_greedy, y, regularise=False, mean_reduce=False
                    )
                    for i in [1, 2, 3, 4]:
                        pred_dig = model.predict_n_digits(y_greedy, i)
                        batch[f"pred_{i}dig"] = pred_dig
                        batch[f"mse_{i}dig"] = model.loss_function(
                            pred_dig, y_greedy, y, regularise=False, mean_reduce=False
                        )
            elif model_type == "mag_reg":
                predictions = model.predict_all(x)
                batch["final_pred"] = predictions["final_pred"]
                batch["pred_order"] = predictions["pred_order"]
                batch["pred_scale"] = predictions["pred_scale"]
                batch["exp_pred"] = predictions["expected_pred"]

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            batches.append(batch)

    # Create results dataframe
    batches[0].keys()

    results_df = pd.DataFrame()
    for k in batches[0].keys():
        if batches[0][k].ndim == 2 and batches[0][k].shape[-1] == 1:
            results_df[k] = (
                torch.concat([batches[i][k] for i in range(len(batches))])
                .cpu()
                .numpy()
            )
        elif batches[0][k].ndim == 1:
            results_df[k] = (
                torch.concat([batches[i][k] for i in range(len(batches))])
                .cpu()
                .numpy()
            )
        else:
            results_df[k] = list(
                torch.concat([batches[i][k] for i in range(len(batches))])
                .cpu()
                .numpy()
            )

    return results_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="mag_reg_cond_testing_centre/mag_reg_57",
        help="Directory where the model and config are saved",
    )
    parser.add_argument(
        "--splits",
        type=list,
        default=["test"],
        help="List of splits to eval on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_dataset_model",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # Load the config
    save_dir = args.save_dir
    model, config = load_model_and_config(save_dir)
    model.to(args.device)

    if args.test_dataset_model is not None:
        config.dataset.model_name = args.test_dataset_model

    if args.test_dataset_name is None:
        # Load the test data
        dataloaders, meta_dfs = load_data(config, splits=args.splits)
        eval_dataset_name = config.dataset.name
    else:
        test_dataset_path = os.path.join(
            DATA_PATH,
            args.test_dataset_name,
            "embedded_data",
            config.dataset.model_name,
        )
        test_split_path = os.path.join(
            DATA_PATH,
            args.test_dataset_name,
            "splits",
            config.dataset.model_name,
            args.test_split_file + ".csv",
        )
        dataloaders, meta_dfs = _load_data(
            test_dataset_path,
            test_split_path,
            config.dataset.hidden_states_list,
            config.dataset.num_samples,
            splits=args.splits,
        )
        eval_dataset_name = args.test_dataset_name

    for split in args.splits:
        print(f"Evaluating {split} split")
        # Evaluate the model
        results_df = eval_model(
            model,
            dataloaders[split],
            device=args.device,
            model_type=config.model.model_type,
        )

        if "multi_func" in eval_dataset_name:
            if "context" in meta_dfs[split].columns:
                meta_cols = ["train", "y_test", "noise", "func", "input_str", "context"]
            else:
                meta_cols = ["train", "y_test", "noise", "func", "input_str"]
        elif "monash" in eval_dataset_name or "darts" in eval_dataset_name:
            meta_cols = ["train", "y_test", "dataset"]
        else:
            print("Unknown dataset name. Defaulting to meta_cols = ['train'] only.")
            meta_cols = ["train", "y_test"]
        if "series_id" in meta_dfs[split].columns:
            meta_cols.append("series_id")

        # Merge with meta data
        results_df = pd.merge(
            results_df,
            meta_dfs[split][meta_cols],
            left_on="index",
            right_index=True,
            how="left",
        )

        # Free up memory
        del dataloaders[split]
        del meta_dfs[split]

        # Add input statistics
        results_df["sample_mean"] = np.stack(results_df["y"].values).mean(axis=1)
        results_df["sample_median"] = np.median(
            np.stack(results_df["y"].values), axis=1
        )
        results_df["sample_std"] = np.stack(results_df["y"].values).std(axis=1)
        results_df["n_context"] = results_df["train"].apply(lambda x: len(x))

        # Serialize array columns:
        for c in results_df.columns:
            if isinstance(results_df[c].iloc[0], np.ndarray):
                results_df[c] = results_df[c].apply(lambda x: json.dumps(x.tolist()))
            elif isinstance(results_df[c].iloc[0], torch.Tensor):
                results_df[c] = results_df[c].apply(
                    lambda x: json.dumps(x.cpu().numpy().tolist())
                )
            elif isinstance(results_df[c].iloc[0], list):
                results_df[c] = results_df[c].apply(lambda x: json.dumps(x))

        # Drop columns for memory efficiency
        results_df.drop(columns=["x"], inplace=True)
        if args.test_dataset_name is not None or args.test_dataset_model is not None:
            save_name = (
                f"{split}_{args.test_dataset_name if args.test_dataset_name is not None else ''}"
                + f"_{args.test_dataset_model if args.test_dataset_model is not None else ''}"
                + f"_{args.test_split_file if args.test_split_file is not None else ''}"
                + f"_results.csv"
            )
        else:
            save_name = f"{split}_results.csv"
        results_df.to_csv(os.path.join(ROOT, "saves", save_dir, save_name), index=False)
