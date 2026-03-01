from guess_llm.utils.utils import set_seed
import numpy as np
from guess_llm.llm_utils.llm_no_scaling import serialize
from tqdm import tqdm
from datasets import Dataset
from pathlib import Path
from omegaconf import OmegaConf
import darts.datasets
import os
import pickle

DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "data")


def get_datasets():
    data_dir = os.path.join(DATA_PATH, "raw_monash/datasets/monash/")

    datasets = {}
    for file in os.listdir(data_dir):
        datasets[file.split(".")[0]] = pickle.load(
            open(os.path.join(data_dir, file), "rb")
        )

    for ds in datasets.keys():
        x = datasets[ds][0]
        train, test = zip(*x)

        # Concatenate the train and test sets along the last axis
        full = [
            np.concatenate([train[i], test[i]], dtype=np.float32)
            for i in range(len(train))
        ]

        # subsample such that there is less than 1000 time steps for each series
        for i in range(len(full)):
            if full[i].shape[0] > 1000:
                full[i] = full[i][:: int(full[i].shape[0] / 1000)]

        datasets[ds] = full

    return datasets


def get_monash(config):
    # Set the seed for reproducibility
    set_seed(config.random_seed)

    n_list = np.array([3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 35, 40]).astype(int)

    dataset = []
    monash = get_datasets()

    for ds in tqdm(monash.keys()):
        series = monash[ds]

        # subsample max max_n_series series
        if len(series) > config.max_n_series:
            idx = np.random.choice(len(series), config.max_n_series, replace=False)
            series = [series[i] for i in idx]

        for y in series:
            # print(f"Dataset: {ds}, min: {y.min()}, max: {y.max()}, mean: {y.mean()}, std: {y.std()}")
            t = y.shape[0]

            for n in n_list:
                if n >= t:
                    continue
                n_repeat = min(config.n_repeat, t - n)
                offset = np.floor((t - n) / n_repeat).astype(int)

                for i in range(n_repeat):
                    train = y[(i * offset) : (i * offset + n)].copy()
                    y_test = y[(i * offset + n)].copy()

                    input_str = serialize(train, precision=config.precision)

                    dataset.append(
                        {
                            "input_str": input_str,
                            "y_test": y_test,
                            "train": train,
                            "dataset": ds,
                        }
                    )

    ds = Dataset.from_list(dataset)
    print(f"Dataset length: {len(ds)}")
    ds.save_to_disk(f"{DATA_PATH}/{config.dataset_name}/ts_data")

    # return ds


if __name__ == "__main__":
    config = OmegaConf.load("configs/monash_dataset.yaml")
    get_monash(config)
