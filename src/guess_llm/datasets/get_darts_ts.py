from guess_llm.utils.utils import set_seed
import numpy as np
from guess_llm.llm_utils.llm_no_scaling import serialize
from tqdm import tqdm
from datasets import Dataset
from pathlib import Path
from omegaconf import OmegaConf
import darts.datasets

DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "data")
DATASETS = [
    "AirPassengersDataset",
    "AusBeerDataset",
    "GasRateCO2Dataset",
    "MonthlyMilkDataset",
    "SunspotsDataset",
    "WineDataset",
    "WoolyDataset",
    "HeartRateDataset",
]


def get_dataset(dsname):
    series = getattr(darts.datasets, dsname)().load().all_values()

    if dsname == "SunspotsDataset":
        series = series[::4]
    if dsname == "HeartRateDataset":
        series = series[::2]

    return series


def get_darts(config):
    # Set the seed for reproducibility
    set_seed(config.random_seed)
    n_list = np.array(
        [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    ).astype(int)
    dataset = []
    series_id = 0
    for ds in tqdm(DATASETS):
        series = get_dataset(ds)
        t, dim, _ = series.shape
        for d in range(dim):
            y = series[:, d, :].flatten()
            print(
                f"Dataset: {ds}, min: {y.min()}, max: {y.max()}, mean: {y.mean()}, std: {y.std()}"
            )
            for _ in range(config.n_repeat):
                if len(y) > n_list[-1]:
                    start = np.random.randint(0, len(y) - n_list[-1])
                    for n in n_list:
                        train = y[start : start + n].copy()
                        y_test = y[start + n :].copy()
                        input_str = serialize(train, precision=config.precision)
                        dataset.append(
                            {
                                "input_str": input_str,
                                "y_test": y_test,
                                "orig_y": y[start:],
                                "train": train,
                                "dataset": ds,
                                "series_id": series_id,
                            }
                        )
                    series_id += 1

    ds = Dataset.from_list(dataset)
    print(f"Dataset length: {len(ds)}")
    ds.save_to_disk(f"{DATA_PATH}/{config.dataset_name}/ts_data")

    # return ds


if __name__ == "__main__":
    config = OmegaConf.load("configs/darts_dataset.yaml")
    get_darts(config)
