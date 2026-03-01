from guess_llm.utils.utils import set_seed
import numpy as np
from guess_llm.llm_utils.llm_no_scaling import serialize
from tqdm import tqdm
from datasets import Dataset
from pathlib import Path
from omegaconf import OmegaConf

DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "data")

# All function fit into the range (-1, 1)
func_dict = {
    # "exp": lambda x: np.exp(x - 1),
    # "log": lambda x: np.log(x - 1),
    # "sigmoid": lambda x: 10 / (1 + np.exp(-(x))),
    "sin": lambda x: np.sin(x),
    "linear_sin": lambda x: 0.2 * np.sin(x) + x / 450,
    "sinc": lambda x: np.sinc(x),
    "xsine": lambda x: np.sin((x - 30)) * (x - 30) / 50,
    "beat": lambda x: np.sin(x) * np.sin(x / 2),
    "gaussian_wave": lambda x: np.exp(-((x - 2) ** 2) / 2)
    * np.cos((x - 2) * 2 * np.pi * 5),
    "random": lambda x: np.random.rand(len(x)) * 2 - 1,
}

a_range_dict = {
    # "exp": (0.05, 0.08),
    # "log": (0.01, 1),
    # "sigmoid": (0.5, 6),
    "sin": (0.5, 6),
    "linear_sin": (0.5, 6),
    "sinc": (0.05, 0.2),
    "xsine": (0.5, 1.3),
    "beat": (0.1, 6),
    "gaussian_wave": (0.01, 0.1),
    "random": (0, 1),
}


def get_time_series(config):
    # Set the seed for reproducibility
    set_seed(config.random_seed)

    n_list = np.array([3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 35, 40]).astype(int)
    x = np.linspace(0, 60, 120)

    assert config.n_repeat < (120 - n_list[-1]), "Too large number of repeats."

    dataset = []

    series_id = 0
    for func in tqdm(config.func_list):
        fx = func_dict[func]
        a_list = np.linspace(
            a_range_dict[func][0],
            a_range_dict[func][1],
            config.n_a,
        )
        for a in a_list:
            raw_y = fx(a * x)
            for noise in config.noise_list:
                b = (
                    np.random.rand(1) * config.max_vert_stretch
                    if config.max_vert_stretch > 0
                    else 1.0
                )
                d = config.max_vert_translation * (2 * np.random.rand(1) - 1.0)
                noisy_y = (raw_y + np.random.normal(0, noise, len(raw_y))) * b + d
                orig_y = raw_y * b + d
                for _ in range(config.n_repeat):
                    start = np.random.randint(0, 120 - n_list[-1])
                    for n in n_list:
                        train = noisy_y[start : (start + n)].copy()
                        y_test = noisy_y[(start + n) :].copy()
                        input_str = serialize(train, precision=config.precision)
                        dataset.append(
                            {
                                "input_str": input_str,
                                "y_test": y_test,
                                "train": train,
                                "orig_y": orig_y[start:],
                                "noisy_y": noisy_y[start:],
                                "noise": noise,
                                "vert_stretch": b,
                                "displacement": d,
                                "func": func,
                                "series_id": series_id,
                            }
                        )
                    series_id += 1

    ds = Dataset.from_list(dataset)
    print(f"Dataset length: {len(ds)}")
    ds.save_to_disk(f"{DATA_PATH}/{config.dataset_name}/ts_data")


if __name__ == "__main__":
    config = OmegaConf.load("configs/multi_func_dataset.yaml")
    get_time_series(config)
