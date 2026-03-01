import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from multiprocessing import Pool, set_start_method
import logging
import hydra
import numpy as np
import os

from guess_llm.utils.utils import set_seed
from guess_llm.llm_utils.llm_no_scaling import llama_generate_samples
from guess_llm.llm_utils.llm_no_scaling import (
    llama_nll_no_scaling,
    get_hidden_states_no_scaling,
)
from transformers.utils.logging import disable_progress_bar


DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "data")
CONFIG_PATH = str(Path(__file__).parent.parent.parent.parent / "configs/datagen")

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)

print("Dataset will be saved to", DATA_PATH)


def initialize_model(model_name, device):
    print("Initializing model on", device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return model


def initialize_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    if tokenizer.eos_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_embeddings(examples, model_name, model, tokenizer, config):
    """
    Function that returns the embedding of the LLM.
    Examples is a dict with keys = column names; under each value is a list of batched values from the given column.
    It must return a dict, like here:
    """

    hidden_states = []
    y_preds = []
    y_greedys = []
    if config.get_nll:
        nlls = []
    errors = []

    for i in range(len(examples["input_str"])):
        try:
            y_pred = llama_generate_samples(
                input_str=examples["input_str"][i],
                n_samples=config.n_samples,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                temp=config.temp,
                top_p=config.top_p,
            )

            if len(y_pred) < config.n_samples * config.n_samples_sensitivity:
                raise ValueError(
                    f"Not enough correct samples generated for input {examples['input_str'][i]}."
                )

            y_greedy = llama_generate_samples(
                input_str=examples["input_str"][i],
                n_samples=1,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
            )

            if len(y_greedy) != 1:
                raise ValueError(
                    f"Not enough greedy samples generated for input {examples['input_str'][i]}."
                )

            if config.get_nll:
                target_strs = [f"{t:0.{config.precision}f}" for t in y_pred]

                nll = np.array(
                    [
                        llama_nll_no_scaling(
                            model, tokenizer, examples["input_str"][i], t
                        )["adjusted_nll"]
                        for t in target_strs
                    ]
                )

            hidden_state = get_hidden_states_no_scaling(
                model, tokenizer, examples["input_str"][i]
            )

        except Exception as e:
            print(e)
            # Log the error for debugging
            logging.info(
                f"Error generating sample for input {examples['input_str'][i]}."
            )

            hidden_states.append(
                [np.zeros((4096,), dtype=np.float16) for _ in range(33)]
            )
            if config.get_nll:
                nlls.append(np.zeros((config.n_samples)))
            y_preds.append([0.0 for _ in range(config.n_samples)])
            y_greedys.append([0.0])
            errors.append(1)
            continue

        hidden_states.append(hidden_state)
        if config.get_nll:
            nlls.append(nll)
        y_preds.append(y_pred)
        y_greedys.append(y_greedy)
        errors.append(0)

    final_dict = {
        "hidden_states": hidden_states,
        "y_pred": y_preds,
        "y_greedy": y_greedys,
        "error": errors,
    }

    if config.get_nll:
        final_dict["nll"] = nlls

    return final_dict


def embed_dataset(dataset, model_name, model, tokenizer, config):
    print("Embedding dataset on device", model.device)
    return dataset.map(
        lambda x: get_embeddings(x, model_name, model, tokenizer, config),
        batched=True,
        batch_size=8,
    )


@hydra.main(config_path=CONFIG_PATH, config_name="multi_func_10000.0_dataset")
def get_llm_data(config: DictConfig):

    # Disable strict mode to add new keys
    OmegaConf.set_struct(config, False)
    set_seed(config.random_seed)

    set_start_method("spawn", force=True)
    devices = [f"cuda:{i}" for i in config.devices_idx]

    ts_data_path = f"{DATA_PATH}/{config.dataset_name}/ts_data"
    if not os.path.exists(ts_data_path):
        logging.info("TS dataset does not exist yet. Generating the time series data..")
        hydra.utils.call(config.time_series_func)
    else:
        logging.info("TS dataset already exists. Loading the time series data..")

    ds = datasets.load_from_disk(f"{DATA_PATH}/{config.dataset_name}/ts_data")

    # Save hydra config
    config_path = f"{DATA_PATH}/{config.dataset_name}/config.yaml"
    OmegaConf.save(config, config_path)

    n_devices = len(devices)
    n = len(ds)
    chunked_datasets = [
        ds.select(list(range(i, n, n_devices))) for i in range(n_devices)
    ]

    logging.info("Loading the tokenizers and the models ...")
    model_name = config.model_name
    models = [initialize_model(model_name, device) for device in devices]
    tokenizer = initialize_tokenizer(model_name)
    disable_progress_bar()

    logging.info("Generating the embeddings ...")
    with Pool(n_devices) as pool:
        embedded_datasets = pool.starmap(
            embed_dataset,
            [
                (ds, model_name, model, tokenizer, config)
                for ds, model in zip(chunked_datasets, models)
            ],
        )

    # Concatenate the results from all devices
    logging.info("Concatenating the datasets ...")
    embedded_dataset = datasets.concatenate_datasets(embedded_datasets)
    save_path = f"{DATA_PATH}/{config.dataset_name}/embedded_data/{config.model_name.split('/')[-1]}"
    logging.info(f"Saving the dataset to: {save_path}")
    embedded_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    get_llm_data()
