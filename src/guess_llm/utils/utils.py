import torch
import numpy as np
import random
import lightning.pytorch as L
import wandb
import torch.nn as nn

from typing import Any
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42):
    random.seed(seed)  # Python built-in
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # CUDA
    torch.cuda.manual_seed_all(seed)  # All CUDA devices (if using multi-GPU)
    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True  # Force deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable benchmark for determinism


def flatten_config(cfg: DictConfig | dict) -> dict[str, Any]:
    """Flatten a Hydra/dict config into a dict. This is useful for logging.

    Args:
        cfg (DictConfig | dict): Config to flatten.

    Returns:
        dict[str, Any]: Flatenned config.
    """
    if "llm_models" in cfg and "api_key" in cfg["llm_models"]:
        del cfg["llm_models"]["api_key"]

    cfg_dict = (
        OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, DictConfig)
        else cfg
    )
    assert isinstance(cfg_dict, dict)

    cfg_flat: dict[str, Any] = {}
    for k, v in cfg_dict.items():
        # If the value is a dict, make a recursive call
        if isinstance(v, dict):
            if "_target_" in v:
                cfg_flat[k] = v["_target_"]  # type: ignore
            cfg_flat.update(**flatten_config(v))
        # If the value is a list, make a recursive call for each element
        elif isinstance(v, list):
            v_ls = []
            for v_i in v:
                if isinstance(v_i, dict):
                    if "_target_" in v_i:
                        v_ls.append(v_i["_target_"])
                    cfg_flat.update(**flatten_config(v_i))
            cfg_flat[k] = v_ls  # type: ignore
        # Exclude uninformative keys
        elif k not in {"_target_", "_partial_"}:
            cfg_flat[k] = v  # type: ignore
    return cfg_flat


def initialize_wandb(cfg: DictConfig) -> str | None:
    """Initialize wandb."""
    cfg_flat = flatten_config(cfg)
    wandb.init(project="guess_llm", config=cfg_flat)
    assert wandb.run is not None
    run_id = wandb.run.id
    assert isinstance(run_id, str)
    return run_id


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch of data to the specified device."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def build_mlp(input_dim, hidden_layers, hidden_dim, output_dim=None):
    if hidden_layers == 0:
        return nn.Identity()

    else:
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            input_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
