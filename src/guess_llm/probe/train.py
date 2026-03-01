import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
from guess_llm.utils.utils import flatten_config
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import os
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from guess_llm.probe.models import *
from guess_llm.datasets.dataloaders import load_datasets
from guess_llm.utils.utils import set_seed


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


DATA_PATH = str(Path(__file__).parent.parent.parent.parent / "data")
CONFIG_PATH = str(Path(__file__).parent.parent.parent.parent / "configs")
SAVE_PATH = str(Path(__file__).parent.parent.parent.parent / "saves")

OmegaConf.register_new_resolver("replace", lambda s, a, b: s.replace(a, b))


# Main training function
@hydra.main(config_path=CONFIG_PATH, config_name="train")
def train(config: DictConfig):
    # Disable strict mode to add new keys
    OmegaConf.set_struct(config, False)

    run_name_prefix = getattr(config.wandb, "run_name_prefix", "run")
    save_dir = f"{SAVE_PATH}/{config.wandb.project_name}"

    if os.path.exists(save_dir):
        save_no = [
            int(x.split("_")[-1])
            for x in os.listdir(save_dir)
            if x.startswith(run_name_prefix)
        ]
        save_no = max(save_no) + 1 if len(save_no) > 0 else 0
    else:
        os.makedirs(save_dir)
        save_no = 0

    save_dir = os.path.join(save_dir, f"{run_name_prefix}_{save_no}")

    # Create the saving directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize logging
    if config.wandb.log:
        wandb.config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        cfg_flat = flatten_config(config)
        run_id = wandb.init(
            project=config.wandb.project_name,
            name=f"{run_name_prefix}_{save_no}",
            dir=save_dir,
            config=cfg_flat,
        )

        logging.info(f"Wandb run ID: {run_id}")
        wandb.log({"save_dir": str(save_dir)})

    set_seed(config.random_seed)

    # Load the datasets
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

    datasets = load_datasets(
        dataset_path=dataset_path,
        split_path=split_path,
        hidden_states_list=hidden_states_list,
        num_samples=config.dataset.num_samples,
        splits=["train", "val"],
    )
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    logging.info(f"Train dataset length: {len(train_dataset)}")
    logging.info(f"Validation dataset length: {len(val_dataset)}")
    logging.info(f"Dimensions of the input data: {train_dataset[0]['x'].shape}")

    train_y_min = train_dataset.ds["y_pred"].min().item()
    train_y_greedy_min = train_dataset.ds["y_greedy"].min().item()

    if config.model.log_scaling and config.model.standard_scaling:
        train_y_mean = (
            torch.log(train_dataset.ds["y_pred"] - train_y_min + 1e-3).mean().item()
        )
        train_y_std = (
            torch.log(train_dataset.ds["y_pred"] - train_y_min + 1e-3).std().item()
        )
        train_y_greedy_mean = (
            torch.log(train_dataset.ds["y_greedy"] - train_y_greedy_min + 1e-3)
            .mean()
            .item()
        )
        train_y_greedy_std = (
            torch.log(train_dataset.ds["y_greedy"] - train_y_greedy_min + 1e-3)
            .std()
            .item()
        )
    elif config.model.standard_scaling:
        train_y_mean = train_dataset.ds["y_pred"].mean().item()
        train_y_std = train_dataset.ds["y_pred"].std().item()
        train_y_greedy_mean = train_dataset.ds["y_greedy"].mean().item()
        train_y_greedy_std = train_dataset.ds["y_greedy"].std().item()
    else:
        train_y_mean = 0
        train_y_std = 1
        train_y_min = 0
        train_y_greedy_mean = 0
        train_y_greedy_std = 1
        train_y_greedy_min = 0

    config.model.y_mean = train_y_mean
    config.model.y_std = train_y_std
    config.model.y_min = train_y_min
    config.model.y_greedy_mean = train_y_greedy_mean
    config.model.y_greedy_std = train_y_greedy_std
    config.model.y_greedy_min = train_y_greedy_min

    config.model.n_hidden_states = len(config.dataset.hidden_states_list)
    config.model.model_name = config.dataset.model_name

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # Initialize the model
    if config.model.model_type == "mag_reg":
        model = LitMagnitudeRegressionPredictor(config.model)
    elif config.model.model_type == "quantile_mag_reg":
        model = LitQuantileConditionalPredictor(config.model)
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")

    # setting up the training
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config.model.patience, mode="min", verbose=False
    )

    # Wandb logger setup
    wandb_logger = WandbLogger(
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_model",
        save_top_k=1,
        dirpath=save_dir,
    )

    # Save wandb config to save_dir
    with open(f"{save_dir}/config.yaml", "w") as f:
        OmegaConf.save(config, f)

    callbacks = [checkpoint_callback]

    if config.model.patience > 0:
        callbacks.append(early_stopping)  # type: ignore

    # PyTorch Lightning Trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.model.max_epochs,
        accelerator="gpu",
        devices=config.model.devices,
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=1,
    )

    # Start training
    # Two-phase training for MagnitudeRegressionPredictor
    if "training" in config.model and config.model.training == "separate":

        # ======== PHASE 1 ===============

        # setting up the training
        early_stopping1 = EarlyStopping(
            monitor="val_loss",
            patience=config.model.patience,
            mode="min",
            verbose=False,
        )

        # Wandb logger setup
        wandb_logger1 = WandbLogger(
            log_model=True,
        )

        checkpoint_callback1 = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best_class_model",
            save_top_k=1,
            dirpath=save_dir,
        )

        callbacks1 = [checkpoint_callback1]

        if config.model.patience > 0:
            callbacks1.append(early_stopping1)  # type: ignore

        # PyTorch Lightning Trainer
        trainer1 = L.Trainer(
            logger=wandb_logger1,
            max_epochs=config.model.max_epochs,
            accelerator="gpu",
            devices=config.model.devices,
            callbacks=callbacks1,  # type: ignore
            log_every_n_steps=1,
        )

        # Phase 1: Train classification head only
        model.model.unfreeze_classification_head()
        model.model.freeze_regression_head()
        model.model.beta = 0.0  # Disable regression loss
        print("Phase 1: Training classification head only")
        trainer1.fit(model, train_loader, val_loader)

        del trainer1
        torch.cuda.empty_cache()

        best_phase1_path = os.path.join(save_dir, "best_class_model.ckpt")

        # ======== PHASE 2 ===============
        if config.model.model_type == "mag_reg":
            model = LitMagnitudeRegressionPredictor.load_from_checkpoint(
                best_phase1_path
            )
        elif config.model.model_type == "quantile_mag_reg":
            model = LitQuantileConditionalPredictor.load_from_checkpoint(
                best_phase1_path
            )

        # Setting up the training
        early_stopping2 = EarlyStopping(
            monitor="val_loss",
            patience=config.model.patience,
            mode="min",
            verbose=False,
        )

        # Wandb logger setup
        wandb_logger2 = WandbLogger(
            log_model=True,
        )

        checkpoint_callback2 = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best_model",
            save_top_k=1,
            dirpath=save_dir,
        )

        callbacks2 = [checkpoint_callback2]

        if config.model.patience > 0:
            callbacks2.append(early_stopping2)  # type: ignore

        # PyTorch Lightning Trainer
        trainer2 = L.Trainer(
            logger=wandb_logger2,
            max_epochs=config.model.max_epochs,
            accelerator="gpu",
            devices=config.model.devices,
            callbacks=callbacks2,  # type: ignore
            log_every_n_steps=1,
        )
        # Phase 2: Train regression head only
        model.model.freeze_classification_head()
        model.model.unfreeze_regression_head()
        model.model.alpha = 0.0  # Disable classification loss
        model.model.beta = config.model.beta  # Enable regression loss
        print("Phase 2: Training regression head only")
        trainer2.fit(model, train_loader, val_loader)

    else:
        # Standard training
        trainer.fit(model, train_loader, val_loader)

    wandb.finish()


if __name__ == "__main__":
    train()
