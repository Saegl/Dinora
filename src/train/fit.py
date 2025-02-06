import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Literal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner  # type: ignore

from dinora import PROJECT_ROOT
from train.datamodules import WandbDataModule
from train.train_callbacks import (
    BoardsEvaluator,
    SampleGameGenerator,
    ValidationCheckpointer,
)

warnings.filterwarnings("ignore", message="The dataloader, .* to improve performance.")
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


@dataclass
class Config:
    matmul_precision: Literal["highest", "high", "medium"]
    max_time: dict | None  # type: ignore
    max_epochs: int  # set -1 to ignore
    dataset_label: str
    z_weight: float
    q_weight: float

    tune_batch: bool
    batch_size: int  # will be overwritten if tune_batch = True

    tune_learning_rate: bool
    learning_rate: float  # will be overwritten if tune_learning_rate = True
    lr_scheduler_gamma: float  # Multiplicative factor for StepLR
    lr_scheduler_freq: int  # change each steps

    enable_checkpointing: bool
    checkpoint_train_time_interval: dict  # type: ignore

    enable_sample_game_generator: bool
    enable_boards_evaluator: bool
    enable_validation_checkpointer: bool

    log_every_n_steps: int

    val_check_interval: float | int

    limit_train_batches: int | None
    limit_val_batches: int | None
    limit_test_batches: int | None

    model_type: Literal["alphanet"]

    res_channels: int
    res_blocks: int
    policy_channels: int
    value_channels: int
    value_lin_channels: int

    # TODO: def from_json, also use this in jupyter notebook


def get_model(config: Config) -> pl.LightningModule:
    if config.model_type == "alphanet":
        from dinora.models.alphanet import AlphaNet

        return AlphaNet(
            filters=config.res_channels,
            res_blocks=config.res_blocks,
            policy_channels=config.policy_channels,
            value_channels=config.value_channels,
            value_fc_hidden=config.value_lin_channels,
            learning_rate=config.learning_rate,
            lr_scheduler_gamma=config.lr_scheduler_gamma,
            lr_scheduler_freq=config.lr_scheduler_freq,
        )
    else:
        raise ValueError("This model is not supported")


def fit(config: Config) -> None:
    torch.set_float32_matmul_precision(config.matmul_precision)
    max_time = timedelta(**config.max_time) if config.max_time else None

    callbacks: list[Callback] = []

    if config.enable_sample_game_generator:
        callbacks.append(SampleGameGenerator())

    if config.enable_boards_evaluator:
        callbacks.append(BoardsEvaluator())

    if config.enable_validation_checkpointer:
        callbacks.append(ValidationCheckpointer())

    if config.enable_checkpointing:
        checkpoint_train_time_interval = timedelta(
            **config.checkpoint_train_time_interval
        )
        mc = ModelCheckpoint(
            dirpath=PROJECT_ROOT / "checkpoints/models",
            filename="{epoch}epoch-{step}step",
            save_weights_only=True,
            train_time_interval=checkpoint_train_time_interval,
        )
        callbacks.append(mc)

    model = get_model(config)

    wandb_logger = WandbLogger(
        project="dinora-chess",
        log_model="all",  # save model weights to wandb
        config={"config_file": asdict(config)},
    )

    if not wandb_logger.experiment:
        # Calling .experiment prop causes wandb run to init
        # before creating WandbDataModule
        raise Exception("wandb run not initialized")

    datamodule = WandbDataModule(
        dataset_label=config.dataset_label,
        batch_size=config.batch_size,
        z_weight=config.z_weight,
        q_weight=config.q_weight,
    )

    trainer = pl.Trainer(
        max_time=max_time,
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=config.enable_checkpointing,
        default_root_dir=PROJECT_ROOT / "checkpoints",
        callbacks=callbacks,
        val_check_interval=config.val_check_interval,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
    )

    tuner = Tuner(trainer)

    if config.tune_batch:
        tuner.scale_batch_size(model, datamodule=datamodule)
        config.batch_size = datamodule.hparams.batch_size  # type: ignore

    if config.tune_learning_rate:
        tuner.lr_find(model, datamodule=datamodule)

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


def validate(config: Config) -> None:
    model = get_model(config)
    model.load_from_checkpoint("models/model-eliteq.ckpt")

    import wandb

    wandb.init(project="dinora-chess")

    datamodule = WandbDataModule(
        dataset_label=config.dataset_label,
        batch_size=config.batch_size,
        z_weight=config.z_weight,
        q_weight=config.q_weight,
    )

    trainer = pl.Trainer(
        limit_val_batches=config.limit_val_batches,
    )
    trainer.validate(model, datamodule)


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError:
        print("Provide path to config, examples at train_configs/dev.json")
        sys.exit(1)

    with open(path, encoding="utf") as f:
        config = Config(**json.load(f))
    fit(config)
