import json
import pathlib
import time
from dataclasses import dataclass
from datetime import timedelta

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from dataset.make import convert_dir as dataset_convert_dir
from dinora import PROJECT_ROOT
from dinora.models.alphanet import AlphaNet
from rl.selfplay import analyze_pgn, selfplay
from train.datamodules import CompactDataModule


@dataclass
class Config:
    opening_noise_moves: int = 15
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25

    cpuct: float = 1.4

    generations: int = 10
    games_per_generation: int = 200
    epochs_per_generation: int = 1
    nodes_per_move: int = 15

    batch_size: int = 1024
    learning_rate: float = 0.001

    @staticmethod
    def from_file(filepath: pathlib.Path):
        with filepath.open("rt", encoding="utf8") as f:
            data = json.load(f)
        return Config(**data)


def collect_games(
    config: Config, model: AlphaNet, output_dir: pathlib.Path
) -> pathlib.Path:
    print("STAGE: Game collection")
    start_time = time.time()
    pgn_file = output_dir / "games.pgn"

    selfplay(
        model,
        config.games_per_generation,
        config.nodes_per_move,
        config.batch_size,
        config.cpuct,
        config.opening_noise_moves,
        config.dirichlet_alpha,
        config.noise_fraction,
        pgn_file,
    )
    analyze_pgn(pgn_file)
    print(
        f"STAGE: Game collection took {timedelta(seconds=int(time.time() - start_time))}"
    )

    return pgn_file


def fit(config: Config, model, datamodule, generation_output_dir: pathlib.Path):
    print("STAGE: Fit")
    start_time = time.time()
    wandb_logger = WandbLogger(
        project="dinora-chess",
        # log_model="all",  # TODO: save model weights to wandb
        # config={"config_file": asdict(config)}, # TODO: save config
    )
    trainer = pl.Trainer(
        max_epochs=config.epochs_per_generation,
        logger=wandb_logger,
        default_root_dir=PROJECT_ROOT / "checkpoints",
    )
    trainer.fit(model=model, datamodule=datamodule)

    model_file = generation_output_dir / "model.ckpt"
    torch.save(model, model_file)

    print(f"STAGE: Fit took {timedelta(seconds=int(time.time() - start_time))}")


def start_rl(config: Config):
    model = AlphaNet(learning_rate=config.learning_rate)
    output_dir = pathlib.Path.cwd() / "data" / "rl_data"

    wandb.init(job_type="rl", project="dinora-chess")

    for generation in range(config.generations):
        generation_output_dir = output_dir / f"generation-{generation}"

        pgns_output_dir = generation_output_dir / "pgns"
        pgns_output_dir.mkdir(parents=True, exist_ok=True)

        collect_games(config, model, pgns_output_dir)

        dataset_dir = generation_output_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_convert_dir(
            pgns_output_dir,
            dataset_dir,
            None,
            q_nodes=0,
            train_percentage=1.0,
            val_percentage=0.0,
            test_percentage=0.0,
        )

        datamodule = CompactDataModule(
            dataset_dir, z_weight=1.0, q_weight=0.0, batch_size=config.batch_size
        )
        fit(config, model, datamodule, generation_output_dir)
