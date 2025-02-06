import datetime
import json
import pathlib
from dataclasses import dataclass
from io import TextIOWrapper

import chess
import chess.pgn
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from dataset.make import convert_dir as dataset_convert_dir
from dinora import PROJECT_ROOT
from dinora.engine import Engine
from dinora.models.alphanet import AlphaNet
from dinora.search.mcts.mcts import MCTS
from dinora.search.stoppers import NodesCount
from train.datamodules import CompactDataModule


@dataclass
class Config:
    first_n_moves: int = 15
    dirichlet_alpha: float = 0.3
    noise_fraction: float = 0.25

    ucb_score: float = 1.4

    generations: int = 10
    games_per_generation: int = 200
    epochs_per_generation: int = 1
    nodes_per_move: int = 15

    batch_size: int = 1024

    @staticmethod
    def from_file(filepath: pathlib.Path):
        with filepath.open("rt", encoding="utf8") as f:
            data = json.load(f)
        return Config(**data)


def selfplay(
    config: Config, game_number: int, model: AlphaNet, pgn_output: TextIOWrapper
) -> chess.pgn.Game:
    engine = Engine("mcts")
    engine._model = model
    assert isinstance(engine.searcher, MCTS)
    engine.searcher.params.first_n_moves = config.first_n_moves
    engine.searcher.params.noise_eps = config.noise_fraction
    engine.searcher.params.dirichlet_alpha = config.dirichlet_alpha

    current_datetime = datetime.datetime.now()
    utc_datetime = datetime.datetime.now(datetime.timezone.utc)
    game = chess.pgn.Game(
        headers={
            "Event": "RL selfplay",
            "Site": "Dinora engine",
            "Date": current_datetime.date().strftime(r"%Y.%m.%d"),
            "UTCDate": utc_datetime.date().strftime(r"%Y.%m.%d"),
            "Time": current_datetime.strftime("%H:%M:%S"),
            "UTCTime": utc_datetime.strftime("%H:%M:%S"),
            "Round": f"Game number {game_number}",
        }
    )
    node: chess.pgn.GameNode = game

    board = chess.Board()

    while not board.outcome(claim_draw=True) and board.ply() < 512 * 2:
        move = engine.get_best_move(board, NodesCount(config.nodes_per_move))
        node = node.add_variation(move)
        board.push(move)

    if board.ply() >= 512 * 2:
        result = "1/2-1/2"
    else:
        result = board.result(claim_draw=True)

    assert result in {"1-0", "0-1", "1/2-1/2"}

    game.headers["Result"] = result
    print(game, end="\n\n", flush=True, file=pgn_output)
    print(".", end="", flush=True)
    return game


def collect_games(
    config: Config, model: AlphaNet, output_dir: pathlib.Path
) -> pathlib.Path:
    print("STAGE: Game collection")
    pgn_file = output_dir / "games.pgn"

    white = 0
    black = 0
    draw = 0
    with pgn_file.open("w", encoding="utf8") as pgn_output:
        for game_number in range(config.games_per_generation):
            game = selfplay(config, game_number, model, pgn_output)
            if game.headers["Result"] == "1-0":
                white += 1
            elif game.headers["Result"] == "0-1":
                black += 1
            else:
                draw += 1

    print(f"\nDRAW: {draw}, WHITE: {white}, BLACK {black}")
    return pgn_file


def fit(config: Config, model, datamodule, generation_output_dir: pathlib.Path):
    print("STAGE: Fit")
    wandb_logger = WandbLogger(
        project="dinora-chess",
        # log_model="all",  # TODO: save model weights to wandb
        # config={"config_file": asdict(config)}, TODO: Add config
    )
    trainer = pl.Trainer(
        max_epochs=config.epochs_per_generation,
        logger=wandb_logger,
        default_root_dir=PROJECT_ROOT / "checkpoints",
    )
    trainer.fit(model=model, datamodule=datamodule)

    model_file = generation_output_dir / "model.ckpt"
    torch.save(model, model_file)


def start_rl(config: Config):
    model = AlphaNet()
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
