import pathlib
from io import BytesIO
from typing import Any

import cairosvg
import chess
import chess.svg
import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image

import wandb
from train.handmade_val_dataset.dataset import POSITIONS


class SampleGameGenerator(Callback):
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.table = wandb.Table(columns=["moves"])  # type: ignore

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        board = chess.Board()
        while board.result() == "*" and board.ply() != 120:
            policy, _ = pl_module.evaluate(board)
            bestmove = max(policy, key=lambda k: policy[k])
            board.push(bestmove)
        moves = " ".join(map(lambda m: m.uci(), board.move_stack))
        trainer.logger.log_text(key="sample_game", columns=["moves"], data=[[moves]])  # type: ignore


class BoardsEvaluator(Callback):
    def __init__(self, render_image: bool = False) -> None:
        self.positions = POSITIONS
        self.render_image = render_image

    @staticmethod
    def board_to_image(board: chess.Board) -> Any:
        svg = chess.svg.board(board)
        png_out = BytesIO()
        cairosvg.svg2png(svg.encode(), write_to=png_out)
        image = Image.open(png_out)
        return image

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        data = []
        COLUMNS = ["image"] * self.render_image + [
            "fen",
            "text",
            "type",
            "stockfish_cp",
            "stockfish_wdl",
            "stockfish_top3_lines",
            "model_v",
            "model_bestmove",
        ]

        for position in self.positions:
            board = chess.Board(fen=position["fen"])
            policy, value = pl_module.evaluate(board)  # TODO: eval in batch
            bestmove = max(policy, key=lambda k: policy[k])

            entry = [
                position["fen"],
                position["text"],
                position["type"],
                position["stockfish_cp"],
                position["stockfish_wdl"],
                position["stockfish_top3_lines"],
                value,
                bestmove.uci(),
            ]
            if self.render_image:
                entry = [wandb.Image(self.board_to_image(board))] + entry

            data.append(entry)

        trainer.logger.log_text(key="val_positions", columns=COLUMNS, data=data)  # type: ignore


class ValidationCheckpointer(Callback):
    def __init__(self) -> None:
        self.saves_counter = 0

    def save_model(self, pl_module: pl.LightningModule, label: str) -> None:
        self.saves_counter += 1
        filepath = pathlib.Path(f"{label}.ckpt").absolute()

        torch.save(pl_module, filepath)

        import wandb

        final_state = wandb.Artifact(name=label, type="valid-state")
        final_state.add_file(filepath)  # type: ignore
        wandb.log_artifact(final_state)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.save_model(pl_module, f"valid-state-{self.saves_counter}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.save_model(pl_module, "valid-state-final")
