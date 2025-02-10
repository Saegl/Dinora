import logging
from collections.abc import Iterator
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt
from chess import Board, Move
from chess.pgn import Game

logging.basicConfig(level=logging.DEBUG)

npf32 = npt.NDArray[np.float32]
npuint64 = npt.NDArray[np.uint64]


def load_chess_games(pgn: TextIO) -> Iterator[chess.pgn.Game]:
    def is_supported_variant(game: chess.pgn.Game) -> bool:
        return game.headers.get("Variant", "Standard") == "Standard"

    game = chess.pgn.read_game(pgn)
    while game:
        if is_supported_variant(game):
            yield game

        game = chess.pgn.read_game(pgn)


def load_game_states(pgn: TextIO) -> Iterator[tuple[Game, Board, Move]]:
    for game in load_chess_games(pgn):
        board = game.board()
        for move in game.mainline_moves():
            yield game, board, move

            try:
                board.push(move)
            except AssertionError:
                logging.warning(
                    f"Broken game found, can't make a move {move}. Skipping game"
                )
                break
