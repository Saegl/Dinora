import pathlib
import time

import chess

from dinora.engine import Engine
from dinora.search.stoppers import MoveTime


def selfplay(model_name: str, weights: pathlib.Path, device: str) -> None:
    engine = Engine("ext_mcts", model_name, weights, device)
    engine.load_model()
    engine.get_best_move(chess.Board(), MoveTime(100))  # Warmup
    print("Engine loaded")

    board = chess.Board()

    total_visits = 0
    total_moves = 0
    start_time = time.time()

    try:
        while not board.is_game_over(claim_draw=True):
            move = engine.get_best_move(board, MoveTime(1000))
            visits = 0  # FIXME: use real visits, broken after new `searcher`
            print(f"{total_moves}. Move {move.uci()}\tnodes {visits}")

            total_moves += 1
            total_visits += visits
            board.push(move)
    except KeyboardInterrupt:
        print("Bench canceled by keyboard interrupt")
    finally:
        elapsed_time = time.time() - start_time
        average_nps = total_visits / elapsed_time
        print(
            f"Average nps {average_nps:.2f}"
            f"\tTotal moves {total_moves}"
            f"\tElapsed_time {elapsed_time}"
        )
