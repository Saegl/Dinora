import concurrent.futures
import json
import logging
import pathlib

import chess.engine
import numpy as np
from chess.engine import LOGGER as engine_logger

from dinora.encoders.board_representation import board_to_compact_state
from dinora.encoders.outcome import stockfish_value, wdl_index, z_value
from dinora.encoders.policy import policy_index
from dinora.pgntools import load_game_states


def convert_dir(
    pgns_dir: pathlib.Path,
    save_dir: pathlib.Path,
    files_count: int | None,
    q_nodes: int,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
) -> None:
    if not (0.9 <= train_percentage + val_percentage + test_percentage <= 1.1):
        raise ValueError("Train/val/test percentage doesn't sum to 1.0")

    pgns_dir = pgns_dir.absolute()
    save_dir = save_dir.absolute()
    save_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (path, save_dir / (path.name + "train.npz"), q_nodes)
        for path in pgns_dir.rglob("*.pgn")
        if path.is_file()
    ]

    chunks = run_parallel_convert(tasks[:files_count])
    generate_report(
        chunks,
        save_dir,
        train_percentage,
        val_percentage,
        test_percentage,
    )


def run_parallel_convert(
    tasks: list[tuple[pathlib.Path, pathlib.Path, int]],
) -> dict[str, int]:
    chunks: dict[str, int] = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_pgn_file, *args) for args in tasks]
        for future in concurrent.futures.as_completed(futures):
            save_path, states_count = future.result()
            print(f"File is converted: {save_path.name}, states {states_count}")
            chunks[save_path.name] = states_count
    return chunks


def generate_report(
    chunks: dict[str, int],
    save_dir: pathlib.Path,
    train_percentage: float,
    val_percentage: float,
    test_percentage: float,
) -> None:
    total_states = sum(chunks.values())
    if not (0.9 <= train_percentage + val_percentage + test_percentage <= 1.1):
        raise ValueError("Train/val/test percentage doesn't sum to 1.0")

    train_sum = total_states * train_percentage
    val_sum = total_states * val_percentage

    train_chunks, val_chunks, test_chunks = {}, {}, {}
    train_current, val_current, test_current = 0, 0, 0

    for name, states in chunks.items():
        if train_current == 0 or train_current + states <= train_sum:
            train_chunks[name] = states
            train_current += states
        elif val_current == 0 or val_current + states <= val_sum:
            val_chunks[name] = states
            val_current += states
        else:
            test_chunks[name] = states
            test_current += states

    report = {
        # TODO: include version
        # TODO: include `q_nodes`
        "train": train_chunks,
        "val": val_chunks,
        "test": test_chunks,
        "meta": {
            "train_percentage": train_current / total_states,
            "val_percentage": val_current / total_states,
            "test_percentage": test_current / total_states,
        },
    }

    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f)


def convert_pgn_file(
    pgn_path: pathlib.Path, save_path: pathlib.Path, q_nodes: int
) -> tuple[pathlib.Path, int]:
    print("Converting", pgn_path.name)

    tensors = {name: [] for name in ["boards", "policies", "wdls", "z_values"]}  # type: ignore
    engine = None

    if q_nodes > 0:
        engine_logger.setLevel(logging.ERROR)
        # TODO: remove this debug msg = DEBUG:asyncio:Using proactor: IocpProactor
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"UCI_ShowWDL": True})
        tensors["q_values"] = []

    try:
        with open(pgn_path, encoding="utf8", errors="ignore") as pgn:
            for game, board, move in load_game_states(pgn):
                flip = not board.turn

                tensors["boards"].append(board_to_compact_state(board, flip))
                tensors["policies"].append(policy_index(move, flip))
                tensors["wdls"].append(wdl_index(game, flip))
                tensors["z_values"].append(z_value(game, flip))

                if engine is not None:
                    tensors["q_values"].append(stockfish_value(board, engine, q_nodes))
    finally:
        if engine is not None:
            engine.close()

    tensors["boards"] = np.array(tensors["boards"], dtype=np.int64)  # type: ignore
    tensors["policies"] = np.array(tensors["policies"], dtype=np.int64)  # type: ignore
    tensors["wdls"] = np.array(tensors["wdls"], dtype=np.int64)  # type: ignore
    tensors["z_values"] = np.array(tensors["z_values"], dtype=np.float32).reshape(-1, 1)  # type: ignore

    if q_nodes > 0:
        tensors["q_values"] = np.array(tensors["q_values"], dtype=np.float32).reshape(  # type: ignore
            -1, 1
        )

    np.savez_compressed(save_path, **tensors)
    return save_path, len(tensors["boards"])
