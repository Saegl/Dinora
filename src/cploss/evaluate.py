import argparse
import json
import pathlib

import chess
import numpy as np
import torch

from dataset.encoders.compact_board_tensor import compact_state_to_board_tensor
from dinora.encoders.policy import extract_logit
from dinora.engine import Engine
from dinora.models import model_selector
from dinora.models.alphanet import AlphaNet
from dinora.search.stoppers import MoveTime, NodesCount

device = "cuda"


def calc_policy_cploss(model, positions, policy_boards, batch_size: int) -> float:
    total_cploss = 0
    total_positions = len(positions)

    for batch_start in range(0, total_positions, batch_size):
        batch_end = batch_start + batch_size
        batch_positions = positions[batch_start:batch_end]
        batch_boards = policy_boards[batch_start:batch_end]

        boards_tensor = torch.from_numpy(
            np.array([compact_state_to_board_tensor(b) for b in batch_boards])
        ).to(device)

        with torch.no_grad():
            policy, _ = model(boards_tensor)

        batch_cploss = 0
        for i, position in enumerate(batch_positions):
            flip = position["flip"]
            uci_moves = list(position["actions"])

            move_logits = np.array(
                [
                    extract_logit(policy[i], chess.Move.from_uci(move), flip)
                    for move in uci_moves
                ]
            )
            best_move_index = move_logits.argmax()
            best_move = uci_moves[best_move_index]
            cploss = position["actions"][best_move]

            batch_cploss += cploss

        total_cploss += batch_cploss

    return total_cploss / total_positions


def calc_value_cploss(model, positions, value_boards, batch_size: int) -> float:
    def load_batch(batch_offset: int):
        value_boards_np = np.array(
            [
                compact_state_to_board_tensor(b)
                for b in value_boards[batch_offset : batch_offset + batch_size]
            ]
        )
        boards = torch.from_numpy(value_boards_np).to(device)

        with torch.no_grad():
            _, value = model(boards)

        value_count = len(value)
        value = value.cpu().numpy().reshape(value_count)
        return value

    batch_offset = 0
    value = load_batch(batch_offset)

    def get_value_positions(pos_start: int, pos_end: int):
        nonlocal batch_offset, value
        if pos_start > pos_end:
            return np.array([])

        batch_pos_start = pos_start - batch_offset
        batch_pos_end = min(batch_size, pos_end - batch_offset)

        prefix = value[batch_pos_start:batch_pos_end]

        if pos_end - batch_offset <= batch_size:
            return prefix

        batch_offset += batch_size
        value = load_batch(batch_offset)

        return np.concat([prefix, get_value_positions(batch_offset, pos_end)])

    total_cploss = 0
    moves_offset = 0
    for i in range(len(positions)):
        position = positions[i]
        moves_uci_seq = position["moves_uci_seq"]

        pos_start = moves_offset
        pos_end = moves_offset + len(moves_uci_seq)

        values = -get_value_positions(pos_start, pos_end)

        best_index = values.argmax()
        best_move = moves_uci_seq[best_index]

        assert len(moves_uci_seq) == len(values)

        move_loss = position["actions"][best_move]
        total_cploss += move_loss

        moves_offset += len(moves_uci_seq)

    return total_cploss / len(positions)


def calc_engine_cploss(
    model: AlphaNet, positions: dict, searcher: str, stopper_creator
) -> float:
    engine = Engine(searcher=searcher)
    engine._model = model

    total_cploss = 0
    for position in positions:
        board = chess.Board(fen=position["fen"])
        move = engine.get_best_move(board, stopper=stopper_creator())
        move_loss = position["actions"][move.uci()]
        total_cploss += move_loss

    return total_cploss / len(positions)


def make_stopper_creator(movetime, nodes):
    if nodes is not None:
        return lambda: NodesCount(nodes)
    else:
        return lambda: MoveTime(int(1000 * movetime))


def load_cploss(loaddir: pathlib.Path, count: int):
    value_boards_file = loaddir / "value_boards.npz"
    boards_file = loaddir / "policy_boards.npz"
    positions_file = loaddir / "positions.json"

    value_boards = np.load(value_boards_file)["boards"]
    policy_boards = np.load(boards_file)["boards"]
    positions = json.load(positions_file.open())

    return value_boards, policy_boards, positions[:count]


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_path")
    argparser.add_argument("batch_size")
    argparser.add_argument("loaddir")
    argparser.add_argument("--max_positions", default=99_999, type=int)
    argparser.add_argument("--searcher", default="auto")
    argparser.add_argument("--movetime", default=1.0, type=float)
    argparser.add_argument("--nodes", type=int)

    args = argparser.parse_args()

    model_path = pathlib.Path(args.model_path)
    batch_size = int(args.batch_size)
    max_positions = args.max_positions
    searcher = args.searcher
    movetime = args.movetime
    nodes = args.nodes

    stopper_creator = make_stopper_creator(movetime, nodes)

    print(f"Chosen Engine {searcher} {stopper_creator()}")

    print("Model loading")
    model = model_selector("alphanet", model_path, "cuda")
    print("Model loaded")

    assert isinstance(model, AlphaNet)

    loaddir = pathlib.Path(args.loaddir)
    value_boards, policy_boards, positions = load_cploss(loaddir, max_positions)

    print(f"Positions: {len(positions)}")

    value_cploss = calc_value_cploss(model, positions, value_boards, batch_size)
    print("Value loss", value_cploss)

    policy_cploss = calc_policy_cploss(model, positions, policy_boards, batch_size)
    print("Policy loss", policy_cploss)

    engine_cploss = calc_engine_cploss(model, positions, searcher, stopper_creator)
    print("Engine loss", engine_cploss)


if __name__ == "__main__":
    main()
