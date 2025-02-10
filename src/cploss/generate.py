import argparse
import json
import pathlib

import chess.engine
import chess.pgn
import numpy as np

from dataset.encoders.compact_board_tensor import board_to_compact_state

argparser = argparse.ArgumentParser()
argparser.add_argument("pgn")
argparser.add_argument("nodes")
argparser.add_argument("positions")
argparser.add_argument("savedir")

args = argparser.parse_args()

nodes = int(args.nodes)
positions = int(args.positions)

output_policy_boards = []
output_value_boards = []
output_positions = []

engine = chess.engine.SimpleEngine.popen_uci("stockfish")

pos = 0

with open(args.pgn) as pgn_file:
    while game := chess.pgn.read_game(pgn_file):
        board = game.board()

        for move in game.mainline_moves():
            actions = []
            moves_seq = list(board.legal_moves)
            if len(moves_seq) == 0:
                break

            moves_uci_seq = [move.uci() for move in moves_seq]

            for legal_move in moves_seq:
                board.push(legal_move)
                info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
                flip = not board.turn
                output_value_boards.append(board_to_compact_state(board, flip))
                board.pop()

                assert "score" in info
                score = info["score"].relative.score(mate_score=3000)
                actions.append((score, legal_move.uci()))

            best_score = min(actions)[0]

            normalized_actions = {move: score - best_score for score, move in actions}

            flip = not board.turn

            output_policy_boards.append(board_to_compact_state(board, flip))
            output_positions.append(
                {
                    "flip": flip,
                    "moves_uci_seq": moves_uci_seq,
                    "actions": normalized_actions,
                    "fen": board.fen(),
                }
            )

            if pos % 100 == 0:
                print(pos)

            pos += 1
            if pos >= positions:
                break

            board.push(move)

        if pos >= positions:
            break

engine.close()


assert len(output_policy_boards) == len(output_positions)
assert pos >= positions, "PGN doesn't contain enough positions"

save_dir = pathlib.Path(args.savedir)
save_dir.mkdir(exist_ok=True)

value_boards_file = save_dir / "value_boards.npz"
policy_boards_file = save_dir / "policy_boards.npz"
positions_file = save_dir / "positions.json"


np.savez_compressed(policy_boards_file, boards=output_policy_boards)
np.savez_compressed(value_boards_file, boards=output_value_boards)

with positions_file.open("w") as f:
    json.dump(output_positions, f)
