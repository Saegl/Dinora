import math
from os.path import dirname, realpath, join

import chess
import pylru
from tensorflow import keras
import numpy as np

from .board_representation import canon_input_planes
from .policy import move_lookup, flipped_move_lookup


MODELS_DIR = join(dirname(realpath(__file__)), "..", "models")
BEST_MODEL = join(MODELS_DIR, "latest.h5")


class ChessModel:
    def __init__(self, model_path=BEST_MODEL) -> None:
        self.model: keras.Model = keras.models.load_model(model_path)

    def model_out(self, board: chess.Board):
        flip = not board.turn
        plane = canon_input_planes(board.fen(), flip)
        plane = np.array([plane])
        model_output = self.model(plane, training=False)
        return model_output

    def raw_eval(self, board: chess.Board, softmax_temp):
        policy, value = self.model_out(board)
        # unwrap tensor
        policy = policy[0]
        value = float(value[0])

        # take only legal moves from policy
        t = softmax_temp
        moves = []
        policies = []
        lookup = move_lookup if board.turn else flipped_move_lookup
        for move in board.legal_moves:
            i = lookup[move]
            moves.append(move.uci())
            policies.append(math.exp(float(policy[i]) / t))

        # map to sum(policies) == 1
        s = sum(policies)
        policies_map = map(lambda e: math.exp(e / t) / s, policies)

        return dict(zip(moves, policies_map)), value

    def evaluate(self, board: chess.Board, softmax_temp):
        result = None
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)

        if result != None:
            if result == "1/2-1/2":
                return dict(), 0.0
            else:
                # Always return -1.0 when checkmated
                # and we are checkmated because it's our turn to move
                return dict(), -1.0
        return self.raw_eval(board, softmax_temp)


class ChessModelWithCache:
    def __init__(self, size=200000, model_path=BEST_MODEL):
        self.cache = pylru.lrucache(size)
        self.net = ChessModel(model_path=model_path)

    def evaluate(self, board: chess.Board, softmax_temp):
        epd = board.epd()
        if epd in self.cache:
            policy, value = self.cache[epd]
            return policy, value
        else:
            policy, value = self.net.evaluate(board, softmax_temp)
            self.cache[epd] = [policy, value]
            return policy, value
