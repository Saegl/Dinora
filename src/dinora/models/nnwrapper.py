from abc import ABC, abstractmethod

import chess
import numpy as np
import numpy.typing as npt

from dinora.encoders.policy import extract_logit
from dinora.models.base import BaseModel, Evaluation

npf32 = npt.NDArray[np.float32]


def softmax(x: npf32, tau: float = 1.0) -> npf32:
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()  # type: ignore


class NNWrapper(BaseModel, ABC):
    """
    This class converts raw NN outputs in numpy into
    `BaseModel` compatible interface
    """

    @abstractmethod
    def raw_outputs(self, board: chess.Board) -> tuple[npf32, npf32]:
        """
        NN outputs as numpy arrays
        priors numpy of shape (1880,)
        state value of shape (1,)
        """

    def nn_evaluate(self, board: chess.Board) -> Evaluation:
        raw_policy, raw_value = self.raw_outputs(board)

        moves = list(board.legal_moves)
        move_logits = [
            extract_logit(raw_policy, move, not board.turn) for move in moves
        ]

        move_priors = softmax(np.array(move_logits))
        priors = dict(zip(moves, move_priors))

        return priors, float(raw_value[0])

    def evaluate(self, board: chess.Board) -> Evaluation:
        return self.nn_evaluate(board)

    def reset(self) -> None:
        pass
