from dataclasses import dataclass

import chess

from dinora.models import BaseModel
from dinora.search.base import BaseSearcher
from dinora.search.stoppers import Stopper


@dataclass
class Params:
    pass


class OneMove(BaseSearcher):
    def __init__(self) -> None:
        self._params = Params()

    @property
    def params(self):
        return self._params

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        priors, _ = evaluator.evaluate(board)
        maxval = 0.0
        bestmove = None
        for move, prob in priors.items():
            if prob > maxval:
                maxval = prob
                bestmove = move

        if not bestmove:
            raise Exception("Cannot find any move")

        return bestmove
