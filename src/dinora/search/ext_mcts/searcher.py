import chess

from dinora.models.base import BaseModel
from dinora.search.base import BaseSearcher
from dinora.search.ext_mcts.params import MCTSparams
from dinora.search.ext_mcts.search import run_mcts
from dinora.search.stoppers import Stopper


class ExtMcts(BaseSearcher):
    def __init__(self) -> None:
        self._params = MCTSparams()

    @property
    def params(self):
        return self._params

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        move = run_mcts(board, stopper, evaluator, self.params).best_mixed().move
        assert move
        return move
