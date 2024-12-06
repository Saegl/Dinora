from abc import ABC, abstractmethod

import chess

Priors = dict[chess.Move, float]
StateValue = float
Evaluation = tuple[Priors, StateValue]


class BaseModel(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return model name for logs/debug_info"""

    @abstractmethod
    def evaluate(self, board: chess.Board) -> Evaluation:
        pass

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        return [self.evaluate(board) for board in boards]

    @abstractmethod
    def reset(self) -> None:
        """Delete caches"""
