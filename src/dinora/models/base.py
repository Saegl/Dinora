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
        """
        Evaluate board by returning policy and value

        sum(policy.values()) == 1.0
        -1.0 <= value <= 1.0
        value = 1.0 => Current side (board.turn) is winning
        value = -1.0 => Current side (board.turn) is losing
        value = 0.0 => Draw
        """

    def evaluate_batch(self, boards: list[chess.Board]) -> list[Evaluation]:
        """
        Same as `evaluate` but for batch of boards

        Defaults to slow fallback, use faster methods of your
        neural networks library
        """
        return [self.evaluate(board) for board in boards]

    @abstractmethod
    def reset(self) -> None:
        """
        Delete caches
        Useful to call between games in benchmarks/evaluators
        """
