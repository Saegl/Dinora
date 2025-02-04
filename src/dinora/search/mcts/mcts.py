from __future__ import annotations

import math

import chess

from dinora.models import BaseModel
from dinora.models.base import Priors
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.mcts import params
from dinora.search.noise import apply_noise
from dinora.search.stoppers import Stopper


class Node:
    parent: Node | None
    visits: int
    prior: float
    children: dict[chess.Move, Node]

    def __init__(self, parent: Node | None, prior: float) -> None:
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    def is_expanded(self) -> bool:
        return len(self.children) != 0

    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def expand(node: Node, priors: Priors) -> None:
    for move, prior in priors.items():
        node.children[move] = Node(node, prior)


def backpropagate(node: Node, value: float, board: chess.Board) -> None:
    current = node
    turn_factor = -1
    while current.parent:
        current.visits += 1
        current.value_sum += turn_factor * value
        current = current.parent
        turn_factor *= -1
        board.pop()

    current.visits += 1


def select_leaf(root: Node, board: chess.Board) -> Node:
    current = root

    while current.is_expanded():
        move, current = select_child(current)
        board.push(move)

    return current


def select_child(node: Node) -> tuple[chess.Move, Node]:
    return max(node.children.items(), key=lambda el: ucb_score(node, el[1]))


def ucb_score(parent: Node, child: Node) -> float:
    exploration = 1.25 * child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    return child.value() + exploration


def most_visits_move(root: Node) -> chess.Move:
    bestmove, _ = max(root.children.items(), key=lambda el: el[1].visits)
    return bestmove


class MCTS(BaseSearcher):
    def __init__(self) -> None:
        self.params = params.MCTSparams()

    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        return {
            "first_n_moves": (ConfigType.Integer, "15"),
            "dirichlet_alpha": (ConfigType.Float, "0.3"),
            "noise_eps": (ConfigType.Float, "0.0"),
        }

    def set_config_param(self, k: str, v: str) -> None:
        schema = self.config_schema()
        config_type, _ = schema[k]
        native_value = config_type.convert(v)
        setattr(self.params, k, native_value)

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        root = Node(None, 0.0)
        priors, value = evaluator.evaluate(board)

        if board.ply() // 2 < self.params.first_n_moves:
            priors = apply_noise(
                priors,
                dirichlet_alpha=self.params.dirichlet_alpha,
                noise_eps=self.params.noise_eps,
            )

        expand(root, priors)
        while not stopper.should_stop():
            leaf = select_leaf(root, board)
            priors, value = evaluator.evaluate(board)
            expand(leaf, priors)
            backpropagate(leaf, value, board)

        print(f"info nodes {root.visits}")
        return most_visits_move(root)
