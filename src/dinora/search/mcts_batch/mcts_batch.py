import math
from dataclasses import dataclass, field
from typing import Optional

import chess

from dinora.models.base import BaseModel
from dinora.search.base import BaseSearcher
from dinora.search.noise import apply_noise
from dinora.search.stoppers import Stopper


@dataclass
class MctsParams:
    # exploration constant
    cpuct: float = field(default=3.0)
    # batch
    batch_size: int = field(default=16)
    virtual_visits: int = field(default=1)
    max_collisions: int = field(default=1)
    # random
    opening_noise_moves: int = field(default=15)
    dirichlet_alpha: float = field(default=0.3)
    noise_eps: float = field(default=0.0)  # set to 0.0 to disable random


class Node:
    parent: Optional["Node"]
    children: dict[chess.Move, "Node"]
    value_sum: float
    visits: int
    virtual_visits: int
    prior: float
    move: chess.Move

    def __init__(
        self,
        parent: Optional["Node"],
        value: float,
        prior: float,
        move: chess.Move,
    ):
        self.parent = parent
        self.children = {}
        self.value_sum = value  # value_sum to the perspective of parent
        self.visits = 1
        self.virtual_visits = 0
        self.prior = prior
        self.move = move

    def puct(self, cpuct: float):
        assert self.parent
        exploitation = self.value_sum / (self.visits + self.virtual_visits)
        exploration = (
            cpuct
            * math.sqrt(self.parent.visits)
            * self.prior
            / (self.visits + self.virtual_visits)
        )
        return exploitation + exploration

    def __repr__(self):
        return (
            f"Node <{self.move} prior={self.prior:.3f} "
            f" visits={self.visits} value={self.value_sum / self.visits:.3f} "
            f"virtual={self.virtual_visits} puct={self.puct(3.0):.3f}>"
        )


def select_best_puct(node: Node, cpuct: float) -> Node:
    best = None
    max_puct = None

    for child in node.children.values():
        puct = child.puct(cpuct)
        if max_puct is None or puct > max_puct:
            max_puct = puct
            best = child

    assert best is not None
    return best


def select(root: Node, board: chess.Board, cpuct: float, virtual_visits: int) -> Node:
    node = root
    while len(node.children) != 0:
        node = select_best_puct(node, cpuct)
        node.virtual_visits += virtual_visits
        board.push(node.move)
    return node


def expand(node: Node, child_priors):
    for move, prior in child_priors.items():
        node.children[move] = Node(node, 0.0, prior, move)


def backup(leaf: Node, value_leaf: float):
    node = leaf
    value = value_leaf
    while node.parent:
        value = -value  # perspective alternating on each move
        node.value_sum += value
        node.visits += 1
        node.virtual_visits = 0

        node = node.parent

    # at this point we don't have parent, means we root
    root = node
    value = -value
    root.value_sum += value
    root.visits += 1


def collect_batch(
    root: Node,
    board: chess.Board,
    cpuct: float,
    batch_size: int,
    virtual_visits: int,
    max_collisions: int,
) -> tuple[list[chess.Board], list[Node]]:
    batch_boards: list[chess.Board] = []
    batch_leaves: list[Node] = []
    collisions = 0
    for _ in range(batch_size):
        leaf_board = board.copy()
        leaf = select(root, leaf_board, cpuct, virtual_visits)
        if leaf.virtual_visits > virtual_visits:
            collisions += 1

            if collisions >= max_collisions:
                break
            continue

        terminal_value = terminal_solver(leaf_board)
        if terminal_value is not None:
            value = terminal_value
            backup(leaf, value)
            continue

        batch_boards.append(leaf_board)
        batch_leaves.append(leaf)

    return batch_boards, batch_leaves


def process_batch(
    batch_boards: list[chess.Board], batch_leaves: list[Node], evaluator: BaseModel
):
    if len(batch_boards) > 0:
        for leaf, (priors, value) in zip(
            batch_leaves, evaluator.evaluate_batch(batch_boards)
        ):
            expand(leaf, priors)
            backup(leaf, value)


def most_visited_move(node: Node) -> chess.Move:
    max_visits = 0
    current_move = chess.Move.null()
    for move, child in node.children.items():
        if child.visits > max_visits:
            max_visits = child.visits
            current_move = move
    assert current_move != chess.Move.null(), "Can't play null move"
    return current_move


def terminal_solver(board: chess.Board) -> float | None:
    board_result = board.result(claim_draw=True)
    if board_result != "*":  # This node is terminal
        # No matter who won, we lost because it is our turn to move
        if board_result == "1-0" or board_result == "0-1":
            return -1.0
        elif board_result == "1/2-1/2":
            return 0.0
    return None


class MctsBatch(BaseSearcher):
    def __init__(self):
        self._params = MctsParams()

    @property
    def params(self):
        return self._params

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        priors, value = evaluator.evaluate(board)
        root = Node(None, -value, 1.0, chess.Move.null())

        if board.ply() < 2 * self.params.opening_noise_moves:
            priors = apply_noise(
                priors,
                dirichlet_alpha=self.params.dirichlet_alpha,
                noise_eps=self.params.noise_eps,
            )

        expand(root, priors)

        while not stopper.should_stop():
            batch_boards, batch_leaves = collect_batch(
                root,
                board,
                self.params.cpuct,
                self.params.batch_size,
                self.params.virtual_visits,
                self.params.max_collisions,
            )

            process_batch(batch_boards, batch_leaves, evaluator)

        print(f"info nodes {root.visits}")
        return most_visited_move(root)
