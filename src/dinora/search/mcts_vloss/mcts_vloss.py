from __future__ import annotations

import math

import chess

from dinora.models import BaseModel
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.mcts_vloss.params import MCTSparams
from dinora.search.stoppers import Stopper


class Node:
    parent: Node | None
    visits: int
    prior: float
    children: dict[chess.Move, Node]
    in_batch: bool

    def __init__(self, parent: Node | None, prior: float) -> None:
        self.parent = parent
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}
        self.in_batch = False

    def is_expanded(self) -> bool:
        return len(self.children) != 0

    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def expand(node: Node, priors):
    for move, prior in priors.items():
        node.children[move] = Node(node, prior)


def backpropagate(node: Node, value: float, virtual_loss: int) -> None:
    current = node
    turn_factor = -1
    while current.parent:
        current.visits -= virtual_loss - 1
        current.value_sum += turn_factor * value
        current = current.parent
        turn_factor *= -1
    current.visits -= virtual_loss - 1


def undo_virtual_loss(node: Node, virtual_loss: int) -> None:
    current = node
    current.visits -= virtual_loss
    while current.parent:
        current = current.parent
        current.visits -= virtual_loss


def select_leaf(root: Node, board: chess.Board, virtual_loss: int) -> Node:
    current = root
    current.visits += virtual_loss
    while current.is_expanded():
        move, current = select_child(current)
        current.visits += virtual_loss
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


def terminal_val(board: chess.Board) -> float | None:
    outcome = board.outcome()

    if outcome is not None and outcome.winner is not None:
        # There is a winner, but is is our turn to move
        # means we lost
        return -1.0

    elif (
        outcome is not None  # We know there is no winner by `if` above, so it's draw
        or board.is_repetition(3)  # Can claim draw means it's draw
        or board.can_claim_fifty_moves()
    ):
        # There is some subtle difference between
        # board.can_claim_threefold_repetition() and board.is_repetition(3)
        # I believe it is right to use second one, but I am not 100% sure
        # Gives +-1 ply error on this one for example https://lichess.org/EJ67iHS1/black#94

        return 0.0
    else:
        assert outcome is None
        return None


class MCTSVloss(BaseSearcher):
    def __init__(self) -> None:
        self.params = MCTSparams()

    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        return {
            "batch_size": (ConfigType.Integer, "256"),
            "virtual_loss": (ConfigType.Integer, "40"),
            "max_collisions": (ConfigType.Integer, "15"),
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
        root_board = board
        priors, value = evaluator.evaluate(root_board)
        expand(root, priors)
        while not stopper.should_stop():
            leafs = []
            terminal_leafs = []
            terminal_values: list[float] = []
            leaf_boards = []
            collisions = 0

            for _ in range(self.params.batch_size):
                leaf_board = root_board.copy()
                leaf = select_leaf(root, leaf_board, self.params.virtual_loss)
                term_val = terminal_val(leaf_board)

                if not leaf.in_batch:
                    leaf.in_batch = True
                    leafs.append(leaf)
                    leaf_boards.append(leaf_board)
                elif term_val is not None:
                    leaf.in_batch = True
                    terminal_leafs.append(leaf)
                    terminal_values.append(term_val)
                else:
                    collisions += 1
                    undo_virtual_loss(leaf, self.params.virtual_loss)
                    if collisions > self.params.max_collisions:
                        print(
                            f"info string max collisions break, batch size {len(leafs)}"
                        )
                        break

            for terminal_leaf, value in zip(terminal_leafs, terminal_values):
                terminal_leaf.in_batch = False
                backpropagate(terminal_leaf, value, self.params.virtual_loss)

            evals = evaluator.evaluate_batch(leaf_boards)
            for leaf, (priors, value) in zip(leafs, evals):
                leaf.in_batch = False
                expand(leaf, priors)
                backpropagate(leaf, value, self.params.virtual_loss)

        print(f"info nodes {root.visits}")
        return most_visits_move(root)
