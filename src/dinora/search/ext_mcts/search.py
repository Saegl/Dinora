from collections.abc import Callable

import chess

from dinora.models import BaseModel, Priors
from dinora.search.ext_mcts.node import Node
from dinora.search.ext_mcts.params import MCTSparams
from dinora.search.ext_mcts.reduction import reduction, terminal_val
from dinora.search.ext_mcts.uci_info import UciInfo
from dinora.search.noise import apply_noise
from dinora.search.stoppers import Stopper

SelectionPolicy = Callable[[Node], Node]


def selection(root: Node, selection_policy: SelectionPolicy) -> Node:
    current = root

    while current.is_expanded and current.children:
        current = selection_policy(current)

    return current


def expansion(node: Node, child_priors: Priors, fpu: float) -> None:
    node.is_expanded = True

    for move, prior in child_priors.items():
        node.add_child(move, prior, fpu)


def backpropagation(node: Node, value_estimate: float) -> None:
    current = node
    # Child nodes are multiplied by -1 because we want max(-opponent eval)
    turnfactor = -1.0
    while current.parent is not None:
        current.number_visits += 1
        current.total_value += value_estimate * turnfactor
        current = current.parent
        turnfactor *= -1
    current.number_visits += 1


def run_mcts(
    state: chess.Board | Node,
    stopper: Stopper,
    evaluator: BaseModel,
    params: MCTSparams,
) -> Node:
    uci_info = UciInfo()

    # Can reuse tree
    if isinstance(state, Node):
        root = state
        uci_info.reuse_stats(root.number_visits, params.send_func)
        print(root)
    else:
        root = Node(params.fpu_at_root, lazyboard=state)
        child_priors, root.value_estimate = evaluator.evaluate(root.board)
        child_priors = apply_noise(
            child_priors,
            params.dirichlet_alpha,
            params.noise_eps,
        )
        expansion(root, child_priors, params.fpu)
        backpropagation(root, root.value_estimate)

    while not stopper.should_stop():
        leaf = selection(root, params.selection_policy)

        tval = terminal_val(leaf.board)

        if params.node_reduction and tval is not None:
            leaf.value_estimate = tval
            leaf.to_terminal()
            leaf.til_end = 0
            leaf = reduction(leaf)
        else:
            child_priors, leaf.value_estimate = evaluator.evaluate(leaf.board)
            leaf.value_estimate = leaf.value_estimate if tval is None else tval
            expansion(leaf, child_priors, params.fpu)

        backpropagation(leaf, leaf.value_estimate)

        uci_info.after_iteration(root, params.send_func)
        if root.is_terminal:
            break

    uci_info.at_mcts_end(root, params.send_func)
    return root
