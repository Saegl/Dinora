import chess

from dinora.models import model_selector
from dinora.search.mcts_batch.mcts_batch import (
    Node,
    collect_batch,
    expand,
    process_batch,
)

evaluator = model_selector("handcrafted", None, None)


def check_no_virtual_visits(node: Node):
    assert node.virtual_visits == 0
    for child in node.children.values():
        check_no_virtual_visits(child)


def check_no_duplicate_leaves(leaves: list[Node]):
    ids = [id(leaf) for leaf in leaves]
    assert len(ids) == len(set(ids))


def test_collect_batch():
    board = chess.Board()
    priors, value = evaluator.evaluate(board)
    root = Node(None, -value, 1.0, chess.Move.null())
    expand(root, priors)

    batch_boards, batch_leaves = collect_batch(root, board, 3.0, 16, 1, 1)

    assert len(batch_leaves) == len(batch_boards), "Sanity check"
    for batch_leaf in batch_leaves:
        if batch_leaf.virtual_visits != 1:
            raise Exception(
                f"Selected {batch_leaf} more than once {batch_leaf.virtual_visits}"
            )

    print(f"Batch size {len(batch_boards)}")
    print()

    check_no_duplicate_leaves(batch_leaves)


def test_process_batch():
    board = chess.Board()
    priors, value = evaluator.evaluate(board)
    root = Node(None, -value, 1.0, chess.Move.null())
    expand(root, priors)

    batch_boards, batch_leaves = collect_batch(root, board, 3.0, 16, 1, 1)
    process_batch(batch_boards, batch_leaves, evaluator)
    check_no_virtual_visits(root)
