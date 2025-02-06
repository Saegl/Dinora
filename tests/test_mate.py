"""
These are sneaky tests
Engine doesn't have to find mate
And even if it finds mate, it doesn't have to be shortest mate

These tests are useful to check `reduction` for nodes,
but if they fails, it could be false positive
"""

import chess
import pytest

from dinora.engine import Engine
from dinora.search.stoppers import NodesCount


@pytest.mark.skip(reason="Broken after introducing new `searcher` api")
def test_1ply_mates() -> None:
    fens = [
        "rn2k2r/pQ2ppbp/6p1/q4bB1/2P5/P7/1P3pPP/2KR1BNR w kq - 0 13",
        "8/6bk/1p4pp/2p5/2P5/P4Q1P/4nqP1/1R3B1K b - - 0 37",
        "1r3k1r/2p2p2/p2p3p/4bp1N/2B4P/Ppn5/2PQ1RP1/K4R2 b - - 0 28",
    ]
    engine = Engine("ext_mcts", "onnx", device="cpu")
    engine.mcts_params.node_reduction = True

    for fen in fens:
        board = chess.Board(fen=fen)
        root = engine.get_best_node(board, NodesCount(100)).parent
        assert root
        assert root.is_terminal
        assert len(root.get_pv_line().split()) == 1


@pytest.mark.skip(reason="Broken after introducing new `searcher` api")
def test_3plies_mates() -> None:
    fens = [
        "6k1/1p2qpPp/p5P1/3p3P/3rrB2/1P6/P1P3R1/1Kb2R2 w - - 0 30",
        "8/p7/2n2p2/2P5/2P1Nb1p/7P/k1K5/2Br4 w - - 0 44",
        "6k1/2p2p1p/6p1/8/P4Pn1/5QPK/1Pq5/R7 b - - 6 34",
    ]
    engine = Engine("ext_mcts", "onnx", device="cpu")
    engine.mcts_params.node_reduction = True

    for fen in fens:
        board = chess.Board(fen=fen)
        root = engine.get_best_node(board, NodesCount(500)).parent
        assert root
        assert root.is_terminal
        assert len(root.get_pv_line().split()) == 3


@pytest.mark.skip(reason="Broken after introducing new `searcher` api")
def test_7plies_mate() -> None:
    fen = "r1bqrnk1/pp4pp/2pb1p2/3pN3/3P1P2/2NBP1R1/PPQ3PP/R5K1 w - - 0 17"
    board = chess.Board(fen)
    engine = Engine("ext_mcts", "onnx", device="cpu")
    engine.mcts_params.node_reduction = True

    root = engine.get_best_node(board, NodesCount(1500)).parent
    assert root
    assert root.is_terminal
    assert root.get_pv_line() == "d3h7 f8h7 g3g7 g8g7 c2g6 g7h8 e5f7"
