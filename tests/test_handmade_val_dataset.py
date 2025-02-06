import chess

from train.handmade_val_dataset.dataset import POSITIONS


def test_sanity():
    for pos in POSITIONS:
        board = chess.Board(fen=pos["fen"])
        if pos["type"] in ["WHITE IS LOSING", "WHITE WORSE", "WHITE SLIGHTLY WORSE"]:
            assert board.turn
            assert pos["stockfish_cp"] < 0.0
        elif pos["type"] in [
            "WHITE SLIGHTLY BETTER",
            "WHITE BETTER",
            "WHITE IS WINNING",
        ]:
            assert board.turn
            assert pos["stockfish_cp"] > 0.0
        elif pos["type"] in ["BLACK IS LOSING", "BLACK WORSE", "BLACK SLIGHTLY WORSE"]:
            assert not board.turn
            assert pos["stockfish_cp"] > 0.0
        elif pos["type"] in [
            "BLACK SLIGHTLY BETTER",
            "BLACK BETTER",
            "BLACK IS WINNING",
        ]:
            assert not board.turn
            assert pos["stockfish_cp"] < 0.0
        elif pos["type"] in [
            "WHITE EQUAL",
            "WHITE MATE IN N",
            "WHITE WIN PIECE TACTIC",
            "WHITE OPENING",
            "WHITE MIDDLEGAME",
            "WHITE ENDGAME",
        ]:
            assert board.turn
        elif pos["type"] in [
            "BLACK EQUAL",
            "BLACK MATE IN N",
            "BLACK WIN PIECE TACTIC",
            "BLACK OPENING",
            "BLACK MIDDLEGAME",
            "BLACK ENDGAME",
        ]:
            assert not board.turn
        else:
            raise ValueError(f"Unknown position type {pos['type']}")
