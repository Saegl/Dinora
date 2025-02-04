import chess
import pytest

from dinora.models import model_selector


def test_batch() -> None:
    model = model_selector("alphanet", None, "cuda")
    boards = [
        chess.Board(),
        chess.Board(fen="1r3rk1/2p1qp1p/4p1pB/p6n/B1P5/1P3Q2/2P2PPP/4R1K1 w - - 1 21"),
        chess.Board(
            fen="rn1qkb1r/p1pbpppp/5n2/1p1P4/8/1B6/PPPP1PPP/RNBQK1NR b KQkq - 1 5"
        ),
    ]
    for board, (priors, value) in zip(boards, model.evaluate_batch(boards)):
        priors_single, value_single = model.evaluate(board)
        for move, prior in priors_single.items():
            pytest.approx(prior, priors[move])
        pytest.approx(value_single, value)
