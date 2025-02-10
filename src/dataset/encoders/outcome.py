import chess.engine
import chess.pgn

WIN, DRAW, LOSS = range(3)


class UnexpectedOutcome(Exception):
    pass


def wdl_index(game: chess.pgn.Game, flip: bool) -> int:
    result = game.headers["Result"]

    if result == "1/2-1/2":
        return DRAW
    elif result == "1-0":
        return WIN if not flip else LOSS
    elif result == "0-1":
        return LOSS if not flip else WIN
    else:
        raise UnexpectedOutcome(f"Illegal game result: {result}")


def z_value(game: chess.pgn.Game, flip: bool) -> float:
    result = game.headers["Result"]

    if result == "1/2-1/2":
        return 0.0
    elif result == "1-0":
        return +1.0 if not flip else -1.0
    elif result == "0-1":
        return -1.0 if not flip else +1.0
    else:
        raise UnexpectedOutcome(f"Illegal game result: {result}")


def stockfish_value(
    board: chess.Board, engine: chess.engine.SimpleEngine, nodes: int
) -> float:
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))

    assert "wdl" in info, "Stockfish doesn't reports 'wdl' values"

    # [0:1] scale
    wdl = info["wdl"].pov(board.turn).expectation()

    # [-1:1] scale
    wdl_scaled = wdl * 2.0 - 1.0
    return wdl_scaled
