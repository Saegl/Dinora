import math

import chess

from dinora.models.base import BaseModel, Priors, StateValue


def evaluate_board(board: chess.Board) -> float:
    our_mobility = board.legal_moves.count()
    board.push_uci("0000")
    opponent_mobility = board.legal_moves.count()
    board.pop()

    mobility = our_mobility - opponent_mobility
    our_material = material(board) if board.turn else -material(board)

    # 15 legal moves = 1 extra pawn
    pawns = our_material + mobility / 15.0

    # Need to scale evaluation to (-1: 1)
    # Let's say +5 extra pawns = win
    # tanh(3) ~ 1.0 = win
    return math.tanh(pawns * (3 / 5))


def material(board: chess.Board) -> float:
    white_material = 0.0
    black_material = 0.0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue  # Skip empty squares

        if piece.color:
            white_material += piece_value(piece)
        else:
            black_material += piece_value(piece)
    return white_material - black_material


def piece_value(piece: chess.Piece | None) -> float:
    """
    Values taken from here
    https://lichess.org/@/ubdip/blog/finding-the-value-of-pieces/PByOBlNB
    """
    if not piece:
        # Empty square
        return 0.0
    elif piece.piece_type == chess.PAWN:
        return 1.0
    elif piece.piece_type == chess.KNIGHT:
        return 3.16
    elif piece.piece_type == chess.BISHOP:
        return 3.28
    elif piece.piece_type == chess.ROOK:
        return 4.93
    elif piece.piece_type == chess.QUEEN:
        return 9.82
    elif piece.piece_type == chess.KING:
        return 90.0
    else:
        raise ValueError(f"Unsupported chess piece: {piece}")


def move_ordering(board: chess.Board) -> Priors:
    moves = []
    diffs = []
    for move in board.legal_moves:
        from_piece = board.piece_at(move.from_square)
        to_piece = board.piece_at(move.to_square)

        from_value = piece_value(from_piece)
        to_value = piece_value(to_piece)

        material_diff = to_value - from_value
        moves.append(move)
        diffs.append(material_diff)

    bottom = sum(math.exp(x) for x in diffs)
    softmax = (math.exp(x) / bottom for x in diffs)

    return dict(zip(moves, softmax))


class DummyModel(BaseModel):
    def name(self) -> str:
        return "Handcrafted"

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        return move_ordering(board), evaluate_board(board)

    def reset(self) -> None:
        pass
