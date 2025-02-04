"""
Board tensor in NCHW format:
    N is the number of boards = 1
    C [channels] is the number of planes
        (12 for pieces, 4 for castling, 1 for fifty move, 1 for en passant)
        = 12 + 4 + 1 + 1 = 18
    H [height] of chess board = 8
    W [width] of chess board = 8

Array weight is 18 * 8 * 8 * 4 = 4608 bytes, where 4 is float32
"""

import typing

import chess
import numpy as np
import numpy.typing as npt

npf32 = npt.NDArray[np.float32]
npuint64 = npt.NDArray[np.uint64]

# Divide with this constant to normalize (set range to [0:1])
MAX_HALFMOVES = 100  # For 50 Move Rule

PLANE_NAMES = [
    "WHITE KING",
    "WHITE QUEEN",
    "WHITE ROOK",
    "WHITE BISHOP",
    "WHITE KNIGHT",
    "WHITE PAWN",
    "BLACK KING",
    "BLACK QUEEN",
    "BLACK ROOK",
    "BLACK BISHOP",
    "BLACK KNIGHT",
    "BLACK PAWN",
    "WHITE SHORT CASTLE",
    "WHITE LONG CASTLE",
    "BLACK SHORT CASTLE",
    "BLACK LONG CASTLE",
    "FIFTY MOVES COUNTER",
    "CAN EN PASSANT",
]

assert len(PLANE_NAMES) == 18

PIECE_INDEX = {
    (chess.KING, chess.WHITE): 0,
    (chess.QUEEN, chess.WHITE): 1,
    (chess.ROOK, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3,
    (chess.KNIGHT, chess.WHITE): 4,
    (chess.PAWN, chess.WHITE): 5,
    (chess.KING, chess.BLACK): 6,
    (chess.QUEEN, chess.BLACK): 7,
    (chess.ROOK, chess.BLACK): 8,
    (chess.BISHOP, chess.BLACK): 9,
    (chess.KNIGHT, chess.BLACK): 10,
    (chess.PAWN, chess.BLACK): 11,
}

assert len(PIECE_INDEX) == 12


def board_to_tensor(board: chess.Board, flip: bool) -> npf32:
    "Convert current state (chessboard) to tensor"
    tensor = np.zeros((18, 8, 8), np.float32)

    # Set pieces [0: 12)
    for square in chess.scan_reversed(board.occupied):
        file, rank = divmod(square, 8)
        piece = typing.cast(chess.Piece, board.piece_at(square))
        index = PIECE_INDEX[piece.piece_type, piece.color ^ flip]
        if flip:
            file = 7 - file
        tensor[index, file, rank] = 1.0

    # Set castling rights [12: 16)
    if board.castling_rights & (chess.BB_H8 if flip else chess.BB_H1):
        tensor[12].fill(1.0)
    if board.castling_rights & (chess.BB_A8 if flip else chess.BB_A1):
        tensor[13].fill(1.0)
    if board.castling_rights & (chess.BB_H1 if flip else chess.BB_H8):
        tensor[14].fill(1.0)
    if board.castling_rights & (chess.BB_A1 if flip else chess.BB_A8):
        tensor[15].fill(1.0)

    # Set fifty move counter [16: 17)
    tensor[16].fill(board.halfmove_clock / MAX_HALFMOVES)

    # Set en passant square [17: 18)
    if board.has_legal_en_passant():
        assert board.ep_square
        file, rank = divmod(board.ep_square, 8)
        if flip:
            file = 7 - file
        tensor[17, file, rank] = 1.0

    return tensor


def board_to_compact_state(board: chess.Board, flip: bool) -> npuint64:
    if flip:
        board = board.mirror()

    pieces_planes = np.zeros((12, 8, 8), np.uint8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            index = PIECE_INDEX[piece.piece_type, piece.color]
            pieces_planes[index][square // 8][square % 8] = 1

    pieces_array = np.packbits(
        pieces_planes.reshape(-1),
        axis=0,
    ).view(np.uint64)

    castling = board.castling_rights
    en_passant = board.ep_square if board.has_legal_en_passant() else 64
    halfmove = board.halfmove_clock
    config = (castling, en_passant, halfmove)

    config_array = np.array(config, dtype=np.uint64)

    return np.concatenate((pieces_array, config_array))


def compact_state_to_board_tensor(array: npuint64) -> npf32:
    pieces_array = np.unpackbits(array[:12].view(np.uint8))
    pieces_planes = pieces_array.reshape((12, 8, 8)).astype(np.float32)

    castling = int(array[-3])
    en_passant = int(array[-2])
    halfmove = array[-1]

    configs = np.zeros((6, 8, 8), dtype=np.float32)

    # Set castling rights [12: 16)
    if castling & chess.BB_H1:
        configs[0].fill(1.0)
    if castling & chess.BB_A1:
        configs[1].fill(1.0)
    if castling & chess.BB_H8:
        configs[2].fill(1.0)
    if castling & chess.BB_A8:
        configs[3].fill(1.0)

    # Set fifty move counter [16: 17)
    configs[4].fill(halfmove / MAX_HALFMOVES)

    # Set en passant square [17: 18)
    # print(en_passant)
    if en_passant != 64:
        square: chess.Square = en_passant
        configs[5][square // 8][square % 8] = 1.0

    return np.concatenate((pieces_planes, configs))
