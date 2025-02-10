import chess
import numpy as np
import numpy.typing as npt

from dinora.encoders.board_tensor import MAX_HALFMOVES, PIECE_INDEX

npf32 = npt.NDArray[np.float32]
npuint64 = npt.NDArray[np.uint64]


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
