import chess
import numpy as np
import numpy.typing as npt

from dinora.encoders.board_tensor import MAX_HALFMOVES

npf32 = npt.NDArray[np.float32]
npuint64 = npt.NDArray[np.uint64]


def board_to_compact_state(board: chess.Board, flip: bool) -> npuint64:
    pieces_bb = np.array(
        [
            chess.flip_vertical(board.pawns) if flip else board.pawns,
            chess.flip_vertical(board.knights) if flip else board.knights,
            chess.flip_vertical(board.bishops) if flip else board.bishops,
            chess.flip_vertical(board.rooks) if flip else board.rooks,
            chess.flip_vertical(board.queens) if flip else board.queens,
            chess.flip_vertical(board.kings) if flip else board.kings,
            chess.flip_vertical(board.occupied_co[chess.BLACK])
            if flip
            else board.occupied_co[chess.WHITE],
            chess.flip_vertical(board.occupied_co[chess.WHITE])
            if flip
            else board.occupied_co[chess.BLACK],
        ],
        dtype=np.uint64,
    )

    if flip:
        castling = chess.flip_vertical(board.castling_rights)
    else:
        castling = board.castling_rights

    if board.has_legal_en_passant():
        assert board.ep_square
        file, rank = divmod(board.ep_square, 8)
        if flip:
            file = 7 - file

        en_passant = file * 8 + rank
    else:
        # En passant can't happen on square 64
        # Mark it as no en passant
        en_passant = 64

    halfmove = board.halfmove_clock
    config = (castling, en_passant, halfmove)

    config_array = np.array(config, dtype=np.uint64)

    return np.concatenate((pieces_bb, config_array))


def compact_state_to_board_tensor(array: npuint64) -> npf32:
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    pawns, knights, bishops, rooks, queens, kings = array[0:6]
    p1_occ = array[6]
    p2_occ = array[7]

    bitboards = np.array(
        [
            kings & p1_occ,
            queens & p1_occ,
            rooks & p1_occ,
            bishops & p1_occ,
            knights & p1_occ,
            pawns & p1_occ,
            kings & p2_occ,
            queens & p2_occ,
            rooks & p2_occ,
            bishops & p2_occ,
            knights & p2_occ,
            pawns & p2_occ,
        ],
        dtype=np.uint64,
    )

    tensor[0:12] = np.unpackbits(bitboards.view(np.uint8), bitorder="little").reshape(
        12, 8, 8
    )

    castling = int(array[-3])
    en_passant = int(array[-2])
    halfmove = array[-1]

    # Set castling rights [12: 16)
    if castling & chess.BB_H1:
        tensor[12].fill(1.0)
    if castling & chess.BB_A1:
        tensor[13].fill(1.0)
    if castling & chess.BB_H8:
        tensor[14].fill(1.0)
    if castling & chess.BB_A8:
        tensor[15].fill(1.0)

    # Set fifty move counter [16: 17)
    tensor[16].fill(halfmove / MAX_HALFMOVES)

    # Set en passant square [17: 18)
    if en_passant != 64:
        square: chess.Square = en_passant
        tensor[17][square // 8][square % 8] = 1.0

    return tensor
