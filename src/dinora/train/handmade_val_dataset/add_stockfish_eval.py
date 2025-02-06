import pprint

import chess
import chess.engine

from dinora.train.handmade_val_dataset.dataset import POSITIONS

engine = chess.engine.SimpleEngine.popen_uci("stockfish")


for pos in POSITIONS:
    board = chess.Board(fen=pos["fen"])
    text = pos["text"]

    info = engine.analyse(board, limit=chess.engine.Limit(nodes=100_000), multipv=3)
    # pprint.pprint(info)
    score = info[0]["score"]
    pos["stockfish_cp"] = score.white().score(mate_score=10_000) / 100
    wdl = score.wdl(ply=board.ply()).pov(board.turn)
    win = wdl.winning_chance()
    draw = wdl.drawing_chance()
    loss = wdl.losing_chance()
    pos["stockfish_wdl"] = f"{win:.2f} | {draw:.2f} | {loss:.2f}"
    pos["stockfish_top3_lines"] = "\n".join(
        [f"{[e.uci() for e in line['pv'][:3]]}" for line in info[:3]]
    )

engine.quit()
pprint.pprint(POSITIONS)
