import contextlib
import sys

import chess

from dinora.engine import Engine, ParamNotFound
from dinora.search.base import ConfigType
from dinora.uci.uci_go_parser import parse_go_params


def send(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


class UciState:
    def __init__(self, engine: Engine) -> None:
        self.board = chess.Board()
        self.engine = engine

    def load_model(self) -> None:
        if not self.engine.loaded():
            send("info string loading nn, it make take a while")
            self.engine.load_model()
            send("info string nn is loaded")
            send(f"info string model name: {self.engine.model.name()}")

    def dispatcher(self, line: str) -> None:
        command, *tokens = line.strip().split()
        supported_commands = [
            "uci",
            "ucinewgame",
            "setoption",
            "isready",
            "position",
            "go",
            "quit",
        ]
        if command in supported_commands:
            command_method = getattr(self, command)
            command_method(tokens)
        else:
            send(f"info string command is not processed: {line}")

    def loop(self) -> None:
        send("Dinora chess engine")
        while True:
            line = input()
            self.dispatcher(line)

    ### UCI Commands:
    def uci(self, _: list[str]) -> None:
        send("id name Dinora")
        send("id author Saegl")
        for name, (config_type, default) in self.engine.get_config_schema().items():
            match config_type:
                case ConfigType.String:
                    uci_type_name = "string"
                case ConfigType.Float:
                    uci_type_name = "string"
                case ConfigType.Integer:
                    uci_type_name = "spin"
                case _:
                    raise Exception("Cannot convert config_type to uci_type")
            send(f"option name {name} type {uci_type_name} default {default}")
        send("uciok")

    def ucinewgame(self, _: list[str]) -> None:
        self.load_model()
        self.board = chess.Board()

    def setoption(self, tokens: list[str]) -> None:
        name = tokens[tokens.index("name") + 1].lower()
        value = tokens[tokens.index("value") + 1]
        with contextlib.suppress(ParamNotFound):
            self.engine.set_config_param(name, value)

    def isready(self, _: list[str]) -> None:
        self.load_model()
        send("readyok")

    def position(self, tokens: list[str]) -> None:
        if tokens[0] == "startpos":
            self.board = chess.Board()
            if "moves" in tokens:
                moves = tokens[tokens.index("moves") + 1 :]
                for move_token in moves:
                    self.board.push_uci(move_token)

        if tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)

    def go(self, tokens: list[str]) -> None:
        self.load_model()

        go_params = parse_go_params(tokens)
        send(f"info string parsed params {go_params}")

        stopper = go_params.get_search_stopper(self.board)
        send(f"info string chosen stopper {stopper}")

        move = self.engine.get_best_move(self.board, stopper)
        send(f"bestmove {move}")

    def quit(self, _: list[str]) -> None:
        sys.exit(0)
