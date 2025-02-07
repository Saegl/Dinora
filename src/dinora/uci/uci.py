import contextlib
import queue
import sys
import threading
from dataclasses import dataclass, fields
from typing import Any

import chess

from dinora.engine import Engine, ParamNotFound
from dinora.search.stoppers import Stopper
from dinora.uci.uci_go_parser import parse_go_params


def send(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()


class UciCommand:
    """Command sent by UciLoop"""


@dataclass
class Go(UciCommand):
    board: chess.Board
    stopper: Stopper


@dataclass
class SetOption(UciCommand):
    name: str
    value: str


@dataclass
class IsReady(UciCommand):
    pass


@dataclass
class Quit(UciCommand):
    pass


def uci_start(engine: Engine):
    commands_queue: queue.Queue[UciCommand] = queue.Queue()
    uciloop = UciCommunicator(
        commands_queue,
        engine.searcher.params,
    )
    uciengine = UciEngine(commands_queue, engine)

    controller_thread = threading.Thread(target=uciengine.loop)
    controller_thread.start()

    uciloop.start_communication()


@dataclass
class UciEngine:
    commands_queue: queue.Queue[UciCommand]
    engine: Engine

    def loop(self):
        running = True
        while running:
            command = self.commands_queue.get()
            match command:
                case IsReady():
                    self.engine.load_model()

                case Go(board, stopper):
                    move = self.engine.get_best_move(board, stopper)
                    send(f"bestmove {move}")
                    stopper.early_stop.set()
                case SetOption(name, value):
                    with contextlib.suppress(ParamNotFound):
                        self.engine.set_config_param(name, value)
                case Quit():
                    running = False

            self.commands_queue.task_done()
            if not running:
                break


class UciCommunicator:
    commands_queue: queue.Queue[UciCommand]
    params: Any
    active_stopper: Stopper | None
    board: chess.Board
    running: bool

    def __init__(self, commands_queue: queue.Queue[UciCommand], params):
        self.commands_queue = commands_queue
        self.params = params
        self.active_stopper = None
        self.board = chess.Board()
        self.running = True

    def start_communication(self):
        send("Dinora chess engine")
        while self.running:
            try:
                line = input()
            except KeyboardInterrupt:
                self.quit([])
                break

            self.dispatcher(line)

    def dispatcher(self, line: str):
        command, *tokens = line.strip().split()
        supported_commands = [
            "uci",
            "ucinewgame",
            "setoption",
            "isready",
            "position",
            "go",
            "stop",
            "quit",
        ]
        if command in supported_commands:
            command_method = getattr(self, command)
            command_method(tokens)
        else:
            send(f"info string command is not processed: {line}")

    def uci(self, _: list[str]) -> None:
        send("id name Dinora")
        send("id author Saegl")
        for field in fields(self.params):
            if field.type is int:
                uci_type_name = "spin"
            elif field.type is float or field.type is str:
                uci_type_name = "string"
            else:
                continue

            send(
                f"option name {field.name} type {uci_type_name} default {field.default}"
            )
        send("uciok")

    def ucinewgame(self, _: list[str]) -> None:
        self.board = chess.Board()

    def setoption(self, tokens: list[str]) -> None:
        name = tokens[tokens.index("name") + 1].lower()
        value = tokens[tokens.index("value") + 1]
        self.commands_queue.put(SetOption(name, value))

    def isready(self, _: list[str]) -> None:
        self.commands_queue.put(IsReady())
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
        if self.active_stopper and not self.active_stopper.early_stop.is_set():
            raise Exception("Can't run second `go`")

        self.commands_queue.put(IsReady())

        go_params = parse_go_params(tokens)
        send(f"info string parsed params {go_params}")

        self.active_stopper = go_params.get_search_stopper(self.board)
        send(f"info string chosen stopper {self.active_stopper}")

        self.commands_queue.put(Go(self.board.copy(), self.active_stopper))

    def stop(self, tokens: list[str]) -> None:
        if self.active_stopper and not self.active_stopper.early_stop.is_set():
            self.active_stopper.early_stop.set()

    def quit(self, _: list[str]) -> None:
        self.stop([])
        self.commands_queue.put(Quit())
        self.running = False
