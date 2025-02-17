import threading
from math import cos
from time import time

extra_time = 0.5


class Stopper:
    """
    Stoppers control when to stop searching
    """

    def __init__(self):
        self.early_stop = threading.Event()
        self.called = False

    def should_stop(self) -> bool:
        if not self.called:  # Calculate minimum 1 node
            self.called = True
            return False
        return self.early_stop.is_set()


def time_manager(moves_number: int, time_left: int, inc: int = 0) -> float:
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining_time = time_left / 1000 + moves_left * inc / 1000
    move_time = remaining_time / moves_left - extra_time
    return move_time


class Time(Stopper):
    def __init__(
        self,
        moves_number: int,
        engine_time: int,
        engine_inc: int,
    ) -> None:
        super().__init__()
        self.move_time = time_manager(moves_number, engine_time, engine_inc)
        self.starttime = time()
        self.steps = 0

    def should_stop(self) -> bool:
        if not self.called:
            self.called = True
            return False
        if super().should_stop():
            return True
        return time() - self.starttime > self.move_time

    def __str__(self) -> str:
        return f"<Time: {self.move_time=} {self.starttime=}>"


class MoveTime(Stopper):
    movetime: int
    starttime: float

    def __init__(self, movetime: int) -> None:
        super().__init__()
        self.movetime = movetime
        self.starttime = time()

    def should_stop(self) -> bool:
        if not self.called:
            self.called = True
            return False
        if super().should_stop():
            return True
        return time() - self.starttime > (self.movetime / 1000)

    def __str__(self) -> str:
        return f"<MoveTime: {self.movetime=} {self.starttime=}>"


class NodesCount(Stopper):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.step = 0
        self.count = count

    def should_stop(self) -> bool:
        if super().should_stop():
            return True
        self.step += 1
        return self.count < self.step

    def __str__(self) -> str:
        return f"<NodesCount: {self.count=} {self.step=}>"


class Infinite(Stopper):
    def __init__(self):
        super().__init__()

    def should_stop(self) -> bool:
        return super().should_stop()

    def __str__(self) -> str:
        return "<Infinite 8>"
