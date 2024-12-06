from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import cos
from time import time

extra_time = 0.5


class Stopper(ABC):
    """
    Stoppers controll when to stop searching
    """

    @abstractmethod
    def should_stop(self) -> bool:
        pass


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
        self.move_time = time_manager(moves_number, engine_time, engine_inc)
        self.starttime = time()
        self.steps = 0

    def should_stop(self) -> bool:
        if self.steps < 2:  # Calculate minimum 2 nodes
            self.steps += 1
            return False
        return time() - self.starttime > self.move_time

    def __str__(self) -> str:
        return f"<Time: {self.move_time=} {self.starttime=}>"


@dataclass
class MoveTime(Stopper):
    movetime: int
    starttime: float = field(default_factory=time)
    movetime_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.movetime_seconds = self.movetime / 1000

    def should_stop(self) -> bool:
        return time() - self.starttime > (self.movetime / 1000)


class NodesCount(Stopper):
    def __init__(self, count: int) -> None:
        self.step = 0
        self.count = count

    def should_stop(self) -> bool:
        self.step += 1
        return self.count < self.step

    def __str__(self) -> str:
        return f"<NodesCount: {self.count=} {self.step=}>"


class Infinite(Stopper):
    def should_stop(self) -> bool:
        return False

    def __str__(self) -> str:
        return "<Infinite 8>"
