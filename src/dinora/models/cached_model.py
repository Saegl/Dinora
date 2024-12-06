import chess
import pylru

from dinora.models.base import BaseModel, Priors, StateValue


class CachedModel(BaseModel):
    def __init__(self, model: BaseModel, size: int = 200000) -> None:
        self.cache = pylru.lrucache(size)
        self.model = model
        self.hit = 0
        self.misses = 0

    def name(self) -> str:
        return f"Cached [{self.model.name()}]"

    def clear(self) -> None:
        self.hit = 0
        self.misses = 0
        self.cache.clear()

    @property
    def hit_ratio(self) -> float:
        return self.hit / (self.hit + self.misses)

    def evaluate(self, board: chess.Board) -> tuple[Priors, StateValue]:
        epd = board.epd()
        if epd in self.cache:
            self.hit += 1
            policy, value = self.cache[epd]
            return policy, value
        else:
            self.misses += 1
            policy, value = self.model.evaluate(board)
            self.cache[epd] = [policy, value]
            return policy, value

    def reset(self) -> None:
        self.clear()
        self.model.reset()
