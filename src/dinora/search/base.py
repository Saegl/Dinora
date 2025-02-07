import abc
from dataclasses import fields
from typing import Any

import chess

from dinora.models.base import BaseModel
from dinora.search.stoppers import Stopper


class BaseSearcher(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def params(self) -> Any:
        pass

    def set_config_param(self, k: str, v: str) -> None:
        for field in fields(self.params):  # linear search is fast enough for 3 params
            assert callable(field.type)
            if field.name == k:
                setattr(self.params, field.name, field.type(v))

    @abc.abstractmethod
    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        pass
