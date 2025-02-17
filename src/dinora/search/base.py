import abc
from dataclasses import fields
from typing import Generic, TypeVar

import chess

from dinora.models.base import BaseModel
from dinora.search.stoppers import Stopper

ParamsType = TypeVar("ParamsType")


class BaseSearcher(abc.ABC, Generic[ParamsType]):
    params: ParamsType

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    def set_config_param(self, k: str, v: str) -> None:
        # linear search is fast enough for 3 params
        for field in fields(self.params):  # type: ignore
            assert callable(field.type)
            if field.name == k:
                setattr(self.params, field.name, field.type(v))

    def update_from_dict(self, d: dict) -> None:
        for field in fields(self.params):  # type: ignore
            assert callable(field.type)
            if field.name in d:
                setattr(self.params, field.name, field.type(d[field.name]))

    @abc.abstractmethod
    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        pass
