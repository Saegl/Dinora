import abc
import enum
from typing import Any

import chess

from dinora.models.base import BaseModel
from dinora.search.stoppers import Stopper

DefaultValue = str


class ConfigType(enum.Enum):
    Float = enum.auto()
    String = enum.auto()
    Boolean = enum.auto()
    Integer = enum.auto()

    def convert(self, data: str) -> Any:
        match self:
            case ConfigType.Float:
                return float(data)
            case ConfigType.String:
                return data
            case ConfigType.Boolean:
                return bool(data)
            case ConfigType.Integer:
                return int(data)


class BaseSearcher(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        pass

    @abc.abstractmethod
    def set_config_param(self, k: str, v: str) -> None:
        pass

    @abc.abstractmethod
    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        pass
