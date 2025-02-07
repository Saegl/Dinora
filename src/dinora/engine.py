import pathlib
from typing import Any

import chess

from dinora.models import BaseModel, model_selector
from dinora.search.registered import get_searcher
from dinora.search.stoppers import Stopper


class ParamNotFound(Exception):
    """Thrown if config param not exists"""


class Engine:
    def __init__(
        self,
        searcher: str = "auto",
        model_name: str | None = None,
        weights_path: pathlib.Path | None = None,
        device: str | None = None,
    ):
        self.searcher = get_searcher(searcher)
        self._model_name = model_name
        self._model: BaseModel | None = None
        self.weights_path = weights_path
        self.device = device

    @property
    def model(self) -> BaseModel:
        if self._model:
            return self._model
        else:
            raise Exception("Model is not loaded")

    def loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is None:
            self._model = model_selector(
                self._model_name, self.weights_path, self.device
            )

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def set_config_param(self, name: str, value: Any) -> None:
        try:
            self.searcher.set_config_param(name, value)
        except KeyError as exc:
            raise ParamNotFound from exc

    def get_best_move(self, board: chess.Board, stopper: Stopper) -> chess.Move:
        return self.searcher.search(board, stopper, self.model)
