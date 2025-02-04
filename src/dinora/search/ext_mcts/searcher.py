import chess

from dinora.models.base import BaseModel
from dinora.search.base import BaseSearcher, ConfigType, DefaultValue
from dinora.search.ext_mcts.params import MCTSparams
from dinora.search.ext_mcts.search import run_mcts
from dinora.search.stoppers import Stopper


class ExtMcts(BaseSearcher):
    def __init__(self) -> None:
        self.params = MCTSparams()

    def config_schema(self) -> dict[str, tuple[ConfigType, DefaultValue]]:
        return {
            "fpu": (ConfigType.Float, "-1.0"),
            "fpu_at_root": (ConfigType.Float, "0.0"),
            "selection_policy_name": (ConfigType.String, "puct"),
            "cpuct": (ConfigType.Float, "3.0"),
            "t": (ConfigType.Float, "1.0"),
            "dirichlet_alpha": (ConfigType.Float, "0.3"),
            "noise_eps": (ConfigType.Float, "0.0"),
        }

    def set_config_param(self, k: str, v: str) -> None:
        schema = self.config_schema()
        config_type, _ = schema[k]
        native_value = config_type.convert(v)
        setattr(self.params, k, native_value)

    def search(
        self, board: chess.Board, stopper: Stopper, evaluator: BaseModel
    ) -> chess.Move:
        move = run_mcts(board, stopper, evaluator, self.params).best_mixed().move
        assert move
        return move
