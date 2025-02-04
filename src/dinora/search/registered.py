from dinora.search.base import BaseSearcher
from dinora.search.ext_mcts.searcher import ExtMcts
from dinora.search.mcts.mcts import MCTS
from dinora.search.mcts_vloss.mcts_vloss import MCTSVloss
from dinora.search.onemove.onemove import OneMove

registered_searchers: dict[str, type[BaseSearcher]] = {
    "auto": ExtMcts,
    "ext_mcts": ExtMcts,
    "mcts": MCTS,
    "mcts_vloss": MCTSVloss,
    "onemove": OneMove,
}


def get_searcher(searcher_name: str) -> BaseSearcher:
    cls = registered_searchers[searcher_name]
    return cls()
