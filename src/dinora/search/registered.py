from dinora.search.base import BaseSearcher
from dinora.search.ext_mcts.searcher import ExtMcts
from dinora.search.mcts.mcts import MCTS
from dinora.search.mcts_batch.mcts_batch import MctsBatch
from dinora.search.onemove.onemove import OneMove

registered_searchers: dict[str, type[BaseSearcher]] = {
    "auto": ExtMcts,
    "ext_mcts": ExtMcts,
    "mcts": MCTS,
    "mcts_batch": MctsBatch,
    "onemove": OneMove,
}


def get_searcher(searcher_name: str) -> BaseSearcher:
    cls = registered_searchers[searcher_name]
    return cls()
