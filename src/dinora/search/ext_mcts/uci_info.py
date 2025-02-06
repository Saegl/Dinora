from collections.abc import Callable
from time import time

from dinora.search.ext_mcts.node import Node

NONZERO = 0.001


def cp(q: float) -> int:
    """Convert UCT Q values to Stockfish like centipawns"""
    return int(295 * q / (1 - 0.976953125 * q**14))


def calc_score(node: Node) -> int:
    score = int(round(cp(node.q()), 0))
    return score


def send_info(
    send: Callable[[str], None],
    count: int,
    delta: float,
    score: int,
    pv: str,
) -> None:
    if send is not None:
        send(
            f"info score cp {score} nodes {count} nps {int(round(count / (delta + NONZERO), 0))} pv {pv}"
        )


def send_tree_info(send: Callable[[str], None], root: Node) -> None:
    if send is not None:
        for nd in sorted(root.children.items(), key=lambda item: item[1].number_visits):
            send(
                f"info string {nd[1].move} {nd[1].number_visits} \t(P: {round(nd[1].prior * 100, 2)}%) \t(Q: {round(nd[1].q(), 5)})"
            )


class UciInfo:
    def __init__(self) -> None:
        self.start_time = time()
        self.count = 0
        self.delta_last = 0.0

    def after_iteration(self, root: Node, send_func: Callable[[str], None]) -> None:
        self.count += 1
        now = time()
        delta = now - self.start_time

        if delta - self.delta_last > 5:  # Send info every 5 sec
            self.delta_last = delta
            bestnode = root.best_n()
            score = calc_score(bestnode)
            send_info(
                send_func,
                self.count,
                delta,
                score,
                root.get_pv_line(),
            )

    def at_mcts_end(self, root: Node, send_func: Callable[[str], None]) -> None:
        bestnode = root.best_mixed()
        delta = time() - self.start_time
        score = calc_score(bestnode)

        assert bestnode.move
        send_info(send_func, self.count, delta, score, root.get_pv_line())
        send_tree_info(send_func, root)

    def reuse_stats(self, visits: int, send_func: Callable[[str], None]) -> None:
        send_func(f"info string reuse {visits} nodes")
