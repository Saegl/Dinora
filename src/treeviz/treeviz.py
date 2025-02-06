"""
Visualization of MCTS
"""

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import cairosvg
import chess.svg
import graphviz

from dinora import PROJECT_ROOT
from dinora.models.base import BaseModel
from dinora.search.ext_mcts.node import Node
from dinora.search.ext_mcts.params import MCTSparams
from dinora.search.ext_mcts.search import run_mcts
from dinora.search.stoppers import NodesCount

NodeID = str
SVG_PREFIX = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/treeviz"


def node_id(node: object) -> NodeID:
    return str(id(node))


@dataclass
class RenderParams:
    render_nodes: int = 150  # Number of nodes to render
    show_other_node: bool = True
    show_prior: bool = True
    open_default_gui: bool = True
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR
    imgformat: Literal["png", "svg"] = "svg"


DEFAULT_RENDER_PARAMS = RenderParams()


def get_all_nodes(node: Node) -> Iterator[Node]:
    """Get all reachable nodes from a given node"""
    for _, child in node.children.items():
        yield child
        yield from get_all_nodes(child)


def select_most_visited_nodes(node: Node, n: int) -> set[NodeID]:
    """Select n most visited nodes"""
    all_nodes = list(get_all_nodes(node))
    all_nodes.sort(reverse=True, key=lambda e: e.number_visits)
    return set(map(node_id, all_nodes[:n]))


def tree_shape(root: Node) -> list[int]:
    """Calculate number of nodes on each depth by using dfs"""

    depthlist: list[int] = []

    def dfs(node: Node, depth: int) -> None:
        for _, child in node.children.items():
            if len(depthlist) == depth:
                depthlist.append(0)
            if not child.children:
                continue
            depthlist[depth] += 1
            dfs(child, depth + 1)

    dfs(root, 0)
    return depthlist


def get_pv_set(root: Node) -> set[NodeID]:
    ans = set()

    curr = root
    while len(curr.children) != 0:
        bestchild = curr.best_mixed()
        ans.add(node_id(bestchild))
        curr = bestchild
    return ans


def build_info_node(graph: graphviz.Digraph, root: Node) -> None:
    children = list(root.children.items())
    children.sort(key=lambda t: (t[1].number_visits, t[1].q()), reverse=True)
    top_visited_labels = [
        f"{t[0]} - N:{t[1].number_visits}, Q:{t[1].q():.3f}" for t in children[:5]
    ]
    top_visited = "\n".join(top_visited_labels)

    children.sort(key=lambda t: (t[1].q(), t[1].number_visits), reverse=True)
    top_q_highest_labels = [
        f"{t[0]} - Q:{t[1].q():.3f}, N{t[1].number_visits}" for t in children[:5]
    ]
    top_q = "\n".join(top_q_highest_labels)

    bestchild = root.best_mixed()
    info = (
        f"Side to move: {'white' if root.board.turn else 'black'} \n"
        f"Q at bestchild {bestchild.q():.3f} \n"
        f"VE at bestchild {bestchild.value_estimate:.3f} \n"
        f"VE at root {root.value_estimate:.3f} \n\n"
        f"TOP 5 most visited nodes at root: \n{top_visited} \n\n"
        f"TOP 5 highest Q nodes at root: \n{top_q} \n\n"
        f"Tree Shape: {tree_shape(root)}\n"
        f"PV line: {root.get_pv_line()}\n"
        f"Ply til end: {root.til_end if root.is_terminal else 'inf'}\n"
        f"Best move: {root.best_mixed().move}"
    )
    graph.node("info", label=info, shape="box")


def build_root_node(
    output_dir: pathlib.Path, graph: graphviz.Digraph, fen: str, imgformat: str
) -> None:
    """Save root node in digraph and attach board preview"""
    svg = SVG_PREFIX + chess.svg.board(chess.Board(fen))

    if imgformat == "svg":
        board_filename = "board.svg"
        board_path = output_dir / board_filename
        with open(board_path, "w", encoding="utf8") as f:
            f.write(svg)
    elif imgformat == "png":
        board_filename = "board.png"
        board_path = output_dir / board_filename
        cairosvg.svg2png(bytestring=svg, write_to=str(board_path))
    else:
        raise Exception("Unsupported image format")

    graph.node("root", label="", image=board_filename, imagescale="false", shape="box")


def build_children_nodes(
    graph: graphviz.Digraph,
    node: Node,
    parent_id: str,
    selected_nodes: set[NodeID],
    pv_set: set[NodeID],
    params: RenderParams,
) -> None:
    others_visits = 0
    others_id = node_id(node.children)
    others_prior = 0.0
    for move, child in chain(node.children.items(), node.terminals.items()):
        child_id = node_id(child)

        if not child.is_terminal and (
            child_id not in selected_nodes or child.number_visits == 0
        ):
            others_visits += child.number_visits
            others_prior += child.prior
            continue

        label = f"{{ {str(move)} | Q:{child.q():.3f} N:{child.number_visits} VE: {child.value_estimate:.3f} }}"
        if child_id in pv_set:
            color = "red"
        elif child.is_terminal:
            color = "blue"
        else:
            color = None
        graph.node(child_id, label, shape="record", color=color)
        prior_props = {"label": f"{child.prior:.3f}"} if params.show_prior else {}
        graph.edge(parent_id, child_id, **prior_props)
        build_children_nodes(graph, child, child_id, selected_nodes, pv_set, params)

    if params.show_other_node and others_visits > 0:
        others_label = f"{{ othr | Q:?.?? N:{others_visits} }}"
        graph.node(others_id, others_label, shape="record", color="orange")
        prior_props = {"label": f"{others_prior:.2f}"} if params.show_prior else {}
        graph.edge(parent_id, others_id, **prior_props)


def build_graph(
    root: Node,
    fen: str = chess.STARTING_BOARD_FEN,
    params: RenderParams = DEFAULT_RENDER_PARAMS,
) -> graphviz.Digraph:
    graph = graphviz.Digraph(
        "search-tree",
        format=params.imgformat,
        comment="Monte Carlo Search Tree",
        graph_attr={
            "rankdir": "LR",
            # "bgcolor": "#00000000",
            "splines": "line",  # FIXME: can you make beautiful straight lines?
            "size": "150,150",
        },
    )

    pv_set = get_pv_set(root)
    selected_nodes = select_most_visited_nodes(root, params.render_nodes).union(pv_set)

    build_info_node(graph, root)
    build_root_node(params.output_dir, graph, fen, params.imgformat)
    build_children_nodes(graph, root, "root", selected_nodes, pv_set, params)

    return graph


def render_state(
    evaluator: BaseModel,
    board: chess.Board,
    nodes: int,
    render_params: RenderParams = DEFAULT_RENDER_PARAMS,
) -> Node:
    root = run_mcts(board, NodesCount(nodes), evaluator, MCTSparams())

    graph = build_graph(root, params=render_params, fen=board.fen())
    graph.render(
        directory=render_params.output_dir,
        filename="state",
        view=render_params.open_default_gui,
    )
    return root
