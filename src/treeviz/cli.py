from __future__ import annotations

import argparse
import pathlib
import typing

import chess

from dinora import PROJECT_ROOT
from dinora.engine import Engine

if typing.TYPE_CHECKING:
    Args = argparse.Namespace


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/treeviz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="treeviz", description="Render MCTS tree")
    parser.add_argument(
        "--model",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
    )
    parser.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="Chess FEN of root node",
    )
    parser.add_argument(
        "--nodes",
        default=15,
        help="Number of nodes to search by engine",
        type=int,
    )
    parser.add_argument(
        "--render-nodes",
        default=10,
        help="Number of nodes to render",
        type=int,
    )
    parser.add_argument(
        "--disable-other-node",
        help="`Other node` shows statistics of non rendered nodes",
        action="store_true",
    )
    parser.add_argument(
        "--disable-prior",
        help="Disable priors on arrows between nodes",
        action="store_true",
    )
    parser.add_argument(
        "--disable-gui",
        help="When not disabled, rendered tree will be opened"
        " in default OS apps: default browser (Google Chrome) or photos app",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated images",
        default=DEFAULT_OUTPUT_DIR,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--imgformat",
        default="svg",
        help="Output image format svg/png",
    )
    parser.add_argument("--selection-policy-name", default="")
    return parser


def run_cli(args: Args) -> None:
    from treeviz.treeviz import RenderParams, render_state

    engine = Engine("ext_mcts", args.model, args.weights, args.device)
    engine.load_model()
    board = chess.Board(fen=args.fen)

    render_state(
        engine.model,
        board=board,
        nodes=args.nodes,
        render_params=RenderParams(
            render_nodes=args.render_nodes,
            show_other_node=not args.disable_other_node,
            show_prior=not args.disable_prior,
            open_default_gui=not args.disable_gui,
            output_dir=args.output_dir,
            imgformat=args.imgformat,
        ),
    )
