from __future__ import annotations

import argparse
import pathlib
import typing

from bench.selfplay import selfplay

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench", description="Benchmark speed of engine"
    )
    parser.add_argument(
        "--model",
        default="alphanet",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--weights",
        help="Path to model weights",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
        type=str,
    )
    return parser


def run_cli(args: Args) -> None:
    selfplay(args.model, args.weights, args.device)
