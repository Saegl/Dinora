from __future__ import annotations

import argparse
import pathlib
import typing

if typing.TYPE_CHECKING:
    Args = argparse.Namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_onnx", description="Export torch model as onnx"
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
    return parser


def run_cli(args: Args) -> None:
    from onnxtools.export_onnx import export_onnx

    export_onnx(args.model, args.weights)
