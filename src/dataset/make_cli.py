"""
Steps to make your own dataset from pgn games:
1. Download your pgn to data/<folder_name>/<your.pgn>
2. Split your pgn to smaller parts of 10k games each with
`pgn-extract.exe -#10000 <your.pgn>` and place it under
data/<folder_name2>
3. Install deps for this script
`pip install -e . chess numpy wandb requests tqdm`
4. Run this program with
`make_compact_dataset.py <folder_name2> <folder_name3>`
You can append `--q-nodes=10` if you want add stockfish evals to
the resulting dataset
5. Now you can train neural network with this dataset, use <folder_name3>
as `dataset_label` in dinora.train
"""

from __future__ import annotations

import argparse
import pathlib
import typing

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        name="make", help="Tool to convert pgns to compact dataset"
    )
    parser.add_argument(
        "pgn_dir",
        help="directory of pgn files",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output_dir",
        help="directory to save compact dataset",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--files-count",
        help="number of files to convert, all if not specified",
        type=int,
    )
    parser.add_argument(
        "--q-nodes",
        help="pass positive integer to add stockfish q values",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train",
        help="train percentage",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--val",
        help="val percentage",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--test",
        help="test percentage",
        type=float,
        default=0.1,
    )


def run_cli(args: Args) -> None:
    from dataset.make import convert_dir

    convert_dir(
        args.pgn_dir,
        args.output_dir,
        args.files_count,
        args.q_nodes,
        args.train,
        args.val,
        args.test,
    )
