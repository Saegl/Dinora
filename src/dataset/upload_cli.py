from __future__ import annotations

import argparse
import pathlib
import typing

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def build_parser(subparsers: Subparsers) -> None:
    parser = subparsers.add_parser(
        name="upload", help="Tool to upload dataset to wandb"
    )
    parser.add_argument(
        "dataset_dir",
        help="Dataset directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "wandb_label",
        help="Wandb dataset label",
        type=str,
    )


def run_cli(args: Args) -> None:
    from dataset.upload import upload_dataset

    upload_dataset(args.dataset_dir, args.wandb_label)
