import argparse

import dinora.train.compact_dataset.make_cli as make_dataset_cli
import dinora.train.compact_dataset.upload_cli as upload_dataset_cli
import dinora.uci.cli as uci_cli


def run_cli() -> None:
    parser = build_root_cli()
    args = parser.parse_args()

    if args.subcommand == "make_dataset":
        make_dataset_cli.run_cli(args)
    elif args.subcommand == "upload_dataset":
        upload_dataset_cli.run_cli(args)
    else:
        uci_cli.run_cli(args)


def build_root_cli() -> argparse.ArgumentParser:
    parser = uci_cli.build_parser()
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    make_dataset_cli.build_parser(subparsers)
    upload_dataset_cli.build_parser(subparsers)

    return parser
