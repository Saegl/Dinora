import argparse

import dataset.make_cli
import dataset.upload_cli


def run_cli() -> None:
    parser = build_root_cli()
    args = parser.parse_args()

    if args.subcommand == "make":
        dataset.make_cli.run_cli(args)
    elif args.subcommand == "upload":
        dataset.upload_cli.run_cli(args)
    else:
        parser.print_help()


def build_root_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset")
    subparsers = parser.add_subparsers(title="Subcommands", dest="subcommand")

    dataset.make_cli.build_parser(subparsers)
    dataset.upload_cli.build_parser(subparsers)

    return parser


run_cli()
