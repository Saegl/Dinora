import argparse

import dinora.uci.cli as uci_cli


def run_cli() -> None:
    parser = build_root_cli()
    args = parser.parse_args()

    uci_cli.run_cli(args)


def build_root_cli() -> argparse.ArgumentParser:
    parser = uci_cli.build_parser()
    return parser
