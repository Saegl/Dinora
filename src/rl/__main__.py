import pathlib

from rl.pipeline import Config, start_rl

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("rl")
    parser.add_argument("config", type=pathlib.Path)
    args = parser.parse_args()

    config = Config.from_file(args.config)
    start_rl(config)
