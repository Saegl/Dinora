from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import typing
from datetime import datetime, timezone

import tqdm
from colorama import Fore, just_fix_windows_console

from elofish.elofish import load_players, play_match
from elofish.glicko2 import glicko2

if typing.TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]
    Args = argparse.Namespace


def run_cli(args: Args) -> None:
    just_fix_windows_console()

    with args.config.open(encoding="utf8") as f:
        config = json.load(f)

    env = glicko2.Glicko2()  # type: ignore
    teacher_player, student_player = load_players(config)

    start_datetime = datetime.now(timezone.utc)
    dir_name = f"{start_datetime.strftime('%Y-%m-%d %H:%M')} {student_player.name}"

    output_dir = pathlib.Path("reports") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pgn_file = output_dir / "game.pgn"
    pgn_output = pgn_file.open("wt", encoding="utf8")

    report_file = output_dir / "report.txt"
    report_output = report_file.open("wt", encoding="utf8")

    logs_file = output_dir / "logs.txt"
    logs_output = logs_file.open("wt", encoding="utf8")

    config_file = output_dir / "config.json"
    shutil.copyfile(args.config, config_file)

    options_file = output_dir / "options.json"
    json.dump(
        [
            teacher_player.dump_info(),
            teacher_player.dump_options(),
            student_player.dump_info(),
            student_player.dump_options(),
        ],
        options_file.open("wt", encoding="utf8"),
    )

    report_output.write(f"Start date {start_datetime.strftime('%Y-%m-%d')}\n")
    report_output.write(f"Start time {start_datetime.strftime('%H:%M')}\n")
    report_output.write(
        f"Initial Rating {student_player.rating.mu} ({student_player.rating.phi})\n"
    )

    wins = 0
    draws = 0
    losses = 0

    try:
        for game in tqdm.tqdm(
            play_match(
                env,
                student_player,
                teacher_player,
                max_games=config["max_games"],
                min_phi=config["min_phi"],
                min_mu=config["min_mu"],
                game_tick=args.game_tick,
            )
        ):
            # print(game, end="\n\n", flush=True)
            print(game, end="\n\n", flush=True, file=pgn_output)

            round_ind = game.headers["Round"]
            elo = game.headers[
                "WhiteElo"
            ]  # Doesn't matter white or black since Teacher copycats Student
            student_rating_deviation = game.headers["StudentRatingDeviation"]
            student_nodes = game.headers["AvgStudentNodes"]
            teacher_nodes = game.headers["AvgTeacherNodes"]

            result = game.headers["Result"]
            student_is_white = game.headers["White"] == student_player.fullname

            if (
                student_is_white
                and result == "1-0"
                or not student_is_white
                and result == "0-1"
            ):
                result_string = f"{Fore.GREEN}Win{Fore.RESET}"
                wins += 1
            elif (
                student_is_white
                and result == "0-1"
                or not student_is_white
                and result == "1-0"
            ):
                result_string = f"{Fore.RED}Loss{Fore.RESET}"
                losses += 1
            else:
                result_string = f"{Fore.YELLOW}Draw{Fore.RESET}"
                draws += 1

            termination = game.headers["TerminationEnum"]

            game_log = (
                f"{Fore.BLUE}{round_ind}{Fore.RESET}:"
                f" {result_string} by {termination}"
                f", Elo = {elo} ({student_rating_deviation})"
                f", {Fore.MAGENTA}StudentSpeed{Fore.RESET} = {student_nodes} n/ply"
                f", {Fore.CYAN}TeacherSpeed{Fore.RESET} = {teacher_nodes} n/ply"
            )

            tqdm.tqdm.write(game_log)
            print(game_log, file=logs_output)

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}Elo estimator early stopping{Fore.RESET}")

    end_datetime = datetime.now(timezone.utc)

    report_output.write(f"Wins {wins}\n")
    report_output.write(f"Draws {draws}\n")
    report_output.write(f"Losses {losses}\n")
    report_output.write(f"End time {end_datetime.strftime('%H:%M')}\n")
    report_output.write(f"Time taken {(end_datetime - start_datetime)}\n")
    report_output.write(
        f"Final rating {student_player.rating.mu} ({student_player.rating.phi})\n"
    )

    print(f"{Fore.GREEN}Result saved at {output_dir}{Fore.RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("elofish")

    parser.add_argument(
        "config",
        help="Path to config, look to configs/elo_match",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--game-tick",
        help="Prints move and nodes after each move (for debug)",
        action="store_true",
    )

    args = parser.parse_args()
    run_cli(args)
