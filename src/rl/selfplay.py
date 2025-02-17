import datetime
import pathlib
from io import TextIOWrapper

import chess
import chess.pgn

from dinora.models.alphanet import AlphaNet
from dinora.search.mcts import mcts
from dinora.search.noise import apply_noise


class Game:
    def __init__(self, nodes_per_move: int, cpuct: float):
        self.root = None
        self.leaf = None
        self.board = chess.Board()
        self.played_moves = []
        self.nodes_per_move = nodes_per_move
        self.cpuct = cpuct

    def advance(self):
        assert self.root
        move = mcts.most_visited_move(self.root)
        self.board.push(move)
        self.played_moves.append(move)
        self.root = self.root.children[move]
        self.root.parent = None

    def root_ended(self):
        terminal_value = mcts.terminal_solver(self.board)
        game_too_long = self.board.ply() >= 256 * 2
        return terminal_value is not None or game_too_long

    def next(self) -> bool:
        if self.root is None:
            return False

        if self.root.visits > self.nodes_per_move:
            self.advance()
            if self.root_ended():
                return True

        leaf_selected = False
        while not leaf_selected:
            self.leaf = mcts.select_leaf(self.root, self.board, self.cpuct)
            terminal_value = mcts.terminal_solver(self.board)
            if terminal_value is not None:
                priors, value = {}, terminal_value
                mcts.expand(self.leaf, priors)
                mcts.backup(self.leaf, self.board, value)

                if self.root.visits > self.nodes_per_move:
                    self.advance()
                    if self.root_ended():
                        return True
                    return self.next()
            else:
                leaf_selected = True

        return False

    def submit(
        self,
        priors,
        value,
        opening_noise_moves: int,
        dirichlet_alpha: float,
        noise_eps: float,
    ):
        if self.board.ply() < 2 * opening_noise_moves:
            priors = apply_noise(
                priors,
                dirichlet_alpha=dirichlet_alpha,
                noise_eps=noise_eps,
            )

        if self.root is None:
            self.root = mcts.Node(None, value, 1.0, chess.Move.null())
            mcts.expand(self.root, priors)
            return

        assert self.leaf
        mcts.expand(self.leaf, priors)
        mcts.backup(self.leaf, self.board, value)


def save_game_pgn(game: Game, pgn_output: TextIOWrapper):
    current_datetime = datetime.datetime.now()
    utc_datetime = datetime.datetime.now(datetime.timezone.utc)

    game_pgn = chess.pgn.Game(
        headers={
            "Event": "RL selfplay",
            "Site": "Dinora engine",
            "Date": current_datetime.date().strftime(r"%Y.%m.%d"),
            "UTCDate": utc_datetime.date().strftime(r"%Y.%m.%d"),
            "Time": current_datetime.strftime("%H:%M:%S"),
            "UTCTime": utc_datetime.strftime("%H:%M:%S"),
            # "Round": f"Game number {game_number}",
        }
    )
    node: chess.pgn.GameNode = game_pgn
    board = chess.Board()
    for move in game.played_moves:
        node = node.add_variation(move)
        board.push(move)

    if board.ply() >= 256 * 2:
        result = "1/2-1/2"
    else:
        result = board.result(claim_draw=True)
    game_pgn.headers["Result"] = result
    print(game_pgn, end="\n\n", flush=True, file=pgn_output)
    print(".", end="", flush=True)


def selfplay(
    model: AlphaNet,
    games_count: int,
    nodes_per_move: int,
    batch_size: int,
    cpuct: float,
    opening_noise_moves: int,
    dirichlet_alpha: float,
    noise_eps: float,
    pgn_file: pathlib.Path,
):
    games = [Game(nodes_per_move, cpuct) for _ in range(batch_size)]
    completed_games = []

    batch_calls = 0

    pgn_output = pgn_file.open("w", encoding="utf8")

    while len(completed_games) < games_count:
        # gather batch
        for i in range(batch_size):
            game = games[i]
            done = game.next()
            if done:
                completed_games.append(game)
                save_game_pgn(game, pgn_output)
                games[i] = Game(nodes_per_move, cpuct)
                game = games[i]
                game.next()

        batch = [game.board for game in games]
        outs = model.evaluate_batch(batch)

        for i in range(batch_size):
            game = games[i]
            game.submit(*outs[i], opening_noise_moves, dirichlet_alpha, noise_eps)
            # print(f"Game {i} ply", game.board.ply())

        batch_calls += 1
        # print(batch_calls)

    pgn_output.close()


def analyze_pgn(pgn_file: pathlib.Path):
    uniq_positions = set()
    uniq_games = set()

    positions_count = 0
    games_count = 0

    white = 0
    black = 0
    draw = 0

    pgn_output = pgn_file.open("r", encoding="utf8")

    while game := chess.pgn.read_game(pgn_output):
        if game.headers["Result"] == "1-0":
            white += 1
        elif game.headers["Result"] == "0-1":
            black += 1
        else:
            draw += 1

        games_count += 1

        board = chess.Board()
        first_moves = []

        ply_count = 0

        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            uniq_positions.add(board.fen())
            ply_count += 1

            if i < 5:
                first_moves.append(move.uci())

        positions_count += ply_count

        game_hash = "".join(first_moves) + board.fen()
        uniq_games.add(game_hash)

    print(f"Uniq games {(len(uniq_games) / games_count) * 100:.3f}%")
    print(f"Uniq positions rate {(len(uniq_positions) / positions_count) * 100:.3f}%")
    print("White wins:", white)
    print("Black wins:", black)
    print("Draw:", draw)
    print("Total games", white + black + draw)


if __name__ == "__main__":
    from dinora.models import model_selector

    model = model_selector("alphanet", None, "cuda")
    pgn_file = pathlib.Path("example.pgn")
    assert isinstance(model, AlphaNet)
    selfplay(
        model,
        games_count=15,
        nodes_per_move=80,
        batch_size=15,
        cpuct=3.0,
        opening_noise_moves=10,
        dirichlet_alpha=0.1,
        noise_eps=0.3,
        pgn_file=pgn_file,
    )
    print()
    analyze_pgn(pgn_file)
