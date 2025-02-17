"""
WIP: this is pretty much in draft state, don't expect meaningful results
"""

import random
from pathlib import Path

import optuna

from cploss.evaluate import calc_engine_cploss, load_cploss, make_stopper_creator
from dinora.models import model_selector
from dinora.models.alphanet import AlphaNet


class CPLossObjective:
    def __init__(self):
        model_path = Path("models/alphanet_rerun-valid-state-4.ckpt")
        loaddir = Path("data/cploss/elite")

        model = model_selector("alphanet", model_path, "cuda")
        assert isinstance(model, AlphaNet)

        self.model = model
        self.stopper_creator = make_stopper_creator(movetime=1.0)
        _, _, self.positions = load_cploss(loaddir, 3000)
        random.shuffle(self.positions)
        self.positions = self.positions[0:100]

    def __call__(self, trial: optuna.Trial):
        batch_size = trial.suggest_int("batch_size", 1, 256)
        virtual_visits = trial.suggest_int("virtual_visits", 1, batch_size)
        max_collisions = trial.suggest_int("max_collisions", 1, virtual_visits)

        params = {
            "cpuct": trial.suggest_float("cpuct", 1.0, 10.0),
            "batch_size": batch_size,
            "virtual_visits": virtual_visits,
            "max_collisions": max_collisions,
        }

        return calc_engine_cploss(
            self.model, self.positions, "mcts_batch", self.stopper_creator, params
        )


def start_tuner():
    objective = CPLossObjective()

    # run `optuna-dashboard sqlite:///tuner.sqlite3` to see realtime progress
    study = optuna.create_study(storage="sqlite:///tuner.sqlite3")
    study.optimize(objective, n_trials=60)

    print("Best params", study.best_params)


if __name__ == "__main__":
    start_tuner()
