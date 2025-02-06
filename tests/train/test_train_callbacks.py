import lightning.pytorch as pl
import pytest
from lightning.pytorch.loggers import Logger

from dinora.models.alphanet import AlphaNet
from train.train_callbacks import (
    BoardsEvaluator,
    SampleGameGenerator,
    ValidationCheckpointer,
)

micro_alphanet_conf = {
    "filters": 16,
    "res_blocks": 2,
    "policy_channels": 8,
    "value_channels": 2,
    "value_fc_hidden": 32,
    "learning_rate": 0.001,
    "lr_scheduler_gamma": 1.0,
    "lr_scheduler_freq": 1000,
}


class MockLogger(Logger):
    @property
    def name(self):
        return "MockLogger"

    @property
    def version(self):
        return -1

    def log_metrics(self, metrics, step=None) -> None:
        ""

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        ""

    def log_text(self, *args, **kwargs):
        pass


def test_sample_game_generator():
    callback = SampleGameGenerator()
    trainer = pl.Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    callback.on_fit_start(trainer, pl_module)
    callback.on_validation_end(trainer, pl_module)


def test_boards_evaluator():
    callback = BoardsEvaluator()
    trainer = pl.Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    callback.on_validation_end(trainer, pl_module)


@pytest.mark.disk_usage
def test_validation_checkpointer():
    callback = ValidationCheckpointer()
    trainer = pl.Trainer(logger=MockLogger())
    pl_module = AlphaNet(**micro_alphanet_conf)

    import wandb

    wandb.init(mode="disabled")

    callback.on_validation_end(trainer, pl_module)
    callback.on_train_end(trainer, pl_module)
