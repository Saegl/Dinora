from dataclasses import dataclass, field


@dataclass
class MCTSparams:
    cpuct: float = field(default=3.0)
    # batch
    batch_size: int = field(default=256)
    virtual_loss: int = field(default=40)
    max_collisions: int = field(default=15)
