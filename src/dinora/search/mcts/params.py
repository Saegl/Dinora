from dataclasses import dataclass, field


@dataclass
class MCTSparams:
    cpuct: float = field(default=3.0)
    # random
    first_n_moves: int = field(default=15)
    dirichlet_alpha: float = field(default=0.3)
    noise_eps: float = field(default=0.0)  # set to 0.0 to disable random
