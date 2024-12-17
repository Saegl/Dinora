"""
Noise can be applied at root
to change engine behavior in each game
"""

import numpy as np

from dinora.models.base import Priors


def apply_noise(priors: Priors, dirichlet_alpha: float, noise_eps: float) -> Priors:
    """
    dirichlet_alpha: Controls the concentration of the Dirichlet noise.
        Range: [0.01 (very spiky), 10.0 (nearly uniform)].
        Increasing this value makes the noise more evenly distributed
        decreasing it concentrates the noise on fewer moves.

    noise_eps: Blending factor for noise and original priors.
        Range: [0.0 (disable noise), 1.0 (all noise)].
    """
    noise = np.random.dirichlet([dirichlet_alpha] * len(priors))  # Spikiness
    return {
        move: (1 - noise_eps) * prior + noise_eps * noise[i]
        for i, (move, prior) in enumerate(priors.items())
    }
