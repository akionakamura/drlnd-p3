from dataclasses import dataclass
from typing import List

import numpy as np



@dataclass(frozen=True, eq=True)
class RunConfig:
    """Holds the main configurations of a run, to facilitate experimentation."""
    learn_step: int
    sync_step: int
    batch_size: int
    gamma: float
    epsilon_decay: float


@dataclass
class RunExperiments:
    """Holds the several configurations to run experiments to."""
    learn_steps: List[int]
    sync_steps: List[int]
    batch_sizes: List[int]
    gammas: List[float]
    epsilon_decays: List[float]

    def get_configs(self, n_configs):
        """Return `n_configs` random configurations."""
        combinations = np.array(np.meshgrid(
            self.learn_steps,
            self.sync_steps,
            self.batch_sizes,
            self.gammas,
            self.epsilon_decays
        )).T.reshape(-1, 5)

        n = min(n_configs, combinations.shape[0])
        idxs = np.random.choice(combinations.shape[0], n)

        return [RunConfig(
            learn_step=int(combinations[i, 0]),
            sync_step=int(combinations[i, 1]),
            batch_size=int(combinations[i, 2]),
            gamma=float(combinations[i, 3]),
            epsilon_decay=float(combinations[i, 4]),
        ) for i in idxs]
