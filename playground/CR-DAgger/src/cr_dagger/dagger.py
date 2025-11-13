"""
DAgger
=======
src/cr_dagger/dagger.py
"""

from typing import Generic

import numpy.typing as npt
from gymnasium import Env
from gymnasium.core import ActType, ObsType

from .base import Dataset, Expert, Policy

NDArray = npt.NDArray


class DAgger(Generic[ObsType, ActType]):
    def __init__(
        self,
        env: Env[ObsType, ActType],
        expert: Expert[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        num_iterations: int = 10,
        episodes_per_iter: int = 5,
    ) -> None:
        self.env = env
        self.expert = expert
        self.policy = policy
        self.num_iterations = num_iterations
        self.episodes_per_iter = episodes_per_iter
        self.dataset = Dataset[ObsType, ActType]()

    def collect_data(self, use_policy: bool = True) -> Dataset[ObsType, ActType]:
        """Collect trajectories using the current policy or expert."""
        data = Dataset[ObsType, ActType]()
        for _ in range(self.episodes_per_iter):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.policy(state) if use_policy else self.expert.act(state)
                next_state, _, done, _, _ = self.env.step(action)
                expert_action = self.expert.act(state)
                data.update(state, expert_action)
                state = next_state
        return data

    def train(self) -> None:
        """Run the DAgger training loop."""
        print("Collecting initial expert demonstrations...")
        self.dataset = self.collect_data(use_policy=False)
        self.policy.update(self.dataset)

        for iteration in range(self.num_iterations):
            print(f"=== Iteration {iteration + 1}/{self.num_iterations} ===")
            new_data = self.collect_data(use_policy=True)
            self.dataset += new_data
            self.policy.update(self.dataset)
            print(f"Dataset size: {len(self.dataset.states)} samples")
