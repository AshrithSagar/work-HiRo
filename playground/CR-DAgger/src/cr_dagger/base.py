"""
Base
=======
src/cr_dagger/base.py
"""

from abc import ABC, abstractmethod
from typing import Generic, Self, Sequence

from gymnasium.core import ActType, ObsType


class Dataset(Generic[ObsType, ActType]):
    def __init__(
        self, states: Sequence[ObsType] = [], actions: Sequence[ActType] = []
    ) -> None:
        self.states = list(states)
        self.actions = list(actions)

    def __add__(self, other: "Dataset[ObsType, ActType]") -> Self:
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        return self

    def update(self, state: ObsType, action: ActType) -> None:
        self.states.append(state)
        self.actions.append(action)


class Policy(Generic[ObsType, ActType], ABC):
    @abstractmethod
    def __call__(self, state: ObsType) -> ActType: ...

    @abstractmethod
    def update(self, dataset: Dataset[ObsType, ActType]) -> None: ...


class Expert(Generic[ObsType, ActType], ABC):
    @abstractmethod
    def act(self, state: ObsType) -> ActType: ...
