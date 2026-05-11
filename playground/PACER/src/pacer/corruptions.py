"""
Dataset corruptions
=======
"""
# src/pacer/corruptions.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from typing import Literal, override

import numpy as np
import numpy.linalg as la
from typingkit.core import RuntimeGeneric, TypedList

from pacer.base import (
    Action,
    Actions,
    Demonstration,
    Demonstrations,
    State,
    States,
)
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    TimeIndex,
    Vector,
)
from pacer.utils import EPS, SEED, set_seed

## ── Corruptions ──────────────────────────────────────────────────────────────


@dataclass
class DemonstrationCorrupter(
    RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]

    @abstractmethod
    def inject_corruptions(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        raise NotImplementedError


@dataclass
class NoisyDemonstrationCorrupter(
    DemonstrationCorrupter[NumDemos, NumPoints, DimState, DimAction]
):
    """Applies i.i.d. corruption processes to actions."""

    _: KW_ONLY
    noise_std: float = 0.0
    outlier_fraction: float = 0.0
    outlier_scale: float = 3.0
    bias_strength: float = 0.0
    dropout_fraction: float = 0.0
    seed: int = SEED

    @override
    def inject_corruptions(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        set_seed(self.seed)
        corrupted_demos = TypedList[
            NumDemos, Demonstration[NumPoints, DimState, DimAction]
        ]()
        for demo in self.demonstrations:
            new_actions = Actions[NumPoints, DimAction]()
            bias_vector = np.random.randn(demo.action_dim)
            bias_vector /= la.norm(bias_vector) + EPS
            for action in demo.actions:
                a = action.copy()

                # Gaussian noise
                if self.noise_std > 0:
                    a += np.random.normal(0, self.noise_std, size=a.shape)

                # Outliers
                if (
                    self.outlier_fraction > 0
                    and random.random() < self.outlier_fraction
                ):
                    a += self.outlier_scale * np.random.randn(*a.shape)

                # Systematic bias
                if self.bias_strength > 0:
                    a += self.bias_strength * bias_vector

                # Dropout
                if (
                    self.dropout_fraction > 0
                    and random.random() < self.dropout_fraction
                ):
                    a = np.zeros_like(a)

                new_actions.append(Action[DimAction](a))
            corrupted_demos.append(
                Demonstration(
                    index=demo.index, states=demo.states.copy(), actions=new_actions
                )
            )
        return Demonstrations(demos=corrupted_demos)


@dataclass(frozen=True)
class SegmentGaussianCorruption(RuntimeGeneric[DimState, DimAction]):
    """
    Deterministic specification for a localized Gaussian-shaped corruption.
    Corruption magnitude follows:
        `envelope(t) = amplitude * exp(-0.5 * ((t - mu) / sigma)^2)`
    over the interval `[start, end)`.
    """

    demo_index: DemoIndex

    start: TimeIndex
    end: TimeIndex

    direction: Vector[int]
    """
    Explicit corruption direction.

    Caller is expected to provide a unit-normalized direction.
    """

    _: KW_ONLY

    amplitude: float
    sigma: float

    target: Literal["ACTION", "STATE"] = "ACTION"

    @property
    def mean(self) -> float:
        return 0.5 * (self.start + self.end)

    def envelope(self, t: TimeIndex) -> float:
        return self.amplitude * np.exp(-0.5 * ((t - self.mean) / self.sigma) ** 2)

    def contains(self, t: TimeIndex) -> bool:
        return self.start <= t < self.end


@dataclass
class SegmentGaussianCorrupter(
    DemonstrationCorrupter[NumDemos, NumPoints, DimState, DimAction]
):
    """
    Applies deterministic segment-localized Gaussian corruptions.

    No randomness is used internally.
    """

    corruptions: Iterable[SegmentGaussianCorruption[DimState, DimAction]]

    @override
    def inject_corruptions(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        corrupted_demos = TypedList[
            NumDemos, Demonstration[NumPoints, DimState, DimAction]
        ]()
        for demo in self.demonstrations:
            demo_corruptions = [
                corruption
                for corruption in self.corruptions
                if corruption.demo_index == demo.index
            ]

            new_states = States[NumPoints, DimState]()
            new_actions = Actions[NumPoints, DimAction]()
            for t in demo.time_indices:
                state = demo.states[t].copy()
                action = demo.actions[t].copy()
                for corruption in demo_corruptions:
                    if not corruption.contains(t):
                        continue
                    perturbation = corruption.envelope(t) * corruption.direction
                    match corruption.target:
                        case "ACTION":
                            action += perturbation
                        case "STATE":
                            state += perturbation
                new_states.append(State[DimState](state))
                new_actions.append(Action[DimAction](action))

            corrupted_demos.append(
                Demonstration(index=demo.index, states=new_states, actions=new_actions)
            )
        return Demonstrations(demos=corrupted_demos)


@dataclass
class PerPhaseBinCorruptionPlanner(
    RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]
):
    """
    Constructs PACER-style corruption plans where:

    - every demonstration contains corruption,
    - every phase region contains at least one clean demo,
    - corruption coverage rotates across demonstrations.

    Example:
        demo 0 -> early phase corrupted
        demo 1 -> middle phase corrupted
        demo 2 -> late phase corrupted
        ...
    """

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]

    _: KW_ONLY

    n_bins: int

    amplitude: float
    sigma_fraction: float

    directions: list[Vector[int]]
    """
    Explicit direction for each demo/bin corruption.

    Length must equal number of demonstrations.
    """

    target: Literal["ACTION", "STATE"] = "ACTION"

    def plan(self) -> TypedList[int, SegmentGaussianCorruption[DimState, DimAction]]:
        assert self.demonstrations.count == len(self.directions)
        corruptions = TypedList[int, SegmentGaussianCorruption[DimState, DimAction]]()

        for demo, direction in zip(self.demonstrations, self.directions, strict=True):
            T = demo.length
            bin_length = max(1, T // self.n_bins)

            # Rotate corruption region across demonstrations
            bin_index = demo.index % self.n_bins

            start = bin_index * bin_length
            end: int
            if bin_index == self.n_bins - 1:
                end = T
            else:
                end = (bin_index + 1) * bin_length

            sigma = max(1.0, (end - start) * self.sigma_fraction)

            corruptions.append(
                SegmentGaussianCorruption(
                    demo_index=demo.index,
                    start=start,
                    end=end,
                    direction=direction,
                    amplitude=self.amplitude,
                    sigma=sigma,
                    target=self.target,
                )
            )

        return corruptions


## ─────────────────────────────────────────────────────────────────────────────
