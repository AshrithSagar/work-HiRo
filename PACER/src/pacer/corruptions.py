"""
Dataset corruptions
=======
"""
# src/pacer/corruptions.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
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


# ── Noisy Demonstration Corrupter ─────────────────────────────────────────────


@dataclass(kw_only=True)
class NoisyCorruptionConfig:
    noise_std: float = 0.0
    outlier_fraction: float = 0.0
    outlier_scale: float = 3.0
    bias_strength: float = 0.0
    dropout_fraction: float = 0.0


@dataclass
class NoisyDemonstrationCorrupter(
    DemonstrationCorrupter[NumDemos, NumPoints, DimState, DimAction]
):
    """Applies i.i.d. corruption processes to actions."""

    _: KW_ONLY
    config: NoisyCorruptionConfig = field(default_factory=NoisyCorruptionConfig)
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
                if self.config.noise_std > 0:
                    a += np.random.normal(0, self.config.noise_std, size=a.shape)

                # Outliers
                if (
                    self.config.outlier_fraction > 0
                    and random.random() < self.config.outlier_fraction
                ):
                    a += self.config.outlier_scale * np.random.randn(*a.shape)

                # Systematic bias
                if self.config.bias_strength > 0:
                    a += self.config.bias_strength * bias_vector

                # Dropout
                if (
                    self.config.dropout_fraction > 0
                    and random.random() < self.config.dropout_fraction
                ):
                    a = np.zeros_like(a)

                new_actions.append(Action[DimAction](a))
            corrupted_demos.append(
                Demonstration(
                    index=demo.index, states=demo.states.copy(), actions=new_actions
                )
            )
        return Demonstrations(demos=corrupted_demos)


# ── Segment Gaussian Corrupter ────────────────────────────────────────────────

type NormalOrientation = Literal["LEFT", "RIGHT", "AWAY_FROM_CENTRE", "TOWARDS_CENTRE"]


@dataclass(frozen=True)
class SegmentGaussianCorruption(RuntimeGeneric[DimState, DimAction]):
    """
    Deterministic specification for a localized Gaussian-shaped corruption.
    Corruption magnitude follows:
        `envelope(t) = amplitude * exp(-0.5 * ((t - mu) / sigma)^2)`
    over the interval `[start, end)`.
    """

    _: KW_ONLY
    demo_index: DemoIndex
    start: TimeIndex
    end: TimeIndex
    direction: Vector[int] | None = None
    normal_orientation: NormalOrientation = "AWAY_FROM_CENTRE"
    amplitude: float = 1.0
    sigma: float = 0.25
    target: Literal["ACTION", "STATE"] = "ACTION"

    @property
    def mean(self) -> float:
        return 0.5 * (self.start + self.end)

    def envelope(self, t: TimeIndex) -> float:
        return self.amplitude * np.exp(-0.5 * ((t - self.mean) / self.sigma) ** 2)

    def contains(self, t: TimeIndex) -> bool:
        return self.start <= t < self.end

    def infer_perpendicular_direction(
        self,
        demo: Demonstration[NumPoints, DimState, DimAction],
        *,
        centre: Vector[int] | None = None,
    ) -> Vector[int]:
        p0 = demo.states[self.start]
        p1 = demo.states[self.end - 1]
        tangent = p1 - p0
        tangent /= la.norm(tangent) + EPS
        if tangent.shape[0] != 2:
            raise ValueError("Only 2D states supported currently.")
        left = np.array([-tangent[1], tangent[0]])
        right = -left
        match self.normal_orientation:
            case "LEFT":
                normal = left
            case "RIGHT":
                normal = right
            case "AWAY_FROM_CENTRE":
                assert centre is not None
                midpoint = 0.5 * (p0 + p1)
                outward = midpoint - centre
                normal = left if np.dot(left, outward) >= 0 else right
            case "TOWARDS_CENTRE":
                assert centre is not None
                midpoint = 0.5 * (p0 + p1)
                inward = centre - midpoint
                normal = left if np.dot(left, inward) >= 0 else right
        normal /= la.norm(normal) + EPS
        return Vector[int](normal)


@dataclass
class SegmentGaussianCorrupter(
    DemonstrationCorrupter[NumDemos, NumPoints, DimState, DimAction]
):
    """Applies deterministic segment-localized Gaussian corruptions."""

    corruptions: Iterable[SegmentGaussianCorruption[DimState, DimAction]]

    def get_demonstrations_centre(self) -> Vector[int]:
        points: list[State[DimState]] = []
        for demo in self.demonstrations:
            for state in demo.states:
                points.append(state)
        centre = np.mean(np.asarray(points), axis=0)
        return Vector[int](centre)

    @override
    def inject_corruptions(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        centre = self.get_demonstrations_centre()
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
                    direction = (
                        corruption.direction
                        if corruption.direction is not None
                        else corruption.infer_perpendicular_direction(
                            demo, centre=centre
                        )
                    )
                    perturbation = corruption.envelope(t) * direction
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
    normal_orientation: NormalOrientation = "AWAY_FROM_CENTRE"
    target: Literal["ACTION", "STATE"] = "ACTION"

    def plan(self) -> list[SegmentGaussianCorruption[DimState, DimAction]]:
        corruptions = list[SegmentGaussianCorruption[DimState, DimAction]]()
        for demo in self.demonstrations:
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
                    direction=None,
                    normal_orientation=self.normal_orientation,
                    amplitude=self.amplitude,
                    sigma=sigma,
                    target=self.target,
                )
            )
        return corruptions


## ─────────────────────────────────────────────────────────────────────────────
