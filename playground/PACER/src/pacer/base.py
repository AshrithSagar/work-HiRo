"""
Base
=======
Core data structures for representing demonstrations and samples.
"""
# src/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Self, cast, overload, override

import numpy as np
import optype.numpy as onp
from typingkit.core import RuntimeGeneric, TypedDict, TypedList

from pacer.typings import (
    DemoIndex,
    DemoIndices,
    DimAction,
    DimState,
    Matrix,
    NumDemos,
    NumPoints,
    SampleIndex,
    TimeIndex,
    TimeIndices,
    Vector,
    npDType,
)

## ── Base ─────────────────────────────────────────────────────────────────────


class State(Vector[DimState]):  # x_{i, t} \in R^{d_x}
    def __new__(cls, object: onp.ToArrayStrict1D) -> Self:
        return cast(Self, super().__new__(cls, object, dtype=npDType))

    @classmethod
    def zeros(cls, state_dim: DimState) -> Self:
        return cls(np.zeros((state_dim,)))


class States(TypedList[NumPoints, State[DimState]]):
    def numpy(self) -> Matrix[NumPoints, DimState]:
        return Matrix[NumPoints, DimState](self)

    def coord(self, dim: int) -> Vector[NumPoints]:
        return Vector[NumPoints](self.numpy()[:, dim])

    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.time_indices.length
        d_x = demonstration.state_dim
        return cls.full(T_i, State[DimState].zeros(d_x))


class StatesCollection(TypedList[NumDemos, States[NumPoints, DimState]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        N = demonstrations.__len__()
        return cls.full(
            N, lambda i: States[NumPoints, DimState].zeros_like(demonstrations[i])
        )


# ──────────────────────────────────────────────────────────────────────────────


class Action(Vector[DimAction]):  # a_{i, t} \in R^{d_a}
    def __new__(cls, object: onp.ToArrayStrict1D) -> Self:
        return cast(Self, super().__new__(cls, object, dtype=npDType))

    @classmethod
    def zeros(cls, action_dim: DimAction) -> Self:
        return cls(np.zeros((action_dim,)))


class Actions(TypedList[NumPoints, Action[DimAction]]):
    def numpy(self) -> Matrix[NumPoints, DimAction]:
        return Matrix[NumPoints, DimAction](self)

    def coord(self, dim: int) -> Vector[NumPoints]:
        return Vector[NumPoints](self.numpy()[:, dim])

    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.time_indices.length
        d_a = demonstration.action_dim
        return cls.full(T_i, Action[DimAction].zeros(d_a))


class ActionsCollection(TypedList[NumDemos, Actions[NumPoints, DimAction]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        N = demonstrations.__len__()
        return cls.full(
            N, lambda i: Actions[NumPoints, DimAction].zeros_like(demonstrations[i])
        )


# ──────────────────────────────────────────────────────────────────────────────


# (x, a)
@dataclass
class StateActionPair(RuntimeGeneric[DimState, DimAction]):
    """A container for a State-Action pair."""

    state: State[DimState]  # x
    action: Action[DimAction]  # a


# (x_{i, t}, a_{i, t})
@dataclass
class Sample(StateActionPair[DimState, DimAction]):
    """A State-Action pair in the context of Demonstrations."""

    index: SampleIndex  # (i, t)


# [(x_{t}, a_{t})]_{t = 1}^{T}
class Samples(TypedDict[NumPoints, TimeIndex, Sample[DimState, DimAction]]):
    @property
    def time_indices(self) -> TimeIndices[NumPoints]:
        return TimeIndices[NumPoints](self.keys())

    def states(self) -> States[NumPoints, DimState]:
        return States[NumPoints, DimState](sample.state for sample in self.values())

    def actions(self) -> Actions[NumPoints, DimAction]:
        return Actions[NumPoints, DimAction](sample.action for sample in self.values())


# [[(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}]_{i = 1}^{N}
class SamplesCollection(
    TypedDict[NumDemos, DemoIndex, Samples[NumPoints, DimState, DimAction]]
):
    """A collection of state-action pairs (a sequence of state-action pair)."""

    @overload
    def __getitem__(
        self,
        index: DemoIndex,  # i
        /,
    ) -> Samples[
        NumPoints, DimState, DimAction
    ]: ...  # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
    @overload
    def __getitem__(  # ty: ignore[invalid-overload]
        self,
        index: SampleIndex,  # (i, t)
        /,
    ) -> Sample[DimState, DimAction]: ...  # (x_{i, t}, a_{i, t})
    #
    @override
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Samples[NumPoints, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case SampleIndex(i, t):
                return super().__getitem__(i)[t]
            case DemoIndex() as i:
                return super().__getitem__(i)
        raise IndexError

    @property
    def demo_indices(self) -> DemoIndices[NumDemos]:
        return DemoIndices[NumDemos](self.keys())

    @property
    def demo_lengths(self) -> TypedList[NumDemos, NumPoints]:
        return TypedList[NumDemos, NumPoints](
            samples.__len__() for samples in self.values()
        )

    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Iterator[Sample[DimState, DimAction]]:  # (N x T_) or (N-1 x T_)
        for i in self.demo_indices:
            if i == LOO_demo_index:
                continue
            yield from self[i].values()

    def states(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> States[Any, DimState]:
        # (N x T_) or (N-1 x T_)
        return States[Any, DimState](
            sample.state for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    def actions(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Actions[Any, DimAction]:
        # (N x T_) or (N-1 x T_)
        return Actions[Any, DimAction](
            sample.action for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )


# ──────────────────────────────────────────────────────────────────────────────


# Behaves like Samples
@dataclass(kw_only=True)
class Demonstration(RuntimeGeneric[NumPoints, DimState, DimAction]):  # D_i
    index: DemoIndex  # i
    states: States[NumPoints, DimState]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: Actions[NumPoints, DimAction]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert self.states.length == self.actions.length

    def __len__(self) -> NumPoints:
        assert self.states.length == self.actions.length
        return self.states.length  # T_i

    def __getitem__(
        self, index: TimeIndex, /
    ) -> Sample[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return Sample(
            index=SampleIndex(demo=self.index, time=index),
            state=self.states[index],
            action=self.actions[index],
        )

    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
        for t in self.time_indices:
            yield self[t]

    @property
    def time_indices(self) -> TimeIndices[NumPoints]:
        return TimeIndices[NumPoints](range(self.__len__()))

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.states[0].shape[0]

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.actions[0].shape[0]

    def sample(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})

    def samples(self) -> Samples[NumPoints, DimState, DimAction]:
        return Samples(
            TypedDict[NumPoints, TimeIndex, Sample[DimState, DimAction]](
                (sample.index.time, sample) for sample in self
            )
        )


# Behaves like SamplesCollection
@dataclass(slots=True)
class Demonstrations(
    RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]
):  # [D_i]_{i = 1}^{N}
    demos: TypedList[NumDemos, Demonstration[NumPoints, DimState, DimAction]]

    def __len__(self) -> NumDemos:
        return self.demos.length  # N

    @overload
    def __getitem__(
        self, index: DemoIndex, /
    ) -> Demonstration[NumPoints, DimState, DimAction]: ...
    @overload
    def __getitem__(self, index: SampleIndex, /) -> Sample[DimState, DimAction]: ...  # ty: ignore[invalid-overload]
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Demonstration[NumPoints, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case SampleIndex(i, t):
                return self.demos[i][t]
            case DemoIndex() as i:
                return self.demos[i]
        raise IndexError

    def __iter__(self) -> Iterator[Demonstration[NumPoints, DimState, DimAction]]:
        yield from self.demos  # D_i

    @property
    def demo_indices(self) -> DemoIndices[NumDemos]:
        return DemoIndices[NumDemos](demo.index for demo in self.demos)

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.demos[0].state_dim

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.demos[0].action_dim


## ─────────────────────────────────────────────────────────────────────────────
