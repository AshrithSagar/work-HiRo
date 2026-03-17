"""
Base
=======
Core data structures for representing demonstrations and samples.
"""
# src/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeAlias, overload

import numpy as np
from typingkit.core import RuntimeGeneric, TypedDict, TypedList
from typingkit.numpy._typed.helpers import Array1D

from pacer.typings import DimAction, DimState, NumDemos, NumPoints, npDType

## ── Base ─────────────────────────────────────────────────────────────────────

State: TypeAlias = Array1D[DimState, np.dtype[npDType]]  # x_{i, t} \in R^{d_x}


class States(TypedList[NumPoints, State[DimState]]):
    def coord(self, dim: int) -> Array1D[NumPoints, np.dtype[npDType]]:
        return Array1D[NumPoints, np.dtype[npDType]](np.asarray(self)[:, dim])


StatesCollection: TypeAlias = TypedList[NumDemos, States[NumPoints, DimState]]

# ──────────────────────────────────────────────────────────────────────────────

Action: TypeAlias = Array1D[DimAction, np.dtype[npDType]]  # a_{i, t} \in R^{d_a}


class Actions(TypedList[NumPoints, Action[DimAction]]):
    def coord(self, dim: int) -> Array1D[NumPoints, np.dtype[npDType]]:
        return Array1D[NumPoints, np.dtype[npDType]](np.asarray(self)[:, dim])


ActionsCollection: TypeAlias = TypedList[NumDemos, Actions[NumPoints, DimAction]]

# ──────────────────────────────────────────────────────────────────────────────

Phase: TypeAlias = npDType  # tau \in [0, 1]
Phases: TypeAlias = TypedList[NumPoints, Phase]
PhasesCollection: TypeAlias = TypedList[NumDemos, Phases[NumPoints]]

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

ZScore: TypeAlias = npDType  # z_{i, t}
ZScores: TypeAlias = TypedList[NumPoints, ZScore]
ZScoresCollection: TypeAlias = TypedList[NumDemos, ZScores[NumPoints]]

TrustValue: TypeAlias = npDType  # w_{i, t}
TrustValues: TypeAlias = TypedList[NumPoints, TrustValue]
TrustValuesCollection: TypeAlias = TypedList[NumDemos, TrustValues[NumPoints]]

# ──────────────────────────────────────────────────────────────────────────────

DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
DemoIndices: TypeAlias = TypedList[NumPoints, DemoIndex]
DemoIndicesCollection: TypeAlias = TypedList[NumDemos, DemoIndices[NumPoints]]

TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
TimeIndices: TypeAlias = TypedList[NumPoints, TimeIndex]
TimeIndicesCollection: TypeAlias = TypedList[NumDemos, TimeIndices[NumPoints]]


class SampleIndex(NamedTuple):  # (i, t)
    demo: DemoIndex  # i
    time: TimeIndex  # t


SampleIndices: TypeAlias = TypedList[NumPoints, SampleIndex]
SampleIndicesCollection: TypeAlias = TypedList[NumDemos, SampleIndices[NumPoints]]

BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

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
        return DemoIndices[NumDemos](range(self.__len__()))

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.demos[0].state_dim

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.demos[0].action_dim


## ─────────────────────────────────────────────────────────────────────────────
