"""
Base
=======
"""
# src/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Generic, overload

from typingkit.core import TypedList
from typingkit.numpy import enforce_shapes

from pacer.typings import (
    Action,
    Actions,
    DemoIndex,
    DemoIndices,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    SampleIndex,
    State,
    States,
    TimeIndex,
    TimeIndices,
)

## ── Base ─────────────────────────────────────────────────────────────────────


# (x, a)
@dataclass
class StateActionPair(Generic[DimState, DimAction]):
    """A container for a State-Action pair."""

    state: State[DimState]  # x
    action: Action[DimAction]  # a


# (x_{i, t}, a_{i, t})
@dataclass
class Sample(StateActionPair[DimState, DimAction]):
    """A State-Action pair in the context of Demonstrations."""

    index: SampleIndex  # (i, t)


@dataclass
class Samples(Generic[NumPoints, DimState, DimAction]):
    samples: TypedList[NumPoints, Sample[DimState, DimAction]] = field(
        default_factory=TypedList[NumPoints, Sample[DimState, DimAction]]
    )  # [(x_{t}, a_{t})]_{t = 1}^{T}

    def __len__(self) -> NumPoints:
        return self.samples.length  # T

    @enforce_shapes
    def __getitem__(
        self,
        index: TimeIndex,  # t
        /,
    ) -> Sample[DimState, DimAction]:
        return self.samples[index]  # (x_{t}, a_{t})

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        yield from self.samples

    @property
    def time_indices(self) -> TimeIndices[NumPoints]:
        return TimeIndices[NumPoints](range(self.__len__()))

    @enforce_shapes
    def append(self, sample: Sample[DimState, DimAction]) -> None:
        self.samples.append(sample)

    @enforce_shapes
    def states(self) -> States[NumPoints, DimState]:
        return States[NumPoints, DimState](sample.state for sample in self.samples)

    @enforce_shapes
    def actions(self) -> Actions[NumPoints, DimAction]:
        return Actions[NumPoints, DimAction](sample.action for sample in self.samples)

    @property
    def time_indices_and_samples(
        self,
    ) -> Iterator[tuple[TimeIndex, Sample[DimState, DimAction]]]:
        # ) -> TypedList[NumPoints, tuple[TimeIndex, Sample[DimState, DimAction]]]:
        for t in self.time_indices:
            sample = self.samples[t]
            yield (t, sample)
        # return TypedList[NumPoints, tuple[TimeIndex, Sample[DimState, DimAction]]](
        #     (sample.index.time, sample) for sample in self.samples
        # )

    @property
    def time_indices_and_states(
        self,
    ) -> Iterator[tuple[TimeIndex, State[DimState]]]:
        for t, sample in self.time_indices_and_samples:
            yield (t, sample.state)

    @property
    def time_indices_and_actions(
        self,
    ) -> Iterator[tuple[TimeIndex, Action[DimAction]]]:
        for t, sample in self.time_indices_and_samples:
            yield (t, sample.action)


@dataclass
class SamplesCollection(Generic[NumDemos, NumPoints, DimState, DimAction]):
    """A collection of state-action pairs (a sequence of state-action pair)."""

    collection: TypedList[NumDemos, Samples[NumPoints, DimState, DimAction]] = field(
        default_factory=TypedList[NumDemos, Samples[NumPoints, DimState, DimAction]]
    )  # [[(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}]_{i = 1}^{N}

    def __len__(self) -> NumDemos:
        return self.collection.length  # N

    @enforce_shapes
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
                return self.collection[i][t]
            case DemoIndex() as i:
                return self.collection[i]
        raise IndexError

    @enforce_shapes
    def __iter__(self) -> Iterator[Samples[NumPoints, DimState, DimAction]]:
        yield from self.collection

    @property
    def demo_indices(self) -> DemoIndices[NumDemos]:
        return DemoIndices[NumDemos](range(self.__len__()))

    @enforce_shapes
    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Iterator[Sample[DimState, DimAction]]:  # (N x T_) or (N-1 x T_)
        for i, samples in enumerate(self.collection):
            if i == LOO_demo_index:
                continue
            yield from samples

    @enforce_shapes
    def states(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> States[int, DimState]:
        # (N x T_) or (N-1 x T_)
        return States[int, DimState](
            sample.state for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    @enforce_shapes
    def actions(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Actions[int, DimAction]:
        # (N x T_) or (N-1 x T_)
        return Actions[int, DimAction](
            sample.action for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    @property
    def iter_sample_indices_and_samples(
        self,
    ) -> Iterator[tuple[SampleIndex, Sample[DimState, DimAction]]]:
        for i in self.demo_indices:
            samples = self[i]
            for t in samples.time_indices:
                sample_idx = SampleIndex(i, t)
                sample = self[sample_idx]
                yield (sample_idx, sample)

    @property
    def iter_sample_indices_and_states(
        self,
    ) -> Iterator[tuple[SampleIndex, State[DimState]]]:
        for sample_idx, sample in self.iter_sample_indices_and_samples:
            yield (sample_idx, sample.state)

    @property
    def iter_sample_indices_and_actions(
        self,
    ) -> Iterator[tuple[SampleIndex, Action[DimAction]]]:
        for sample_idx, sample in self.iter_sample_indices_and_samples:
            yield (sample_idx, sample.action)


# Behaves like Samples
@dataclass(kw_only=True)
class Demonstration(Generic[NumPoints, DimState, DimAction]):  # D_i
    index: DemoIndex  # i
    states: States[NumPoints, DimState]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: Actions[NumPoints, DimAction]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert self.states.length == self.actions.length

    def __len__(self) -> NumPoints:
        assert self.states.length == self.actions.length
        return self.states.length  # T_i

    @enforce_shapes
    def __getitem__(
        self, index: TimeIndex, /
    ) -> Sample[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return Sample(
            index=SampleIndex(demo=self.index, time=index),
            state=self.states[index],
            action=self.actions[index],
        )

    @enforce_shapes
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
            TypedList[NumPoints, Sample[DimState, DimAction]](sample for sample in self)
        )


# Behaves like SamplesCollection
@dataclass(slots=True)
class Demonstrations(
    Generic[NumDemos, NumPoints, DimState, DimAction]
):  # [D_i]_{i = 1}^{N}
    demos: TypedList[NumDemos, Demonstration[NumPoints, DimState, DimAction]]

    def __len__(self) -> NumDemos:
        return self.demos.length  # N

    @enforce_shapes
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

    @enforce_shapes
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
