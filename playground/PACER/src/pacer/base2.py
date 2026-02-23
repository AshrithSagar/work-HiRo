"""
PACER Base2
=======
[TEMP][WIP]: A refactor preferring 2D numpy arrays over list of 1D numpy arrays.
"""
# src/pacer/base2.py

from dataclasses import dataclass, field
from typing import (
    Generic,
    Iterable,
    Iterator,
    NamedTuple,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from rich.console import Console
from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.helpers import Array1D, Array2D, DType

console = Console()

Phase: TypeAlias = float  # tau \in [0, 1]
DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
State: TypeAlias = Array1D[DimState, DType]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction, DType]  # a_{i, t} \in R^{d_a}
States: TypeAlias = Array2D[NumPoints, DimState, DType]
Actions: TypeAlias = Array2D[NumPoints, DimAction, DType]


# (i, t)
class SampleIndex(NamedTuple):
    demo: DemoIndex  # i
    time: TimeIndex  # t


SampleIndices: TypeAlias = list[SampleIndex]


# (x_{i, t}, a_{i, t})
@dataclass
class StateActionPair(Generic[DimState, DimAction]):
    """A container for a State-Action pair"""

    state: State[DimState]  # x_{i, t}
    action: Action[DimAction]  # a_{i, t}

    @property
    def state_dim(self) -> DimState:
        return self.state.shape[0]

    @property
    def action_dim(self) -> DimAction:
        return self.action.shape[0]


Sample: TypeAlias = StateActionPair[DimState, DimAction]


# [(x_{t}, a_{t})]_{t = 1}^{T}
@dataclass
class StateActionPairs(Generic[NumPoints, DimState, DimAction]):
    states: States[NumPoints, DimState]  # [x_{t}]_{t = 1}^{T}
    actions: Actions[NumPoints, DimAction]  # [a_{t}]_{t = 1}^{T}

    def __len__(self) -> NumPoints:
        assert self.states.shape[0] == self.actions.shape[0]
        return self.states.shape[0]  # T

    @enforce_shapes
    def __getitem__(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return StateActionPair(state=self.states[t], action=self.actions[t])

    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        for t in range(len(self)):
            yield self[TimeIndex(t)]

    @property
    def state_dim(self) -> DimState:
        return self.states.shape[1]

    @property
    def action_dim(self) -> DimAction:
        return self.actions.shape[1]


Samples: TypeAlias = StateActionPairs[NumPoints, DimState, DimAction]


@dataclass(kw_only=True)
class Demonstration(Samples[NumPoints, DimState, DimAction]):
    index: DemoIndex


@dataclass
class SamplesCollection(Generic[DimState, DimAction]):
    """A collection of state-action pairs (a sequence of state-action pair)."""

    collection: list[Samples[int, DimState, DimAction]] = field(
        default_factory=list[Samples[int, DimState, DimAction]]
    )  # [[(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}]_{i = 1}^{N}

    def __len__(self) -> int:
        return len(self.collection)  # N

    @enforce_shapes
    @overload
    def __getitem__(
        self,
        index: DemoIndex,  # i
        /,
    ) -> Samples[int, DimState, DimAction]: ...
    @overload
    def __getitem__(
        self,
        index: SampleIndex,  # (i, t)
        /,
    ) -> Sample[DimState, DimAction]: ...
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Samples[int, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case DemoIndex() as i:
                return self.collection[i]
            case SampleIndex(i, t):
                return self.collection[i][t]

    @enforce_shapes
    def __iter__(self) -> Iterator[Samples[int, DimState, DimAction]]:
        for samples in self.collection:
            yield samples

    @enforce_shapes
    def append(self, samples: Samples[int, DimState, DimAction]) -> None:
        self.collection.append(samples)

    @enforce_shapes
    def extend(self, samples: Iterable[Samples[int, DimState, DimAction]]) -> None:
        self.collection.extend(samples)


@dataclass
class Demonstrations(Generic[DimState, DimAction]):
    demos: list[Demonstration[int, DimState, DimAction]]

    def __len__(self) -> NumPoints:  # type: ignore[misc, type-var]
        return cast(NumPoints, len(self.demos))

    @enforce_shapes
    @overload
    def __getitem__(
        self, index: DemoIndex, /
    ) -> Demonstration[int, DimState, DimAction]: ...
    @overload
    def __getitem__(self, index: SampleIndex, /) -> Sample[DimState, DimAction]: ...
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Demonstration[int, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case DemoIndex() as i:
                return self.demos[i]
            case SampleIndex(i, t):
                return self.demos[i][t]

    def __iter__(self) -> Iterator[Demonstration[int, DimState, DimAction]]:
        for i in range(len(self)):
            yield self[DemoIndex(i)]

    @property
    def samples(self) -> Iterator[Sample[DimState, DimAction]]:
        for i in range(len(self)):
            for t in range(len(self[i])):
                yield self[SampleIndex(i, t)]
