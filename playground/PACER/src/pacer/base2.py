"""
PACER Base2
=======
[TEMP][WIP]: A refactor preferring dense tensors over lists.
"""
# src/pacer/base2.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from dataclasses import dataclass
from typing import Generic, Iterator, Literal, NamedTuple, TypeAlias, TypeVar, overload

import numpy as np
import numpy.linalg as la
import optype.numpy as onp
import torch
from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.helpers import Array1D, Array2D, Array3D, DType

from pacer import console

## ── Typings ──────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

Phase: TypeAlias = float  # tau \in [0, 1]
DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
State: TypeAlias = Array1D[DimState, DType]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction, DType]  # a_{i, t} \in R^{d_a}
States: TypeAlias = Array2D[NumPoints, DimState, DType]
Actions: TypeAlias = Array2D[NumPoints, DimAction, DType]
StatesCollection: TypeAlias = Array3D[NumDemos, NumPoints, DimState, DType]
ActionsCollection: TypeAlias = Array3D[NumDemos, NumPoints, DimAction, DType]


# (i, t)
class SampleIndex(NamedTuple):
    """Represents the index of a `Sample` in the context of a `Demonstration`."""

    demo: DemoIndex  # i
    time: TimeIndex  # t


SampleIndices: TypeAlias = list[SampleIndex]

## ── Utils ────────────────────────────────────────────────────────────────────

SEED = 42
EPS: float = 1e-8


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore  # ty: ignore[unused-ignore-comment]
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def get_torch_device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        torch_device_auto = torch.device("mps")
    elif torch.cuda.is_available():
        torch_device_auto = torch.device("cuda")
    else:
        torch_device_auto = torch.device("cpu")
    console.print(f"Using device: [green]{torch_device_auto}[/]")
    return torch_device_auto


set_seed(SEED)


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec, dtype=npDType)
    match method:
        case "NORM":
            norm = la.norm(vec)
            return vec / (norm + EPS)
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)


## ── Base ─────────────────────────────────────────────────────────────────────


# (x, a)
@dataclass
class Sample(Generic[DimState, DimAction]):
    """A container for a State-Action pair."""

    state: State[DimState]  # x
    action: Action[DimAction]  # a

    @property
    def state_dim(self) -> DimState:
        return self.state.shape[0]

    @property
    def action_dim(self) -> DimAction:
        return self.action.shape[0]


# [(x_{t}, a_{t})]_{t = 1}^{T}
@dataclass
class Samples(Generic[NumPoints, DimState, DimAction]):
    """A collection of `Sample`."""

    states: States[NumPoints, DimState]  # [x_{t}]_{t = 1}^{T}
    actions: Actions[NumPoints, DimAction]  # [a_{t}]_{t = 1}^{T}

    def __len__(self) -> NumPoints:
        assert self.states.shape[0] == self.actions.shape[0]
        return self.states.shape[0]  # T

    @enforce_shapes
    def __getitem__(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return Sample(state=self.states[t], action=self.actions[t])

    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        for t in range(len(self)):
            yield self[TimeIndex(t)]

    @property
    def state_dim(self) -> DimState:
        return self.states.shape[1]

    @property
    def action_dim(self) -> DimAction:
        return self.actions.shape[1]


@dataclass
class SamplesCollection(Generic[NumDemos, NumPoints, DimState, DimAction]):
    """A collection of `Samples`."""

    # [x_{t}]_{t = 1}^{T}
    states_collection: StatesCollection[NumDemos, NumPoints, DimState]
    # [a_{t}]_{t = 1}^{T}
    actions_collection: ActionsCollection[NumDemos, NumPoints, DimAction]

    def __len__(self) -> NumDemos:
        assert self.states_collection.shape[0] == self.actions_collection.shape[0]
        return self.states_collection.shape[0]  # N

    @enforce_shapes
    @overload
    def __getitem__(
        self,
        index: DemoIndex,  # i
        /,
    ) -> Samples[NumPoints, DimState, DimAction]: ...
    @overload
    def __getitem__(
        self,
        index: SampleIndex,  # (i, t)
        /,
    ) -> Sample[DimState, DimAction]: ...
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Samples[NumPoints, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case DemoIndex() as i:
                return Samples(
                    states=States[NumPoints, DimState](self.states_collection[i]),
                    actions=Actions[NumPoints, DimAction](self.actions_collection[i]),
                )
            case SampleIndex(i, t):
                return Sample(
                    state=State[DimState](self.states_collection[i][t]),
                    action=Action[DimAction](self.actions_collection[i][t]),
                )

    @enforce_shapes
    def __iter__(self) -> Iterator[Samples[NumPoints, DimState, DimAction]]:
        for states, actions in zip(self.states_collection, self.actions_collection):
            yield Samples(states=states, actions=actions)

    @property
    def samples(self) -> Iterator[Sample[DimState, DimAction]]:
        for i in range(len(self)):
            for t in range(len(self[i])):
                yield self[SampleIndex(i, t)]

    @property
    def state_dim(self) -> DimState:
        return self[SampleIndex(0, 0)].state_dim

    @property
    def action_dim(self) -> DimAction:
        return self[SampleIndex(0, 0)].action_dim


## ─────────────────────────────────────────────────────────────────────────────
