"""
PACER Base2
=======
[TEMP][WIP]: A refactor preferring dense tensors over lists.
"""
# src/pacer/base2.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, NamedTuple, Self, TypeAlias, TypeVar, overload

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.dimexpr import MinusOne, Mul
from typed_numpy._typed.helpers import Array0D, Array1D, Array2D, Array3D, DType

from pacer import console

## ── Typings ──────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N

DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

State: TypeAlias = Array1D[DimState, DType]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction, DType]  # a_{i, t} \in R^{d_a}
States: TypeAlias = Array2D[NumPoints, DimState, DType]
Actions: TypeAlias = Array2D[NumPoints, DimAction, DType]
StatesCollection: TypeAlias = Array3D[NumDemos, NumPoints, DimState, DType]
ActionsCollection: TypeAlias = Array3D[NumDemos, NumPoints, DimAction, DType]

Phase: TypeAlias = Array0D[np.dtype[npDType]]  # tau \in [0, 1]
Phases: TypeAlias = Array1D[NumPoints, np.dtype[npDType]]
PhasesCollection: TypeAlias = Array2D[NumDemos, NumPoints, np.dtype[npDType]]


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
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
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


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return np.median(arr, axis=axis)


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

    @enforce_shapes
    @classmethod
    def from_sample(cls, samples: Sequence[Sample[DimState, DimAction]]) -> Self:
        states = States[NumPoints, DimState]([sample.state for sample in samples])
        actions = Actions[NumPoints, DimAction]([sample.action for sample in samples])
        return cls(states=states, actions=actions)

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

    @enforce_shapes
    @classmethod
    def from_samples(
        cls, samples_collection: Sequence[Samples[NumPoints, DimState, DimAction]]
    ) -> Self:
        return cls(
            states_collection=StatesCollection[NumDemos, NumPoints, DimState](
                [samples.states for samples in samples_collection]
            ),
            actions_collection=ActionsCollection[NumDemos, NumPoints, DimAction](
                [samples.actions for samples in samples_collection]
            ),
        )

    def __len__(self) -> NumDemos:
        assert self.states_collection.shape[0] == self.actions_collection.shape[0]
        return self.states_collection.shape[0]  # N

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
    @enforce_shapes
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
    def state_dim(self) -> DimState:
        return self[SampleIndex(0, 0)].state_dim

    @property
    def action_dim(self) -> DimAction:
        return self[SampleIndex(0, 0)].action_dim

    @property
    @enforce_shapes
    def samples(self) -> Samples[Mul[NumDemos, NumPoints], DimState, DimAction]:
        # (N x T_)
        return Samples[Mul[NumDemos, NumPoints], DimState, DimAction](
            states=States[Mul[NumDemos, NumPoints], DimState](
                np.vstack(self.states_collection)  # type: ignore[call-overload]
            ),
            actions=Actions[Mul[NumDemos, NumPoints], DimAction](
                np.vstack(self.actions_collection)  # type: ignore[call-overload]
            ),
        )

    @property
    @enforce_shapes
    def states(self) -> States[Mul[NumDemos, NumPoints], DimState]:
        return self.samples.states  # (N x T_)

    @property
    @enforce_shapes
    def actions(self) -> Actions[Mul[NumDemos, NumPoints], DimAction]:
        return self.samples.actions  # (N x T_)

    @enforce_shapes
    def LOO(
        self, index: DemoIndex
    ) -> SamplesCollection[MinusOne[NumDemos], NumPoints, DimState, DimAction]:
        return SamplesCollection[MinusOne[NumDemos], NumPoints, DimState, DimAction](
            states_collection=StatesCollection[MinusOne[NumDemos], NumPoints, DimState](
                np.delete(self.states_collection, index, axis=0)
            ),
            actions_collection=ActionsCollection[
                MinusOne[NumDemos], NumPoints, DimAction
            ](np.delete(self.actions_collection, index, axis=0)),
        )


@dataclass
class Demonstrations(SamplesCollection[NumDemos, NumPoints, DimState, DimAction]):
    # [FIXME]: Handle padding, and masks
    def __init__(
        self,
        states_collection: onp.ToArrayStrict3D,
        actions_collection: onp.ToArrayStrict3D,
    ) -> None:
        super().__init__(
            states_collection=StatesCollection[NumDemos, NumPoints, DimState](
                states_collection
            ),
            actions_collection=ActionsCollection[NumDemos, NumPoints, DimAction](
                actions_collection
            ),
        )


## ── Phase Alignment ──────────────────────────────────────────────────────────


class PhaseScorer(nn.Module, Generic[DimState]):
    """A small neural network (MLP) to estimate state-dependent phase score `g_psi`."""

    def __init__(self, state_dim: DimState, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,) unnormalized phase scores
        forward: Tensor = self.network(states)
        return forward.squeeze(-1)


@dataclass
class PhaseEstimator(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: torch.device = field(kw_only=True, default_factory=get_torch_device_auto)
    ##
    scorer: PhaseScorer[DimState] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def compute_ranking_loss(self, margin: float = 1.0) -> Tensor:  # L_rank
        loss = torch.tensor(0.0, device=self.device)
        for demo in self.demonstrations:
            states = Tensor(np.array(demo.states)).float().to(self.device)
            scores: Tensor = self.scorer(states)  # (T_i,)
            diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (T_i, T_i)
            mask = torch.ones_like(diff).triu(diagonal=1)  # Enforces `t > t'`
            loss_matrix = F.softplus(margin - diff) * mask
            loss += loss_matrix.mean()
        return loss

    def train(
        self,
        *,
        hidden_dim: int = 128,
        margin: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 240,
    ) -> Tensor:
        state_dim = self.demonstrations.state_dim
        scorer = PhaseScorer(state_dim=state_dim, hidden_dim=hidden_dim)
        self.scorer = scorer.to(self.device)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=lr)

        self.scorer.train()
        loss = self.compute_ranking_loss(margin=margin)
        for _epoch in track(range(epochs), description="[bold]Phase training[/]"):
            self.optimiser.zero_grad()
            loss = self.compute_ranking_loss(margin=margin)
            loss.backward()  # type: ignore[no-untyped-call]  # ty: ignore[unused-ignore-comment]
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    @enforce_shapes
    def estimate_phases(
        self,
    ) -> list[Array1D[int]]:  # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.scorer.eval()
        phases = list[Array1D[int]]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(np.array(demo.states)).float().to(self.device)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                normalised = Array1D(normalise(_scores, method="MINMAX"))
                phases.append(normalised)
        return phases


## ── PACER ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RobustStatistics(Generic[DimState, DimAction]):
    """Robust consensus statistics for a set of samples."""

    ## Stable anchors
    median_action: Action[DimAction]  # alpha_a[b] = median{ a_{i, t} : (i, t) \in I_b }
    median_state: State[DimState]  # alpha_s[b] = median{ x_{i, t} : (i, t) \in I_b }

    ## Pace
    median_action_strength: npDType
    # beta_a[b] = median{ ||a_{i, t}|| : (i, t) \in I_b }
    # Captures strength of actions
    median_state_change: npDType
    # beta_s[b] = median{ ||xdot_{i, t}|| : (i, t) \in I_b }
    # Captures typical rate of state change

    ## Local task dynamics
    # NOTE: `action_tangent` and `state_tangent` are not stored here,
    # but instead, computed and stored in RibbonToken.


@dataclass(kw_only=True)
class RibbonToken(Generic[DimState, DimAction]):  # z_b
    """
    Robust structured descriptor for a single phase bin `b`.\\
    Encodes both consensus behaviour and degree of variability present at phase `b`.
    """

    median_action: Action[DimAction]  # alpha_a[b]
    median_action_strength: npDType  # beta_a[b]
    median_state: State[DimState]  # alpha_s[b]
    median_state_change: npDType  # beta_s[b]

    ## Local task dynamics
    action_tangent: Action[DimAction] = field(init=False)
    # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = field(init=False)
    # t_s[b] <- diff{ alpha_s[b] }

    MAD_action_residual: npDType  # Median Absolute Deviation of action residuals


@dataclass(kw_only=True)
class Bin(Generic[NumDemos, NumPoints, DimState, DimAction]):
    index: BinIndex  # b
    sample_indices: SampleIndices  # I_b
    samples_collection: SamplesCollection[NumDemos, NumPoints, DimState, DimAction]
    ribbon_token: RibbonToken[DimState, DimAction] = field(init=False)

    @property
    @enforce_shapes
    def samples(self) -> Samples[Mul[NumDemos, NumPoints], DimState, DimAction]:
        return self.samples_collection.samples  # (N x T_)

    @property
    @enforce_shapes
    def states(self) -> States[Mul[NumDemos, NumPoints], DimState]:
        return self.samples_collection.states  # (N x T_)

    @property
    @enforce_shapes
    def actions(self) -> Actions[Mul[NumDemos, NumPoints], DimAction]:
        return self.samples_collection.actions  # (N x T_)


@dataclass
class PACER(Generic[NumDemos, NumPoints, DimState, DimAction]):
    phase_estimator: PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
    n_bins: int = field(default=96, kw_only=True)  # B
    bins: list[Bin[NumDemos, NumPoints, DimState, DimAction]] = field(init=False)

    @property
    def demonstrations(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        return self.phase_estimator.demonstrations

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (Phase(bin_idx / self.n_bins), Phase((bin_idx + 1) / self.n_bins))


## ─────────────────────────────────────────────────────────────────────────────
