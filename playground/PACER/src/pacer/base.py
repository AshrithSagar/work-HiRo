"""
PACER Base
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/base.py

import math
from dataclasses import dataclass, field
from typing import (
    Generic,
    Iterator,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch
import torch.nn as nn
from torch import Tensor
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.context import enforce_shapes  # type: ignore

DType: TypeAlias = np.float32
Dim1 = TypeVar("Dim1", bound=int, default=int)
Array1D: TypeAlias = TypedNDArray[tuple[Dim1], np.dtype[DType]]

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
State: TypeAlias = Array1D[DimState]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction]  # a_{i, t} \in R^{d_a}
type Phase = float  # tau \in [0, 1]
type DemoIndex = int  # i \i {0, 1, ..., N-1}
type TimeIndex = int  # t \i {0, 1, ..., T_i-1}
type BinIndex = int  # b \in {0, 1, ..., B-1}
type SampleIndex = tuple[int, int]  # (i, t)

# (x_{i, t}, a_{i, t})
StateActionPair: TypeAlias = tuple[State[DimState], Action[DimAction]]
Sample: TypeAlias = StateActionPair[DimState, DimAction]


## Utils

EPS: float = 1e-8
MAD_SCALE: float = 1.4826  # Gaussian consistency factor for MAD


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return np.median(arr, axis=axis)


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec, dtype=DType)
    match method:
        case "NORM":
            norm = la.norm(vec)
            return vec / (norm + EPS)
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)


@dataclass(kw_only=True)
class Demonstration(Generic[DimState, DimAction]):  # D_i
    index: DemoIndex  # i
    states: list[State[DimState]]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: list[Action[DimAction]]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions)
        self.n_pairs = len(self.states)  # T_i

    def __len__(self) -> int:
        return self.n_pairs

    @enforce_shapes
    def __getitem__(
        self, t: int, /
    ) -> StateActionPair[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return (self.states[t], self.actions[t])

    @enforce_shapes
    def __iter__(self) -> Iterator[StateActionPair[DimState, DimAction]]:
        # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
        for t in range(self.n_pairs):
            yield self[t]

    @property
    def state_dim(self) -> DimState:  # d_x
        return cast(DimState, self.states[0].shape[0])

    @property
    def action_dim(self) -> DimAction:  # d_a
        return cast(DimAction, self.actions[0].shape[0])

    def sample(self, t: int, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})


@dataclass(slots=True)
class Demonstrations(Generic[DimState, DimAction]):  # [D_i]_{i = 1}^{N}
    demos: list[Demonstration[DimState, DimAction]]

    def __len__(self) -> int:
        return len(self.demos)

    @enforce_shapes
    @overload
    def __getitem__(self, index: DemoIndex) -> Demonstration[DimState, DimAction]: ...
    @overload
    def __getitem__(self, index: SampleIndex) -> Sample[DimState, DimAction]: ...
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex
    ) -> Demonstration[DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case tuple():
                i, t = index
                return self.demos[i][t]
            case int():
                return self.demos[index]

    @enforce_shapes
    def __iter__(self) -> Iterator[Demonstration[DimState, DimAction]]:
        for demo in self.demos:
            yield demo  # D_i

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.demos[0].state_dim

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.demos[0].action_dim


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


class PhaseEstimator(Generic[DimState, DimAction]):
    def __init__(
        self,
        demonstrations: Demonstrations[DimState, DimAction],
        *,
        margin: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 240,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.demonstrations = demonstrations
        self.margin = margin
        self.lr = lr
        self.epochs = epochs
        self.device = device

        state_dim = self.demonstrations.state_dim
        self.scorer = PhaseScorer(state_dim=state_dim).to(self.device)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=self.lr)

    def compute_ranking_loss(self) -> Tensor:  # L_rank
        ranking_loss = torch.tensor(0.0, device=self.device)
        for demo in self.demonstrations:
            states = torch.from_numpy(demo.states)  # type: ignore
            states = states.float().to(self.device)
            scores: Tensor = self.scorer(states)
            for t in range(len(scores)):
                for t_prime in range(t):
                    score_diff = scores[t] - scores[t_prime]
                    loss = torch.log(1 + torch.exp(self.margin - score_diff))
                    ranking_loss += loss
        return ranking_loss

    def train(self) -> None:
        for _epoch in range(self.epochs):
            self.optimiser.zero_grad()
            loss = self.compute_ranking_loss()
            loss.backward()  # type: ignore
            self.optimiser.step()  # type: ignore

    @enforce_shapes
    def estimate_phases(
        self,
    ) -> list[Array1D[int]]:  # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.scorer.eval()
        phases = list[Array1D[int]]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = torch.from_numpy(demo.states)  # type: ignore
                states = states.float().to(self.device)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                normalised = Array1D(normalise(_scores, method="MINMAX"))
                phases.append(normalised)
        return phases


@dataclass(kw_only=True)
class BinStats(Generic[DimState, DimAction]):
    """Robust bin-level consensus statistics for a single phase bin `b`."""

    ## Stable anchors
    median_action: Action[DimAction]  # alpha_a[b] = median{ a_{i, t} : (i, t) \in I_b }
    median_state: State[DimState]  # alpha_s[b] = median{ x_{i, t} : (i, t) \in I_b }

    ## Pace
    median_action_strength: float
    # beta_a[b] = median{ ||a_{i, t}|| : (i, t) \in I_b }
    # Captures strength of actions
    median_state_change: float
    # beta_s[b] = median{ ||xdot_{i, t}|| : (i, t) \in I_b }
    # Captures typical rate of state change

    ## Local task dynamics
    action_tangent: Action[DimAction]  # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = None  # t_s[b] <- diff{ alpha_s[b] }


@dataclass(kw_only=True)
class RibbonToken(Generic[DimState, DimAction]):
    """
    Robust structured descriptor for a single phase bin `b`.\\
    Encodes both consensus behaviour and degree of variability present at phase `b`.
    """

    median_action: Action[DimAction]  # alpha_a[b]
    median_action_strength: float  # beta_a[b]
    median_state: State[DimState]  # alpha_s[b]
    median_state_change: float  # beta_s[b]
    action_tangent: Action[DimAction]  # t_a[b]
    state_tangent: State[DimState] | None = None  # t_s[b]
    MAD_action: float  # Median Absolute Deviation of actions


@dataclass(kw_only=True)
class Bin(Generic[DimState, DimAction]):
    index: BinIndex  # b
    sample_indices: list[SampleIndex] = field(default_factory=list[SampleIndex])  # I_b
    samples: list[Sample[DimState, DimAction]] = field(
        default_factory=list[Sample[DimState, DimAction]]
    )
    robust_statistics: BinStats[DimState, DimAction] | None = None

    @property
    def states(self) -> list[State[DimState]]:
        return list(state for state, _ in self.samples)

    @property
    def actions(self) -> list[Action[DimAction]]:
        return list(action for _, action in self.samples)


class BinHandler(Generic[DimState, DimAction]):
    def __init__(
        self, phase_estimator: PhaseEstimator[DimState, DimAction], *, n_bins: int = 96
    ) -> None:
        self.phase_estimator = phase_estimator
        self.n_bins = n_bins  # B
        self.demonstrations = self.phase_estimator.demonstrations

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (bin_idx / self.n_bins, (bin_idx + 1) / self.n_bins)

    def make_bins(self) -> None:
        phases = self.phase_estimator.estimate_phases()
        self.bins = [
            Bin[DimState, DimAction](index=bin_idx) for bin_idx in range(self.n_bins)
        ]
        for i in range(len(phases)):
            for t in range(len(phases[i])):
                tau: Phase = phases[i][t]
                bin_idx: BinIndex = math.floor(tau * self.n_bins)
                assert bin_idx < self.n_bins
                bin = self.bins[bin_idx]
                sample_idx: SampleIndex = (i, t)
                bin.sample_indices.append(sample_idx)
                sample = self.demonstrations[sample_idx]
                bin.samples.append(sample)

    def compute_robust_consensus_statistics(
        self, samples: list[Sample[DimState, DimAction]]
    ) -> BinStats[DimState, DimAction]:
        states = list(state for state, _ in samples)
        actions = list(action for _, action in samples)

        action_norms = list(la.norm(action) for action in actions)
        state_change_norms = list(la.norm(np.diff(state)) for state in states)

        median_action = Action[DimAction](median(actions, axis=0))
        median_state = State[DimState](median(states, axis=0))
        median_action_strength = float(median(action_norms, axis=0))
        median_state_change = float(median(state_change_norms, axis=0))
        action_tangent = np.diff(median_action, axis=0)
        state_tangent = np.diff(median_state, axis=0)

        return BinStats(
            median_action=median_action,
            median_state=median_state,
            median_action_strength=median_action_strength,
            median_state_change=median_state_change,
            action_tangent=action_tangent,
            state_tangent=state_tangent,
        )

    def LOO_sample_indices(
        self,
        bin_idx: BinIndex,  # b
        demo_idx: int,  # i
    ) -> list[SampleIndex]:  # I_b^{(-i)}
        bin = self.bins[bin_idx]
        sample_indices = list[SampleIndex]()
        for sample_idx in bin.sample_indices:
            i, _t = sample_idx
            if i == demo_idx:
                continue
            sample_indices.append(sample_idx)
        return sample_indices

    def LOO_samples(
        self,
        bin_idx: BinIndex,  # b
        demo_idx: int,  # i
    ) -> list[Sample[DimState, DimAction]]:
        bin = self.bins[bin_idx]
        sample_indices = self.LOO_sample_indices(bin_idx, demo_idx)
        demo_indices = list(i for i, _t in sample_indices)
        samples = list(
            sample for i, sample in enumerate(bin.samples) if i not in demo_indices
        )
        return samples

    def compute_trust_values(self) -> None:
        for bin in self.bins:
            stats = self.compute_robust_consensus_statistics(bin.samples)
            bin.robust_statistics = stats
            for demo in self.demonstrations:
                loo_samples = self.LOO_samples(bin.index, demo.index)
                loo_stats = self.compute_robust_consensus_statistics(loo_samples)
                bin_median_action = loo_stats.median_action  # alpha_a^{(-i)}[b]
                for _, action in demo:
                    _action_residual = la.norm(action - bin_median_action)  # r_{i, t}


class PACER(Generic[DimState, DimAction]):
    def __init__(self, demonstrations: Demonstrations[DimState, DimAction]) -> None:
        self.demonstrations = demonstrations
