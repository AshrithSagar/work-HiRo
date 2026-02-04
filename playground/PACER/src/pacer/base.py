"""
PACER Base
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/base.py

from dataclasses import dataclass
from typing import Generic, Iterator, Literal, Sequence, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch import Tensor
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.context import enforce_shapes  # type: ignore

type DType = np.dtype[np.float32]
N = TypeVar("N", bound=int, default=int)
Array1D: TypeAlias = TypedNDArray[tuple[N], DType]

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
State: TypeAlias = Array1D[DimState]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction]  # a_{i, t} \in R^{d_a}
type Phase = float  # tau \in [0, 1]
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
    vec: npt.ArrayLike, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec)
    match method:
        case "NORM":
            norm = np.linalg.norm(vec)
            return vec / (norm + EPS)
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)
        case _:
            raise NotImplementedError


@dataclass(kw_only=True)
class Demonstration(Generic[DimState, DimAction]):  # D_i
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
        return self.states[0].shape[0]

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.actions[0].shape[0]

    def sample(self, t: int, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})


# [D_i]_{i = 1}^{N}
Demonstrations: TypeAlias = list[Demonstration[DimState, DimAction]]


class PhaseScorer(nn.Module):
    """A small neural network (MLP) to estimate state-dependent phase score `g_psi`."""

    def __init__(self, state_dim: int, hidden_dim: int = 64):
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


class PhaseEstimator(Generic[DimState]):
    def __init__(
        self,
        demonstrations: Demonstrations[DimState],
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

        state_dim = self.demonstrations[0].state_dim
        self.scorer = PhaseScorer(state_dim=state_dim).to(self.device)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=self.lr)

    def compute_ranking_loss(self) -> Tensor:
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
    def estimate_phases(self) -> list[Array1D[int]]:
        self.scorer.eval()
        phases = list[Array1D[int]]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = torch.from_numpy(demo.states)  # type: ignore
                states = states.float().to(self.device)
                scores: Tensor = self.scorer(states)
                _scores = State[DimState](scores.cpu().numpy())
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
