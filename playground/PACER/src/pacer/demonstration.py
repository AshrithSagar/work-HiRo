"""
Demonstration
"""
# src/pacer/demonstration.py

from dataclasses import dataclass
from typing import Generic, Iterator, TypeAlias, TypeVar

import numpy as np
import torch.nn as nn
from torch import Tensor
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.context import enforce_shapes  # type: ignore

type DType = np.dtype[np.float32]
DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a

State: TypeAlias = TypedNDArray[tuple[DimState], DType]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = TypedNDArray[tuple[DimAction], DType]  # a_{i, t} \in R^{d_a}
type Phase = float  # tau \in [0, 1]
type BinIndex = int  # b \in {0, 1, ..., B-1}
type SampleIndex = tuple[int, int]  # (i, t)

# (x_{i, t}, a_{i, t})
StateActionPair: TypeAlias = tuple[State[DimState], Action[DimAction]]
Sample: TypeAlias = StateActionPair[DimState, DimAction]


@dataclass(kw_only=True)
class Demonstration(Generic[DimState, DimAction]):  # D_i
    states: list[State[DimState]]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: list[Action[DimAction]]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions)
        self.n_pairs = len(self.states)  # T_i

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


Demonstrations: TypeAlias = list[Demonstration]  # [D_i]_{i = 1}^{N}


class PhaseEstimator(nn.Module):
    """Neural network to estimate state-dependent phase score `g_psi`."""

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
    state_tangent: State[DimState]  # t_s[b] <- diff{ alpha_s[b] }


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
    state_tangent: State[DimState]  # t_s[b]
    MAD_action: float  # Median Absolute Deviation of actions
