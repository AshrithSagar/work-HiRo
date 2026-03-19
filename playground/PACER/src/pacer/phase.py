"""
Phase alignment
=======
"""
# src/pacer/phase.py

## ── Imports ──────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Self, TypeAlias

import numpy as np
import optype.numpy as onp
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric, TypedDict, TypedList

from pacer.base import Demonstration, Demonstrations
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    Vector,
    npDType,
)
from pacer.utils import EPS, SEED, TORCH_DEVICE, get_torch_device, normalise, set_seed

## ── Phase Alignment ──────────────────────────────────────────────────────────

Phase: TypeAlias = npDType  # tau \in [0, 1]


class Phases(TypedList[NumPoints, Phase]):
    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.time_indices.length
        return cls.full(T_i, Phase(0))


class PhasesCollection(TypedDict[NumDemos, DemoIndex, Phases[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demonstrations.demo_indices,
            lambda i: Phases[NumPoints].zeros_like(demonstrations[i]),
        )


# ──────────────────────────────────────────────────────────────────────────────


class PhaseEstimator(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC):
    """Abstract interface to estimate phases for a set of demonstrations."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]

    # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
    @abstractmethod
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        raise NotImplementedError


# ── MLP Phase Scorer ──────────────────────────────────────────────────────────


class MLPPhaseScorer(nn.Module, RuntimeGeneric[DimState]):
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
class MLPPhaseEstimator(PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: InitVar[DeviceLikeType] = field(default=TORCH_DEVICE, kw_only=True)
    seed: int = field(default=SEED, kw_only=True)
    ##
    device_: torch.device = field(init=False)
    scorer: MLPPhaseScorer[DimState] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)

    def compute_ranking_loss(self, margin: float = 1.0) -> Tensor:  # L_rank
        loss = torch.tensor(0.0, device=self.device_)
        for demo in self.demonstrations:
            states = Tensor(demo.states.numpy()).float().to(self.device_)
            scores: Tensor = self.scorer(states)  # (T_i,)
            diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (T_i, T_i)
            mask = torch.ones_like(diff).triu(diagonal=1)  # Enforces `t > t'`
            loss_matrix = F.softplus(margin - diff) * mask
            loss += loss_matrix.sum() / (mask.sum() + EPS)  # Normalise over valid pairs
        if (n_demos := self.demonstrations.__len__()) > 0:
            loss /= n_demos  # Normalise over demonstrations
        return loss

    def train(
        self,
        *,
        hidden_dim: int = 128,
        margin: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 240,
    ) -> Tensor:
        set_seed(self.seed)
        state_dim = self.demonstrations.state_dim
        scorer = MLPPhaseScorer(state_dim=state_dim, hidden_dim=hidden_dim)
        self.scorer = scorer.to(self.device_)
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

    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.scorer.eval()
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(demo.states.numpy()).float().to(self.device_)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                taus = Phases[NumPoints](normalise(_scores, method="MINMAX"))
                phases[demo.index] = taus
        return phases


# ── Normalised Time Index Phase Estimation ────────────────────────────────────


# tau_{i, t} = t / (T_i - 1)
@dataclass
class NormalisedTimeIndexPhaseEstimator(
    PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]

    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        for demo in self.demonstrations:
            T_i = demo.states.length
            assert T_i > 1
            taus = Phases[NumPoints]([Phase(t / (T_i - 1)) for t in range(T_i)])
            phases[demo.index] = taus
        return phases


# ── Path Length Phase Estimation ──────────────────────────────────────────────


@dataclass
class PathLengthPhaseEstimator(
    PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]

    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        for demo in self.demonstrations:
            diffs: onp.Array2D[npDType] = np.diff(demo.states, axis=0)  # (T_i-1, d_x)
            norms: onp.Array1D[npDType] = np.linalg.norm(diffs, axis=1)  # (T_i-1,)
            lengths = Vector[NumPoints](np.r_[0, np.cumsum(norms)])
            taus = Phases[NumPoints](lengths / (lengths[-1] + EPS))
            phases[demo.index] = taus
        return phases


## ─────────────────────────────────────────────────────────────────────────────
