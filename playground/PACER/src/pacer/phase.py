"""
Phase alignment
=======
"""
# src/pacer/phase.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import InitVar, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.typings import (
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    Phases,
    PhasesCollection,
)
from pacer.utils import EPS, TORCH_DEVICE, get_torch_device, normalise

## ── Phase Alignment ──────────────────────────────────────────────────────────


class PhaseScorer(nn.Module, RuntimeGeneric[DimState]):
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
class PhaseEstimator(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: InitVar[DeviceLikeType] = field(default=TORCH_DEVICE, kw_only=True)
    ##
    device_: torch.device = field(init=False)
    scorer: PhaseScorer[DimState] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)

    def compute_ranking_loss(self, margin: float = 1.0) -> Tensor:  # L_rank
        loss = torch.tensor(0.0, device=self.device_)
        for demo in self.demonstrations:
            states = Tensor(np.array(demo.states)).float().to(self.device_)
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
        state_dim = self.demonstrations.state_dim
        scorer = PhaseScorer(state_dim=state_dim, hidden_dim=hidden_dim)
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
        phases = PhasesCollection[NumDemos, NumPoints]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(np.array(demo.states)).float().to(self.device_)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                normalised = Phases[NumPoints](normalise(_scores, method="MINMAX"))
                phases.append(normalised)
        return phases


## ─────────────────────────────────────────────────────────────────────────────
