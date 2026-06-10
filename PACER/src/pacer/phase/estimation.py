"""
Phase estimation
=======
"""
# src/pacer/phase/estimation.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import override

import numpy as np
import optype.numpy as onp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dtaidistance import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    dtw_ndim,
)
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.phase.base import Phase, Phases, PhasesCollection
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


@dataclass
class PhaseEstimator(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC):
    """Abstract interface to estimate phases for a set of demonstrations."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ##
    phases: PhasesCollection[NumDemos, NumPoints] = field(init=False)

    # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
    @abstractmethod
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        raise NotImplementedError


# ── MLP Phase Scorer ──────────────────────────────────────────────────────────


class MLPPhaseScorer(nn.Module, RuntimeGeneric[DimState]):
    """A small neural network (MLP) to estimate state-dependent phase score `g_psi`."""

    def __init__(self, state_dim: DimState, hidden_dim: int = 64):
        super().__init__()
        self.network: nn.Module = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @override
    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,) unnormalized phase scores
        forward: Tensor = self.network(states)
        return forward.squeeze(-1)


@dataclass(kw_only=True)
class MLPPhaseEstimatorConfig:
    hidden_dim: int = 128
    margin: float = 1.0  # m
    lr: float = 1e-3
    epochs: int = 240


@dataclass
class MLPPhaseEstimator(PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]):
    r"""
    MLPPhaseScorer + Ranking loss.

    The model assigns higher scores to later time steps within each demonstration.
    Predicted scores are min-max normalised to obtain phase values `tau \in [0, 1]`.
    """

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
        if (n_demos := self.demonstrations.count) > 0:
            loss /= n_demos  # Normalise over demonstrations
        return loss

    def train(self, config: MLPPhaseEstimatorConfig) -> Tensor:
        set_seed(self.seed)
        state_dim = self.demonstrations.state_dim
        scorer = MLPPhaseScorer(state_dim=state_dim, hidden_dim=config.hidden_dim)
        self.scorer = scorer.to(self.device_)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=config.lr)

        self.scorer.train()
        loss = self.compute_ranking_loss(margin=config.margin)
        for _epoch in track(
            range(config.epochs), description="[bold]Phase training[/]"
        ):
            self.optimiser.zero_grad()
            loss = self.compute_ranking_loss(margin=config.margin)
            loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    @override
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        if not hasattr(self, "scorer"):
            raise RuntimeError("Model must be trained before estimating phases.")
        self.scorer.eval()
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(demo.states.numpy()).float().to(self.device_)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                taus = Phases[NumPoints](normalise(_scores, method="MINMAX"))
                phases[demo.index] = taus
        self.phases: PhasesCollection[NumDemos, NumPoints] = phases
        return phases


# ── Normalised Time Index Phase Estimation ────────────────────────────────────


@dataclass
class NormalisedTimeIndexPhaseEstimator(
    PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
):
    r"""
    Phase is time index normalised over demonstration length,
    for every demonstration, while ensuring `tau \in [0, 1]`.

    `tau_{i, t} = t / (T_i - 1)`.
    """

    @override
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        for demo in self.demonstrations:
            T_i = demo.length
            assert T_i > 1
            taus = Phases[NumPoints]([Phase(t / (T_i - 1)) for t in demo.time_indices])
            phases[demo.index] = taus
        self.phases: PhasesCollection[NumDemos, NumPoints] = phases
        return phases


# ── Path Length Phase Estimation ──────────────────────────────────────────────


@dataclass
class PathLengthPhaseEstimator(
    PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
):
    """
    Assigns phase based on cumulative path length.

    Computes trajectory progress using distance traveled and normalises it to the range [0, 1].
    """

    @override
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        for demo in self.demonstrations:
            diffs: onp.Array2D[npDType] = np.diff(demo.states, axis=0)  # (T_i-1, d_x)
            norms: onp.Array1D[npDType] = np.linalg.norm(diffs, axis=1)  # (T_i-1,)
            lengths = Vector[NumPoints](np.r_[0, np.cumsum(norms)])
            taus = Phases[NumPoints](lengths / (lengths[-1] + EPS))
            phases[demo.index] = taus
        self.phases: PhasesCollection[NumDemos, NumPoints] = phases
        return phases


# ── DTW Phase Estimation ──────────────────────────────────────────────────────


@dataclass(kw_only=True)
class DTWPhaseEstimatorConfig:
    reference_demo_index: DemoIndex = 0


@dataclass
class DTWPhaseEstimator(PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]):
    config: DTWPhaseEstimatorConfig = field(default_factory=DTWPhaseEstimatorConfig)

    @override
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        phases = PhasesCollection[NumDemos, NumPoints].zeros_like(self.demonstrations)
        ref_demo = self.demonstrations[self.config.reference_demo_index]
        T_ref = ref_demo.length
        ref_phases = Phases[NumPoints](np.linspace(0.0, 1.0, T_ref, dtype=npDType))
        phases[ref_demo.index] = ref_phases
        ref_states = ref_demo.states.numpy()
        for demo in track(self.demonstrations, description="[bold]DTW alignment[/]"):
            if demo.index == ref_demo.index:
                continue
            states = demo.states.numpy()
            path = dtw_ndim.warping_path(ref_states, states)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            aligned: list[list[Phase]] = [[] for _ in range(demo.length)]
            ref_t: DemoIndex
            demo_t: DemoIndex
            for ref_t, demo_t in path:  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]
                aligned[demo_t].append(ref_phases[ref_t])
            taus = np.zeros(demo.length, dtype=npDType)
            for t in range(demo.length):
                if aligned[t]:
                    taus[t] = npDType(np.mean(aligned[t]))
            taus = np.maximum.accumulate(taus)
            taus = normalise(taus, method="MINMAX")
            phases[demo.index] = Phases[NumPoints](taus)
        self.phases: PhasesCollection[NumDemos, NumPoints] = phases
        return phases


## ─────────────────────────────────────────────────────────────────────────────
