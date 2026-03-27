"""
Phase evaluation
=======
"""
# src/pacer/phase/evaluation.py

## ── Imports ──────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, override

import numpy as np
import optype.numpy as onp
from scipy.stats import spearmanr
from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.phase.base import Phases, PhasesCollection
from pacer.typings import (
    DimAction,
    DimState,
    Matrix,
    NumDemos,
    NumPoints,
    Vector,
    npDType,
)
from pacer.utils import EPS

## ── Phase Evaluation ─────────────────────────────────────────────────────────


@dataclass
class PhaseEvaluator(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC):
    """Abstract interface to evaluate phases for a set of demonstrations."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    phases: PhasesCollection[NumDemos, NumPoints]

    @abstractmethod
    def evaluate(self) -> float:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RankingLossEvaluator(PhaseEvaluator[NumDemos, NumPoints, DimState, DimAction]):
    """
    Computes a pairwise ranking loss to enforce temporal ordering of phases.

    Penalises cases where later time steps do not have sufficiently larger phase
    values than earlier ones using a soft margin logistic loss.
    Lower is better.
    """

    margin: float = 1.0

    @override
    def evaluate(self) -> float:
        loss = float(0.0)
        for taus in self.phases.values():
            _taus = taus.numpy()
            diff = Matrix[NumPoints, NumPoints](_taus[None, :] - _taus[:, None])
            mask = Matrix[NumPoints, NumPoints](np.triu(np.ones_like(diff), k=1))
            loss_matrix = Matrix[NumPoints, NumPoints](
                np.log1p(np.exp(self.margin - diff)) * mask
            )
            loss += loss_matrix.sum() / (mask.sum() + EPS)
        N = self.phases.__len__()
        return float(loss / max(int(N), 1))


@dataclass
class SpearmanEvaluator(PhaseEvaluator[NumDemos, NumPoints, DimState, DimAction]):
    """
    Measures monotonicity of phases using Spearman rank correlation.

    Compares phase values against time indices to evaluate whether phases
    increase consistently over the trajectory.
    Higher is better (max = 1).
    """

    @override
    def evaluate(self) -> float:
        total = float(0.0)
        for taus in self.phases.values():
            _taus = taus.numpy()
            t = np.arange(len(_taus))
            total += float(spearmanr(t, _taus).correlation)  # type: ignore[attr-defined]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
        N = self.phases.__len__()
        return float(total / max(int(N), 1))


@dataclass
class PathConsistencyEvaluator(
    PhaseEvaluator[NumDemos, NumPoints, DimState, DimAction]
):
    """
    Evaluates consistency between phase values and trajectory progress.

    Constructs a reference phase based on cumulative path length and computes
    mean squared error against predicted phases.
    Lower is better.
    """

    @override
    def evaluate(self) -> float:
        total = float(0.0)
        for demo in self.demonstrations:
            diffs: onp.Array2D[npDType] = np.diff(demo.states, axis=0)  # (T_i-1, d_x)
            norms: onp.Array1D[npDType] = np.linalg.norm(diffs, axis=1)  # (T_i-1,)
            lengths = Vector[NumPoints](np.r_[0, np.cumsum(norms)])
            path_tau = Phases[NumPoints](lengths / (lengths[-1] + EPS)).numpy()
            taus = self.phases[demo.index].numpy()
            total += float(np.mean((taus - path_tau) ** 2))
        N = self.demonstrations.__len__()
        return float(total / max(int(N), 1))


@dataclass
class AlignmentEvaluator(PhaseEvaluator[NumDemos, NumPoints, DimState, DimAction]):
    """
    Measures cross-demonstration alignment under phase reparameterisation.

    Resamples trajectories onto a common phase grid and computes variance
    across demonstrations.
    Lower variance indicates better alignment.
    """

    n_grid: int = 50

    def _resample(
        self,
        states: Matrix[NumPoints, DimState],
        taus: Vector[NumPoints],
        grid: onp.Array1D[npDType],
    ) -> Matrix[Any, DimState]:
        return Matrix[Any, DimState](
            np.stack(
                [
                    np.interp(grid, taus, Vector[NumPoints](states[:, d]))
                    for d in range(states.shape[1])
                ],
                axis=-1,
                dtype=npDType,
            )
        )

    @override
    def evaluate(self) -> float:
        grid = np.linspace(0, 1, self.n_grid, dtype=npDType)
        _aligned = list[Matrix[Any, DimState]]()
        for demo in self.demonstrations:
            states = demo.states.numpy()
            taus = self.phases[demo.index].numpy()
            _aligned.append(self._resample(states, taus, grid))
        aligned: onp.Array3D[npDType] = np.stack(_aligned)  # (N, G, D)
        mean: onp.Array2D[npDType] = aligned.mean(axis=0)
        return float(np.mean((aligned - mean) ** 2))


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PhaseEvaluationReport:
    """Container for phase evaluation metrics."""

    ranking_loss: float
    spearman: float
    path_consistency: float
    alignment: float

    @classmethod
    def evaluate(
        cls,
        demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction],
        phases: PhasesCollection[NumDemos, NumPoints],
    ) -> Self:
        return cls(
            ranking_loss=RankingLossEvaluator(demonstrations, phases).evaluate(),
            spearman=SpearmanEvaluator(demonstrations, phases).evaluate(),
            path_consistency=PathConsistencyEvaluator(
                demonstrations, phases
            ).evaluate(),
            alignment=AlignmentEvaluator(demonstrations, phases).evaluate(),
        )


## ─────────────────────────────────────────────────────────────────────────────
