"""
Consensus
=========
Consensus estimators for PACER.
"""
# src/pacer/pacer/consensus.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pacer.base import Action, Actions, State, States
from pacer.pacer.base import MetricValue, Residual
from pacer.typings import DimAction, DimState, FloatLike, NumPoints
from pacer.utils import MAD_SCALE, mean, median

## ── Consensus ────────────────────────────────────────────────────────────────

# ── Location Estimation ───────────────────────────────────────────────────────


class VectorLocationEstimator(Protocol):
    """Computes consensus vector estimate."""

    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]: ...

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]: ...


class ScalarLocationEstimator(Protocol):
    """Computes scalar consensus estimate."""

    def compute(self, values: npt.ArrayLike) -> MetricValue: ...


# ── Mean Estimators ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MeanVectorEstimator:
    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]:
        return Action[DimAction](mean(actions, axis=0))

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]:
        return State[DimState](mean(states, axis=0))


@dataclass(frozen=True, slots=True)
class MeanScalarEstimator:
    def compute(self, values: npt.ArrayLike) -> MetricValue:
        return MetricValue(mean(values))


# ── Median Estimators ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MedianVectorEstimator:
    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]:
        return Action[DimAction](median(actions, axis=0))

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]:
        return State[DimState](median(states, axis=0))


@dataclass(frozen=True, slots=True)
class MedianScalarEstimator:
    def compute(self, values: npt.ArrayLike) -> MetricValue:
        return MetricValue(median(values))


# ── Scale Estimation ──────────────────────────────────────────────────────────


class ResidualScaleEstimator(Protocol):
    """Computes robust scale from residuals."""

    def compute(self, residuals: Sequence[Residual]) -> Residual: ...


@dataclass(frozen=True, slots=True)
class MADResidualScaleEstimator:
    """Median absolute deviation scale estimator."""

    consistency_scale: FloatLike = MAD_SCALE

    def compute(self, residuals: Sequence[Residual]) -> Residual:
        median_residual: Residual = Residual(median(residuals))
        abs_deviations: list[Residual] = [
            Residual(abs(residual - median_residual)) for residual in residuals
        ]
        return Residual(self.consistency_scale * median(abs_deviations))


@dataclass(frozen=True, slots=True)
class StandardDeviationScaleEstimator:
    ddof: int = 0

    def compute(self, residuals: Sequence[Residual]) -> Residual:
        return Residual(np.std(residuals, ddof=self.ddof))


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    vector_estimator: VectorLocationEstimator = MedianVectorEstimator()
    scalar_estimator: ScalarLocationEstimator = MedianScalarEstimator()
    residual_scale_estimator: ResidualScaleEstimator = MADResidualScaleEstimator()


## ─────────────────────────────────────────────────────────────────────────────
