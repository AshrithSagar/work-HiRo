"""
Analysis
=======
Diagnostics and analysis utilities.
"""
# src/pacer/analysis.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeAlias

import numpy as np
import numpy.linalg as la
from typingkit.core import TypedList

from pacer.base import ActionsCollection, Demonstrations
from pacer.pacer import (
    PACERResult,
    Residual,
    Residuals,
    ResidualsCollection,
    TrustValuesCollection,
)
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints, npDType

## ── Analysis ─────────────────────────────────────────────────────────────────

MetricValue: TypeAlias = np.float32
MetricSeries: TypeAlias = TypedList[NumPoints, MetricValue]
MetricCollection: TypeAlias = TypedList[NumDemos, MetricSeries[NumPoints]]

# ── Residual Analysis ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class ResidualAnalysis(Generic[NumDemos, NumPoints, DimState, DimAction]):
    residuals: ResidualsCollection[NumDemos, NumPoints]

    @property
    def _flattened_residuals(self) -> list[Residual]:
        return [residual for residuals in self.residuals for residual in residuals]

    @property
    def mean_residual(self) -> MetricValue:
        return MetricValue(np.mean(self._flattened_residuals))

    @property
    def median_residual(self) -> MetricValue:
        return MetricValue(np.median(self._flattened_residuals))

    @property
    def max_residual(self) -> MetricValue:
        return MetricValue(np.max(self._flattened_residuals))


@dataclass(slots=True)
class ResidualAnalyser(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]

    def compute_action_residuals(
        self,
    ) -> ResidualAnalysis[NumDemos, NumPoints, DimState, DimAction]:
        residuals: list[Residuals[NumPoints]] = []
        pseudo_actions = self.pacer_result.pseudo_labels.actions

        for i, demo in enumerate(self.demonstrations):
            demo_residuals = Residuals[NumPoints]()

            for t in demo.time_indices:
                original = demo.actions[t]
                pseudo = pseudo_actions[i][t]

                residual = Residual(la.norm(pseudo - original))
                demo_residuals.append(residual)
            residuals.append(demo_residuals)
        return ResidualAnalysis(TypedList[NumDemos, Residuals[NumPoints]](residuals))


# ── Trust Value Analysis ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TrustStatistics:
    mean: MetricValue
    median: MetricValue
    minimum: MetricValue
    maximum: MetricValue
    std: MetricValue


@dataclass(slots=True)
class TrustValueAnalysis(Generic[NumDemos, NumPoints]):
    trust_values: TrustValuesCollection[NumDemos, NumPoints]

    def statistics(self) -> TrustStatistics:
        flat = np.asarray(
            [value for values in self.trust_values.values() for value in values],
            dtype=npDType,
        )
        return TrustStatistics(
            mean=MetricValue(np.mean(flat)),
            median=MetricValue(np.median(flat)),
            minimum=MetricValue(np.min(flat)),
            maximum=MetricValue(np.max(flat)),
            std=MetricValue(np.std(flat)),
        )

    def low_trust_fraction(self, threshold: float = 0.25) -> MetricValue:
        flat = np.asarray(
            [value for demo in self.trust_values.values() for value in demo],
            dtype=np.float32,
        )

        return MetricValue(np.mean(flat < threshold))


# ── Correction Magnitude Analysis ─────────────────────────────────────────────


@dataclass(slots=True)
class CorrectionMagnitudeAnalysis(Generic[NumDemos, NumPoints, DimState, DimAction]):
    magnitudes: MetricCollection[NumDemos, NumPoints]

    @property
    def mean_magnitude(self) -> MetricValue:
        flat = [m for demo in self.magnitudes for m in demo]
        return MetricValue(np.mean(flat))

    @property
    def max_magnitude(self) -> MetricValue:
        flat = [m for demo in self.magnitudes for m in demo]
        return MetricValue(np.max(flat))


@dataclass(slots=True)
class CorrectionMagnitudeAnalyser(
    Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    pacer_result: PACERResult[
        NumBins,
        NumDemos,
        NumPoints,
        DimState,
        DimAction,
    ]

    def analyse_actions(
        self,
    ) -> CorrectionMagnitudeAnalysis[
        NumDemos,
        NumPoints,
        DimState,
        DimAction,
    ]:
        magnitudes = MetricCollection[NumDemos, NumPoints]()

        pseudo_actions = self.pacer_result.pseudo_labels.actions

        for i, demo in enumerate(self.demonstrations):
            series = MetricSeries[NumPoints]()

            for t in demo.time_indices:
                delta = pseudo_actions[i][t] - demo.actions[t]
                magnitude = MetricValue(la.norm(delta))
                series.append(magnitude)

            magnitudes.append(series)

        return CorrectionMagnitudeAnalysis(magnitudes=magnitudes)


# ── Smoothness Analysis ───────────────────────────────────────────────────────


@dataclass(slots=True)
class SmoothnessAnalysis(Generic[NumDemos, NumPoints]):
    smoothness_scores: TypedList[NumDemos, MetricValue]

    @property
    def mean_smoothness(self) -> MetricValue:
        return MetricValue(np.mean(self.smoothness_scores))


@dataclass(slots=True)
class SmoothnessAnalyser(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    actions: ActionsCollection[NumDemos, NumPoints, DimAction]

    def analyse(self) -> SmoothnessAnalysis[NumDemos, NumPoints]:
        scores = TypedList[NumDemos, MetricValue]()

        for actions in self.actions:
            arr = actions.numpy()

            if arr.shape[0] < 3:
                scores.append(MetricValue(0))
                continue

            acceleration = np.diff(arr, n=2, axis=0)
            jerk_energy = np.mean(np.sum(acceleration**2, axis=1))

            scores.append(MetricValue(jerk_energy))

        return SmoothnessAnalysis(smoothness_scores=scores)


## ─────────────────────────────────────────────────────────────────────────────
