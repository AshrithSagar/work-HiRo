"""
Trust Value Computation
=======
"""
# src/pacer/pacer/trust.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Literal

import numpy.linalg as la
from typingkit.core import RuntimeGeneric

from pacer.base import Action, Demonstrations, State
from pacer.pacer.base import (
    Residual,
    TrustValue,
    TrustValuesCollection,
    ZScore,
    ZScoresCollection,
)
from pacer.pacer.binning import Bins, RobustStatistics
from pacer.typings import DimAction, DimState, FloatLike, NumBins, NumDemos, NumPoints
from pacer.utils import EPS, MAD_SCALE, median

## ── Trust Value Computation ──────────────────────────────────────────────────


@dataclass(kw_only=True)
class TrustValueParams:
    """Hyperparameters controlling trust value computations."""

    tukey_cutoff: FloatLike = 4.685  # c
    min_trust: FloatLike = 0.02  # w_min

    def __post_init__(self) -> None:
        assert 3 <= self.tukey_cutoff <= 5


@dataclass
class TrustValueComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Computes z-scores and trust values for samples."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    choice: Literal["State", "Action"] = "Action"

    def compute_z_scores(self) -> ZScoresCollection[NumDemos, NumPoints]:  # (N x T_)
        z_scores = ZScoresCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=i
                )

                loo_median: State[DimState] | Action[DimAction]
                match self.choice:
                    case "State":
                        loo_median = loo_stats.median_state
                        loo_residuals = [
                            Residual(la.norm(state - loo_median))
                            for state in bin.states(LOO_demo_index=i)
                        ]
                    case "Action":
                        loo_median = loo_stats.median_action  # alpha_a^{(-i)}[b]
                        loo_residuals = [
                            Residual(la.norm(action - loo_median))
                            for action in bin.actions(LOO_demo_index=i)
                        ]

                median_residual = Residual(median(loo_residuals))
                abs_deviations = [
                    Residual(abs(residual - median_residual))
                    for residual in loo_residuals
                ]
                MAD_residual = Residual(
                    MAD_SCALE * median(abs_deviations)
                )  # MAD_{a}^{(-i)}[b]

                demo_samples = bin.samples_collection[i]
                for t, sample in demo_samples.items():
                    match self.choice:
                        case "State":
                            self_residual = Residual(la.norm(sample.state - loo_median))
                        case "Action":
                            self_residual = Residual(
                                la.norm(sample.action - loo_median)
                            )  # r^{-i}_{i, t}

                    z_score = ZScore(self_residual / (MAD_residual + EPS))  # z_{i, t}
                    z_scores[i][t] = z_score

        return z_scores

    def compute_trust_values(
        self, *, params: TrustValueParams
    ) -> TrustValuesCollection[NumDemos, NumPoints]:  # (N x T_)
        trust_values = TrustValuesCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        z_scores = self.compute_z_scores()
        for i, scores in z_scores.items():
            for t, z_score in enumerate(scores):
                if z_score <= params.tukey_cutoff:
                    trust_value = (1 - (z_score / params.tukey_cutoff) ** 2) ** 2
                else:
                    trust_value = TrustValue(0)
                if trust_value < params.min_trust:
                    trust_value = TrustValue(params.min_trust)
                trust_values[i][t] = trust_value
        return trust_values  # [[w_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}


## ─────────────────────────────────────────────────────────────────────────────
