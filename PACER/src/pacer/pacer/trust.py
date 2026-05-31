"""
Trust Value Computation
=======
"""
# src/pacer/pacer/trust.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Protocol

import numpy as np
import numpy.linalg as la
from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.pacer.base import (
    Residual,
    TrustValue,
    TrustValuesCollection,
    ZScore,
    ZScoresCollection,
)
from pacer.pacer.binning import Bins, ConsensusStatistics
from pacer.pacer.consensus import (
    ConsensusConfig,
    MADResidualScaleEstimator,
    ResidualScaleEstimator,
)
from pacer.pacer.mode import VectorMode
from pacer.typings import (
    CollectionType,
    DimAction,
    DimState,
    FloatLike,
    NumBins,
    NumDemos,
    NumPoints,
    VectorType,
)
from pacer.utils import EPS

## ── Trust Value Computation ──────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True, slots=True)
class TrustComputationContext(RuntimeGeneric[VectorType]):
    """Shared immutable trust-computation context."""

    anchor: VectorType


# ── Residual Computation ──────────────────────────────────────────────────────


class ResidualComputer(Protocol):
    """Computes residual for a sample relative to consensus."""

    def compute(
        self, vector: VectorType, *, ctx: TrustComputationContext[VectorType]
    ) -> Residual: ...


@dataclass(frozen=True, slots=True)
class EuclideanResidualComputer:
    def compute(
        self, vector: VectorType, *, ctx: TrustComputationContext[VectorType]
    ) -> Residual:
        return Residual(la.norm(vector - ctx.anchor))


# ── Trust Kernels ─────────────────────────────────────────────────────────────


class TrustKernel(Protocol):
    """Maps robust z-score to trust value."""

    def compute(self, z_score: ZScore) -> TrustValue: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TukeyBiweightKernel:
    """Tukey biweight robust trust kernel."""

    cutoff: FloatLike = 4.685  # c

    def __post_init__(self) -> None:
        assert 3 <= self.cutoff <= 5

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        if z_score <= self.cutoff:
            weight = (1 - (z_score / self.cutoff) ** 2) ** 2
        else:
            weight = 0.0
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class GaussianKernel:
    sigma: FloatLike = 1.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = float(np.exp(-0.5 * (z_score / self.sigma) ** 2))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class HuberKernel:
    """Huber weighting function."""

    delta: FloatLike = 1.345
    # 95% efficiency under Gaussian

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        abs_z_score = abs(z_score)
        if abs_z_score <= self.delta:
            weight = 1.0
        else:
            weight = self.delta / abs_z_score
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class WelschKernel:
    """Welsch (exponential) weighting function."""

    scale: FloatLike = 2.9846  # Tuning constant

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = np.exp(-((z_score / self.scale) ** 2))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class CauchyKernel:
    """Cauchy (Lorentzian) weighting function."""

    scale: FloatLike = 2.3849  # Tuning constant

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 / (1.0 + (z_score / self.scale) ** 2)
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class LogisticKernel:
    midpoint: FloatLike = 2.0
    sharpness: FloatLike = 2.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 / (1.0 + np.exp(self.sharpness * (z_score - self.midpoint)))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class HardThresholdKernel:
    cutoff: FloatLike = 3.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 if z_score <= self.cutoff else 0.0
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class StudentTKernel:
    dof: FloatLike = 4.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = (self.dof + 1) / (self.dof + z_score**2)
        return TrustValue(min(1.0, weight))


@dataclass(frozen=True, kw_only=True, slots=True)
class GemanMcClureKernel:
    """Geman-McClure weighting function."""

    sigma: FloatLike = 1.0

    def compute(self, z_score: ZScore) -> TrustValue:
        denom = 1.0 + (z_score / self.sigma) ** 2
        return TrustValue(1.0 / (denom * denom))


@dataclass(frozen=True, kw_only=True, slots=True)
class AndrewsSineKernel:
    """Andrews sine weighting function."""

    k: FloatLike = 1.34 * np.pi  # ~4.21

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        if abs(z_score) < self.k:
            t = z_score / self.k
            weight = np.sin(t) / t if t != 0 else 1.0
        else:
            weight = 0.0
        return TrustValue(weight)


# ── Trust Transforms ──────────────────────────────────────────────────────────


class TrustTransform(Protocol):
    """Post-processing transform on trust values."""

    def apply(self, trust: TrustValue) -> TrustValue: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class MinimumTrustFloor:
    """Applies minimum trust floor."""

    minimum: FloatLike = 0.02  # w_min

    def apply(self, trust: TrustValue) -> TrustValue:
        return TrustValue(max(trust, self.minimum))


# ── Trust Pipeline ────────────────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True, slots=True)
class TrustPipeline:
    """Trust computation pipeline."""

    residual_computer: ResidualComputer = field(
        default_factory=EuclideanResidualComputer
    )
    scale_estimator: ResidualScaleEstimator = field(
        default_factory=MADResidualScaleEstimator
    )
    kernel: TrustKernel = field(default_factory=TukeyBiweightKernel)
    transforms: Sequence[TrustTransform] = field(
        default_factory=lambda: [MinimumTrustFloor()]
    )

    def compute_scale(self, residuals: Sequence[Residual]) -> Residual:
        return self.scale_estimator.compute(residuals)

    def compute_trust(
        self, residual: Residual, *, scale: Residual
    ) -> tuple[ZScore, TrustValue]:
        z_score = ZScore(residual / (scale + EPS))
        trust = self.kernel.compute(z_score)
        for transform in self.transforms:
            trust = transform.apply(trust)
        return z_score, trust


# ── Trust Value Computation ───────────────────────────────────────────────────


@dataclass(kw_only=True)
class TrustValueParams:
    pipeline: TrustPipeline = field(default_factory=TrustPipeline)


@dataclass(kw_only=True)
class TrustValueComputationResult(RuntimeGeneric[NumDemos, NumPoints]):
    z_scores: ZScoresCollection[NumDemos, NumPoints]  # (N x T_)
    trust_values: TrustValuesCollection[NumDemos, NumPoints]  # (N x T_)


@dataclass
class TrustValueComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Computes z-scores and trust values for samples."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)

    def compute(
        self,
        *,
        mode: VectorMode[
            CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction
        ],
        params: TrustValueParams,
    ) -> TrustValueComputationResult[NumDemos, NumPoints]:
        pipeline = params.pipeline
        z_scores = ZScoresCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        trust_values = TrustValuesCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                loo_stats = ConsensusStatistics[DimState, DimAction].for_bin(
                    bin, consensus_config=self.consensus_config, LOO_demo_index=i
                )
                ctx = TrustComputationContext(anchor=mode.anchor_from_stats(loo_stats))

                reference_samples = [
                    sample
                    for j, samples in bin.samples_collection.items()
                    if j != i
                    for sample in samples.values()
                ]
                residuals = [
                    pipeline.residual_computer.compute(
                        mode.vector_from_sample(sample), ctx=ctx
                    )
                    for sample in reference_samples
                ]

                scale = pipeline.compute_scale(residuals)  # MAD_{a}^{(-i)}[b]

                for t, sample in bin.samples_collection[i].items():
                    residual = pipeline.residual_computer.compute(
                        mode.vector_from_sample(sample), ctx=ctx
                    )
                    z_score, trust = pipeline.compute_trust(residual, scale=scale)
                    z_scores[i][t] = z_score  # z_{i, t}
                    trust_values[i][t] = trust

        return TrustValueComputationResult(z_scores=z_scores, trust_values=trust_values)


## ─────────────────────────────────────────────────────────────────────────────
