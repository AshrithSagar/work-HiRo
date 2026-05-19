"""
Trust Value Computation
=======
"""
# src/pacer/pacer/trust.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Generic, Protocol

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
from pacer.pacer.binning import Bins, RobustStatistics
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
from pacer.utils import EPS, MAD_SCALE, median

## ── Trust Value Computation ──────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True, slots=True)
class TrustComputationContext(Generic[VectorType]):
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


# ── Scale Estimation ──────────────────────────────────────────────────────────


class ScaleEstimator(Protocol):
    """Computes robust scale from residuals."""

    def compute(self, residuals: list[Residual]) -> Residual: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class MADScaleEstimator:
    """Median absolute deviation scale estimator."""

    consistency_scale: FloatLike = MAD_SCALE

    def compute(self, residuals: list[Residual]) -> Residual:
        median_residual: Residual = Residual(median(residuals))
        abs_deviations: list[Residual] = [
            Residual(abs(residual - median_residual)) for residual in residuals
        ]
        return Residual(self.consistency_scale * median(abs_deviations))


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
    scale_estimator: ScaleEstimator = field(default_factory=MADScaleEstimator)
    kernel: TrustKernel = field(default_factory=TukeyBiweightKernel)
    transforms: tuple[TrustTransform, ...] = field(
        default_factory=lambda: (MinimumTrustFloor(),)
    )

    def compute_scale(self, residuals: list[Residual]) -> Residual:
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


@dataclass
class TrustValueComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Computes z-scores and trust values for samples."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]

    def compute(
        self,
        *,
        mode: VectorMode[
            CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction
        ],
        params: TrustValueParams,
    ) -> tuple[
        ZScoresCollection[NumDemos, NumPoints],  # (N x T_)
        TrustValuesCollection[NumDemos, NumPoints],  # (N x T_)
    ]:
        pipeline = params.pipeline
        z_scores = ZScoresCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        trust_values = TrustValuesCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=i
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

        return z_scores, trust_values


## ─────────────────────────────────────────────────────────────────────────────
