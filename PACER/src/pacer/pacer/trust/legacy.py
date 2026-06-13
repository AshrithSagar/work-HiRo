"""
Trust Value Computation
=======
"""
# src/pacer/pacer/trust/legacy.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Protocol

import numpy.linalg as la
from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.pacer.base import Residual
from pacer.pacer.binning import Bin, Bins, ConsensusStatistics
from pacer.pacer.consensus import (
    ConsensusConfig,
    MADResidualScaleEstimator,
    ResidualScaleEstimator,
)
from pacer.pacer.mode import VectorMode
from pacer.pacer.trust import TrustValue, TrustValuesCollection
from pacer.pacer.trust.base import ZScore, ZScoresCollection
from pacer.pacer.trust.kernels import (
    MinimumTrustFloor,
    TrustKernel,
    TrustTransform,
    TukeyBiweightKernel,
)
from pacer.typings import (
    CollectionType,
    DemoIndex,
    DimAction,
    DimState,
    NumBins,
    NumDemos,
    NumPoints,
    VectorType,
)
from pacer.utils import EPS

## ── Trust Value Computation ──────────────────────────────────────────────────


@dataclass(frozen=True, kw_only=True, slots=True)
class ConsensusInfo(RuntimeGeneric[VectorType]):
    """Consensus information required to evaluate a single sample."""

    anchor: VectorType
    residual_scale: Residual


# ── Residual Computation ──────────────────────────────────────────────────────


class ResidualComputer(Protocol):
    """Computes residual for a sample relative to consensus."""

    def compute(
        self, vector: VectorType, *, consensus: ConsensusInfo[VectorType]
    ) -> Residual: ...


@dataclass(frozen=True, slots=True)
class EuclideanResidualComputer:
    def compute(
        self, vector: VectorType, *, consensus: ConsensusInfo[VectorType]
    ) -> Residual:
        return Residual(la.norm(vector - consensus.anchor))


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

    def compute_trust(
        self, residual: Residual, *, scale: Residual
    ) -> tuple[ZScore, TrustValue]:
        z_score = ZScore(residual / (scale + EPS))
        trust = self.kernel.compute(z_score)
        for transform in self.transforms:
            trust = transform.apply(trust)
        return z_score, trust


# ── Consensus Builder ─────────────────────────────────────────────────────────


@dataclass
class LOOConsensusInfoBuilder(
    RuntimeGeneric[CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction]
):
    mode: VectorMode[
        CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction
    ]
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)
    residual_computer: ResidualComputer = field(
        default_factory=EuclideanResidualComputer
    )
    scale_estimator: ResidualScaleEstimator = field(
        default_factory=MADResidualScaleEstimator
    )

    def build_for(
        self,
        *,
        bin: Bin[NumDemos, NumPoints, DimState, DimAction],
        demo_index: DemoIndex,
    ) -> ConsensusInfo[VectorType]:
        loo_stats = ConsensusStatistics[DimState, DimAction].for_bin(
            bin, consensus_config=self.consensus_config, LOO_demo_index=demo_index
        )
        anchor = self.mode.anchor_from_stats(loo_stats)
        consensus = ConsensusInfo(anchor=anchor, residual_scale=Residual(1))
        reference_samples = [
            sample
            for j, samples in bin.samples_collection.items()
            if j != demo_index
            for sample in samples.values()
        ]
        residuals = [
            self.residual_computer.compute(
                self.mode.vector_from_sample(sample), consensus=consensus
            )
            for sample in reference_samples
        ]
        residual_scale = self.scale_estimator.compute(residuals)  # ~ MAD_{a}^{(-i)}[b]
        return ConsensusInfo(
            anchor=anchor,
            residual_scale=residual_scale,
        )


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
        z_scores = ZScoresCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        trust_values = TrustValuesCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        consensus_builder = LOOConsensusInfoBuilder(
            mode=mode,
            consensus_config=self.consensus_config,
            residual_computer=params.pipeline.residual_computer,
            scale_estimator=params.pipeline.scale_estimator,
        )

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                consensus = consensus_builder.build_for(bin=bin, demo_index=i)
                for t, sample in bin.samples_collection[i].items():
                    residual = params.pipeline.residual_computer.compute(
                        mode.vector_from_sample(sample), consensus=consensus
                    )
                    z_score, trust = params.pipeline.compute_trust(
                        residual, scale=consensus.residual_scale
                    )
                    z_scores[i][t] = z_score  # z_{i, t}
                    trust_values[i][t] = trust

        return TrustValueComputationResult(z_scores=z_scores, trust_values=trust_values)


## ─────────────────────────────────────────────────────────────────────────────
