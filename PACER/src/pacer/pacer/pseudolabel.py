"""
Pseudo-Label Refinement
=======
"""
# src/pacer/pacer/pseudolabel.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Generic, Protocol

import numpy.linalg as la
from typingkit.core import RuntimeGeneric

from pacer.base import ActionsCollection, Demonstrations, StatesCollection
from pacer.pacer.base import MetricValue
from pacer.pacer.binning import Bins, ConsensusStatistics
from pacer.pacer.consensus import ConsensusConfig
from pacer.pacer.mode import ACTION_MODE, STATE_MODE, VectorMode
from pacer.pacer.trust import TrustValue, TrustValuesCollection
from pacer.typings import (
    CollectionType,
    DimAction,
    DimState,
    FloatLike,
    NumBins,
    NumDemos,
    NumPoints,
    VectorType,
    npDType,
)
from pacer.utils import attenuate_perpendicular, normalise, rescale_norm

## ── Pseudo-Label Refinement ──────────────────────────────────────────────────


@dataclass(kw_only=True)
class PseudoLabels(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Container for corrected action/state labels."""

    actions: ActionsCollection[NumDemos, NumPoints, DimAction]
    states: StatesCollection[NumDemos, NumPoints, DimState] | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class PseudoLabelContext(Generic[VectorType, DimState, DimAction]):
    """Immutable shared pseudo-label refinement context."""

    trust: TrustValue
    anchor: VectorType
    unit_tangent: VectorType
    median_strength: MetricValue
    apply_sideways_attenuation: bool


class PseudoLabelRefinementStep(Protocol):
    """Single pseudo-label refinement step."""

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType: ...


@dataclass(frozen=True, slots=True)
class PseudoLabelRefinementPipeline:
    """Sequential pseudo-label refinement pipeline."""

    steps: Sequence[PseudoLabelRefinementStep] = field(
        default_factory=list[PseudoLabelRefinementStep]
    )

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        for step in self.steps:
            y = step.apply(y, ctx=ctx)
        return y


@dataclass(frozen=True, kw_only=True, slots=True)
class DebiasTowardsAnchorStep:
    """Pull labels toward robust consensus anchor."""

    debias_weight: FloatLike = 0.5  # lambda_{debias}

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        gamma = npDType(1.0 - self.debias_weight * (1.0 - ctx.trust))  # gamma_{i, t}
        assert 0 <= gamma <= 1
        y_next = gamma * y + (1.0 - gamma) * ctx.anchor  # y^{(1)}_{i, t}
        return type(y)(y_next)


@dataclass(frozen=True, kw_only=True, slots=True)
class SidewaysAttenuationStep:
    """Shrinks perpendicular components relative to ribbon tangent."""

    shrinkage: FloatLike = 0.5  # rho_0

    def __post_init__(self) -> None:
        assert 0 <= self.shrinkage <= 1

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        if not ctx.apply_sideways_attenuation:
            return y
        rho = npDType(self.shrinkage * (1.0 - ctx.trust))
        y_next = attenuate_perpendicular(  # y^{(2)}_{i, t}
            y, unit_direction=ctx.unit_tangent, attenuation=rho
        )
        return type(y)(y_next)


@dataclass(frozen=True, kw_only=True, slots=True)
class SpeedRegularisationStep:
    """Blends magnitude toward robust consensus magnitude."""

    influence: FloatLike = 0.5  # eta_0

    def __post_init__(self) -> None:
        assert 0 <= self.influence <= 1

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        eta = npDType(self.influence * (1.0 - ctx.trust))  # eta_{i, t}
        target_norm = (1.0 - eta) * la.norm(y) + eta * ctx.median_strength  # s_{i, t}
        y_next = rescale_norm(y, target_norm=target_norm)  # y^{(3)}_{i, t}
        return type(y)(y_next)


@dataclass(slots=True, kw_only=True)
class TemporalSmoother:
    """Exponential moving average smoothing across trajectory."""

    smoothing_weight: FloatLike = 0.0  # kappa

    def smooth(self, labels: Iterable[VectorType]) -> list[VectorType]:
        labels = list(labels)
        if not labels:
            return []
        smoothed = list[VectorType]()
        prev: VectorType | None = None
        for y in labels:
            if prev is None:
                prev = y
            else:
                prev = type(y)(
                    (1.0 - self.smoothing_weight) * y + self.smoothing_weight * prev
                )
            smoothed.append(prev)
        return smoothed


@dataclass(kw_only=True)
class PseudoLabelParams:
    pipeline: PseudoLabelRefinementPipeline = field(
        default_factory=PseudoLabelRefinementPipeline
    )
    smoother: TemporalSmoother = field(default_factory=TemporalSmoother)


@dataclass
class PseudoLabelComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Generates corrected pseudo-labels using trust and bin statistics."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)

    def _compute_labels(
        self,
        trust_values: TrustValuesCollection[NumDemos, NumPoints],
        mode: VectorMode[
            CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction
        ],
        params: PseudoLabelParams,
    ) -> CollectionType:
        pre_smooth = mode.make_collection(self.demonstrations)  # y^{(3)} for all (i, t)
        for bin in self.bins:
            token = bin.ribbon_token
            has_state_tangent = token.state_tangent is not None

            tangent = (
                token.state_tangent
                if token.state_tangent is not None
                and token.state_tangent.shape == token.action_tangent.shape
                else token.action_tangent
            )
            unit_tangent = mode.wrap(normalise(tangent, method="NORM"))

            apply_sideways_attenuation = (
                not mode.attenuation_requires_state_tangent
            ) or has_state_tangent

            for j in self.demonstrations.demo_indices:
                loo_stats = ConsensusStatistics[DimState, DimAction].for_bin(
                    bin, consensus_config=self.consensus_config, LOO_demo_index=j
                )
                anchor = mode.anchor_from_stats(loo_stats)
                median_strength = mode.strength_from_token(token)

                for t, sample in bin.samples_collection[j].items():
                    trust = trust_values[j][t]  # w_{j, t}
                    ctx = PseudoLabelContext[VectorType, DimState, DimAction](
                        trust=trust,
                        anchor=anchor,
                        unit_tangent=unit_tangent,
                        median_strength=median_strength,
                        apply_sideways_attenuation=apply_sideways_attenuation,
                    )
                    y = mode.vector_from_sample(sample)
                    y_refined = params.pipeline.apply(y, ctx=ctx)
                    mode.set_item(pre_smooth, j, t, mode.wrap(y_refined))

        # Temporal smoothing
        smoothed = mode.make_collection(self.demonstrations)  # y* for all (i, t)
        for i in self.demonstrations.demo_indices:
            labels = [
                mode.get_item(pre_smooth, i, t)
                for t in self.demonstrations[i].time_indices
            ]
            refined = params.smoother.smooth(labels)
            for t, y in zip(self.demonstrations[i].time_indices, refined):
                mode.set_item(smoothed, i, t, y)

        return smoothed

    def compute_pseudo_labels(
        self,
        action_trust_values: TrustValuesCollection[NumDemos, NumPoints],
        state_trust_values: TrustValuesCollection[NumDemos, NumPoints] | None = None,
        *,
        action_params: PseudoLabelParams,
        state_params: PseudoLabelParams | None = None,
    ) -> PseudoLabels[NumDemos, NumPoints, DimState, DimAction]:
        """Produces final pseudo-labels for actions and optionally states."""
        actions = self._compute_labels(
            action_trust_values, ACTION_MODE(), action_params
        )
        states = (
            self._compute_labels(state_trust_values, STATE_MODE(), state_params)
            if state_trust_values is not None and state_params is not None
            else None
        )
        return PseudoLabels(actions=actions, states=states)


## ─────────────────────────────────────────────────────────────────────────────
