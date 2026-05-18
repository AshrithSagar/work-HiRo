"""
PACER Base
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/pacer.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterable, Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Callable, Generic, Literal, Protocol, Self, TypeAlias, cast

import numpy.linalg as la
from typingkit.core import RuntimeGeneric, TypedDict, TypedList

from pacer.base import (
    Action,
    Actions,
    ActionsCollection,
    Demonstration,
    Demonstrations,
    Sample,
    Samples,
    SamplesCollection,
    State,
    States,
    StatesCollection,
)
from pacer.phase.base import Phase, PhasesCollection
from pacer.testutils import PhasePipeline, PhasePipelineConfig
from pacer.typings import (
    BinIndex,
    CollectionType,
    DemoIndex,
    DimAction,
    DimState,
    FloatLike,
    NumBins,
    NumDemos,
    NumPoints,
    SampleIndex,
    TimeIndex,
    VectorType,
    npDType,
)
from pacer.utils import (
    EPS,
    MAD_SCALE,
    attenuate_perpendicular,
    median,
    normalise,
    rescale_norm,
)

## ── PACER ────────────────────────────────────────────────────────────────────

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

MetricValue: TypeAlias = npDType
MetricSeries: TypeAlias = TypedList[NumPoints, MetricValue]
MetricCollection: TypeAlias = TypedList[NumDemos, MetricSeries[NumPoints]]

# ──────────────────────────────────────────────────────────────────────────────

ZScore: TypeAlias = npDType  # z_{i, t}
"""Normalised residual (z-score)."""


class ZScores(TypedList[NumPoints, ZScore]):
    @classmethod
    def zeros_like(cls, demo: Demonstration[NumPoints, DimState, DimAction]) -> Self:
        T_i = demo.time_indices.length
        return cls.full(T_i, ZScore(0))


class ZScoresCollection(TypedDict[NumDemos, DemoIndex, ZScores[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demos.demo_indices, lambda i: ZScores[NumPoints].zeros_like(demos[i])
        )


# ──────────────────────────────────────────────────────────────────────────────

TrustValue: TypeAlias = npDType  # w_{i, t}
"""Confidence weight for a sample."""


class TrustValues(TypedList[NumPoints, TrustValue]):
    @classmethod
    def zeros_like(cls, demo: Demonstration[NumPoints, DimState, DimAction]) -> Self:
        T_i = demo.time_indices.length
        return cls.full(T_i, TrustValue(0))


class TrustValuesCollection(TypedDict[NumDemos, DemoIndex, TrustValues[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demos.demo_indices, lambda i: TrustValues[NumPoints].zeros_like(demos[i])
        )


# ── Binning ───────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RobustStatistics(RuntimeGeneric[DimState, DimAction]):
    """Robust consensus statistics for a set of samples."""

    ## Stable anchors
    median_action: Action[DimAction]  # alpha_a[b] = median{ a_{i, t} : (i, t) \in I_b }
    median_state: State[DimState]  # alpha_s[b] = median{ x_{i, t} : (i, t) \in I_b }

    ## Pace
    median_action_strength: MetricValue
    # beta_a[b] = median{ ||a_{i, t}|| : (i, t) \in I_b }
    # Captures strength of actions
    median_state_change: MetricValue
    # beta_s[b] = median{ ||xdot_{i, t}|| : (i, t) \in I_b }
    # Captures typical rate of state change

    median_state_norm: MetricValue

    ## Local task dynamics
    # NOTE: `action_tangent` and `state_tangent` are not stored here,
    # but instead, computed and stored in RibbonToken.

    @classmethod
    def for_bin(
        cls,
        bin: Bin[NumDemos, NumPoints, DimState, DimAction],
        *,
        LOO_demo_index: DemoIndex | None = None,
    ) -> Self:
        states = States[Any, DimState]()
        actions = Actions[Any, DimAction]()
        for sample in bin.samples(LOO_demo_index=LOO_demo_index):
            states.append(sample.state)
            actions.append(sample.action)

        action_norms = [la.norm(action) for action in actions]
        state_norms = [la.norm(state) for state in states]
        state_change_norms = [
            la.norm(states[t + 1] - states[t]) for t in range(len(states) - 1)
        ]

        median_action = Action[DimAction](median(actions, axis=0))
        median_state = State[DimState](median(states, axis=0))
        median_action_strength = MetricValue(median(action_norms, axis=0))
        median_state_norm = MetricValue(median(state_norms, axis=0))
        median_state_change = MetricValue(median(state_change_norms, axis=0))

        return cls(
            median_action=median_action,
            median_state=median_state,
            median_action_strength=median_action_strength,
            median_state_norm=median_state_norm,
            median_state_change=median_state_change,
        )


@dataclass(kw_only=True)
class RibbonToken(RuntimeGeneric[DimState, DimAction]):  # z_b
    """
    Robust structured descriptor for a single phase bin `b`.\\
    Encodes both consensus behaviour and degree of variability present at phase `b`.
    """

    median_action: Action[DimAction]  # alpha_a[b]
    median_action_strength: MetricValue  # beta_a[b]
    median_state: State[DimState]  # alpha_s[b]
    median_state_change: MetricValue  # beta_s[b]

    median_state_norm: MetricValue

    ## Local task dynamics
    action_tangent: Action[DimAction] = field(init=False)
    # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = field(init=False)
    # t_s[b] <- diff{ alpha_s[b] }

    MAD_action_residual: Residual  # Median Absolute Deviation of action residuals


@dataclass(kw_only=True)
class Bin(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Container of samples assigned to a phase bin `b`."""

    index: BinIndex  # b
    samples_collection: SamplesCollection[NumDemos, NumPoints, DimState, DimAction] = (
        field(
            default_factory=SamplesCollection[NumDemos, NumPoints, DimState, DimAction]
        )
    )
    ##
    ribbon_token: RibbonToken[DimState, DimAction] = field(init=False)

    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Iterator[Sample[DimState, DimAction]]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.samples(LOO_demo_index=LOO_demo_index)

    def states(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> States[Any, DimState]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.states(LOO_demo_index=LOO_demo_index)

    def actions(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Actions[Any, DimAction]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.actions(LOO_demo_index=LOO_demo_index)


Bins: TypeAlias = TypedList[NumBins, Bin[NumDemos, NumPoints, DimState, DimAction]]
"""Ordered collection of phase bins."""


@dataclass
class Binner(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    """Assigns samples to bins based on phase."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    phases: PhasesCollection[NumDemos, NumPoints]
    _: KW_ONLY
    n_bins: NumBins = field(default=cast(NumBins, 96))  # B

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (Phase(bin_idx / self.n_bins), Phase((bin_idx + 1) / self.n_bins))

    def sample_index_to_bin_index(self, sample_idx: SampleIndex) -> BinIndex:
        i, t = sample_idx
        tau: Phase = Phase(self.phases[i][t])
        bin_idx: BinIndex = min(int(tau * self.n_bins), self.n_bins - 1)
        assert 0 <= bin_idx < self.n_bins
        return bin_idx

    def make_bins(self) -> Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        bins = Bins[NumBins, NumDemos, NumPoints, DimState, DimAction].full(
            self.n_bins,
            lambda bin_idx: Bin(
                index=bin_idx,
                samples_collection=SamplesCollection[
                    NumDemos, NumPoints, DimState, DimAction
                ]({i: Samples() for i in self.demonstrations.demo_indices}),
            ),
        )
        for sample_idx in self.demonstrations.sample_indices:
            demo_idx, time_idx = sample_idx
            bin_idx = self.sample_index_to_bin_index(sample_idx)
            bin = bins[bin_idx]
            sample = self.demonstrations[sample_idx]
            samples = bin.samples_collection[demo_idx]
            samples[time_idx] = sample
        return bins


@dataclass
class RibbonTokenConsolidator(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Computes ribbon tokens from binned samples."""

    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]

    def consolidate_ribbon_tokens(
        self,
    ) -> Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        bin_median_actions = Actions[NumBins, DimAction]()
        bin_median_states = States[NumBins, DimState]()

        for bin in self.bins:
            stats = RobustStatistics[DimState, DimAction].for_bin(bin)

            bin_median_action = stats.median_action  # alpha_a[b]
            bin_action_residuals = list[Residual]()
            for action in bin.actions():
                residual = Residual(la.norm(action - bin_median_action))  # r_{i, t}
                bin_action_residuals.append(residual)
            bin_median_action_residual = Residual(median(bin_action_residuals))
            abs_deviations = list[Residual]()
            for residual in bin_action_residuals:
                abs_deviation = Residual(abs(residual - bin_median_action_residual))
                abs_deviations.append(abs_deviation)
            MAD_action_residual = Residual(MAD_SCALE * median(abs_deviations))

            bin.ribbon_token = RibbonToken(
                median_action=stats.median_action,
                median_action_strength=stats.median_action_strength,
                median_state=stats.median_state,
                median_state_change=stats.median_state_change,
                median_state_norm=stats.median_state_norm,
                MAD_action_residual=MAD_action_residual,
            )
            bin_median_actions.append(stats.median_action)
            bin_median_states.append(stats.median_state)

        for bin in self.bins:
            b = bin.index
            if b == 0:
                p, q, f = b + 1, b, 1.0
            elif b == self.bins.length - 1:
                p, q, f = b, b - 1, 1.0
            else:
                p, q, f = b, b - 1, 0.5
            action_tangent = Action[DimAction](
                f * (bin_median_actions[p] - bin_median_actions[q])
            )
            state_tangent = State[DimState](
                f * (bin_median_states[p] - bin_median_states[q])
            )
            bin.ribbon_token.action_tangent = action_tangent
            bin.ribbon_token.state_tangent = state_tangent

        return self.bins


# ── Trust Value Computation ───────────────────────────────────────────────────


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


# ── Pseudo-Label Refinement ───────────────────────────────────────────────────


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


class PseudoLabelRefinementStep(Protocol[VectorType]):
    """Single pseudo-label refinement step."""

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType: ...


@dataclass(frozen=True, slots=True)
class RefinementPipeline(Generic[VectorType]):
    """Sequential refinement pipeline."""

    steps: tuple[PseudoLabelRefinementStep[VectorType], ...] = ()

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        for step in self.steps:
            y = step.apply(y, ctx=ctx)
        return y


@dataclass(frozen=True, kw_only=True, slots=True)
class DebiasTowardsAnchorStep(Generic[VectorType]):
    """Pull labels toward robust consensus anchor."""

    debias_weight: FloatLike = 0.5  # lambda_{debias}

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        gamma = npDType(1.0 - self.debias_weight * (1.0 - ctx.trust))
        y_next = gamma * y + (1.0 - gamma) * ctx.anchor
        return type(y)(y_next)


@dataclass(frozen=True, kw_only=True, slots=True)
class SidewaysAttenuationStep(Generic[VectorType]):
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
        y_next = attenuate_perpendicular(
            y, unit_direction=ctx.unit_tangent, attenuation=rho
        )
        return type(y)(y_next)


@dataclass(frozen=True, kw_only=True, slots=True)
class SpeedRegularisationStep(Generic[VectorType]):
    """Blends magnitude toward robust consensus magnitude."""

    influence: FloatLike = 0.5  # eta_0

    def __post_init__(self) -> None:
        assert 0 <= self.influence <= 1

    def apply(
        self, y: VectorType, *, ctx: PseudoLabelContext[VectorType, DimState, DimAction]
    ) -> VectorType:
        eta = npDType(self.influence * (1.0 - ctx.trust))
        target_norm = (1.0 - eta) * la.norm(y) + eta * ctx.median_strength
        y_next = rescale_norm(y, target_norm=target_norm)
        return type(y)(y_next)


@dataclass(slots=True, kw_only=True)
class TemporalSmoother(Generic[VectorType]):
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


# VectorType :: State / Action
# CollectionType :: StatesCollection / ActionsCollection


@dataclass(frozen=True)
class VectorMode(
    RuntimeGeneric[CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction]
):
    """Encapsulates operations for state or action processing."""

    # Field access
    vector_from_sample: Callable[[Sample[DimState, DimAction]], VectorType]
    anchor_from_stats: Callable[[RobustStatistics[DimState, DimAction]], VectorType]
    strength_from_token: Callable[[RibbonToken[DimState, DimAction]], MetricValue]

    # Construction
    wrap: Callable[[Any], VectorType]
    make_collection: Callable[
        [Demonstrations[NumDemos, NumPoints, DimState, DimAction]], CollectionType
    ]
    set_item: Callable[[CollectionType, DemoIndex, TimeIndex, VectorType], None]
    get_item: Callable[[CollectionType, DemoIndex, TimeIndex], VectorType]

    # Correction behaviour
    attenuation_requires_state_tangent: bool


def action_mode() -> VectorMode[
    ActionsCollection[NumDemos, NumPoints, DimAction],
    Action[DimAction],
    NumDemos,
    NumPoints,
    DimState,
    DimAction,
]:
    """`VectorMode` configuration for actions."""
    return VectorMode(
        vector_from_sample=lambda sample: sample.action,
        anchor_from_stats=lambda stats: stats.median_action,
        strength_from_token=lambda token: token.median_action_strength,
        wrap=Action[DimAction],
        make_collection=lambda demos: ActionsCollection[
            NumDemos, NumPoints, DimAction
        ].zeros_like(demos),
        set_item=lambda col, i, t, v: col[i].__setitem__(t, v),
        get_item=lambda col, i, t: col[i][t],
        attenuation_requires_state_tangent=True,
    )


def state_mode() -> VectorMode[
    StatesCollection[NumDemos, NumPoints, DimState],
    State[DimState],
    NumDemos,
    NumPoints,
    DimState,
    DimAction,
]:
    """`VectorMode` configuration for states."""
    return VectorMode(
        vector_from_sample=lambda sample: sample.state,
        anchor_from_stats=lambda stats: stats.median_state,
        strength_from_token=lambda token: token.median_state_norm,
        wrap=State[DimState],
        make_collection=lambda demos: StatesCollection[
            NumDemos, NumPoints, DimState
        ].zeros_like(demos),
        set_item=lambda col, i, t, v: col[i].__setitem__(t, v),
        get_item=lambda col, i, t: col[i][t],
        attenuation_requires_state_tangent=False,
    )


@dataclass(kw_only=True)
class PseudoLabelParams(Generic[VectorType]):
    pipeline: RefinementPipeline[VectorType] = field(
        default_factory=RefinementPipeline[VectorType]
    )
    smoother: TemporalSmoother[VectorType] = field(
        default_factory=TemporalSmoother[VectorType]
    )


@dataclass
class PseudoLabelComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Generates corrected pseudo-labels using trust and bin statistics."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]

    def _compute_labels(
        self,
        trust_values: TrustValuesCollection[NumDemos, NumPoints],
        mode: VectorMode[
            CollectionType, VectorType, NumDemos, NumPoints, DimState, DimAction
        ],
        params: PseudoLabelParams[VectorType],
    ) -> CollectionType:
        pre_smooth = mode.make_collection(self.demonstrations)  # y^{(3)} for all (i, t)
        for bin in self.bins:
            token = bin.ribbon_token
            has_state_tangent = token.state_tangent is not None

            tangent = (
                token.state_tangent
                if token.state_tangent is not None
                else token.action_tangent
            )
            unit_tangent = mode.wrap(normalise(tangent, method="NORM"))

            apply_sideways_attenuation = (
                not mode.attenuation_requires_state_tangent
            ) or has_state_tangent

            for j in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=j
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
        action_params: PseudoLabelParams[Action[DimAction]],
        state_params: PseudoLabelParams[State[DimState]] | None = None,
    ) -> PseudoLabels[NumDemos, NumPoints, DimState, DimAction]:
        """Produces final pseudo-labels for actions and optionally states."""
        actions = self._compute_labels(
            action_trust_values, action_mode(), action_params
        )
        states = (
            self._compute_labels(state_trust_values, state_mode(), state_params)
            if state_trust_values is not None and state_params is not None
            else None
        )
        return PseudoLabels(actions=actions, states=states)


# ── PACER ─────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PACERConfig(RuntimeGeneric[NumBins, DimState, DimAction]):
    phase_pipeline_config: PhasePipelineConfig = field(
        default_factory=PhasePipelineConfig
    )
    n_bins: NumBins = cast(NumBins, 96)  # B
    action_trust_value_params: TrustValueParams = field(
        default_factory=TrustValueParams
    )
    action_pseudo_label_params: PseudoLabelParams[Action[DimAction]] = field(
        default_factory=PseudoLabelParams[Action[DimAction]]
    )
    use_state_labels: bool = False
    state_trust_value_params: TrustValueParams = field(default_factory=TrustValueParams)
    state_pseudo_label_params: PseudoLabelParams[State[DimState]] = field(
        default_factory=PseudoLabelParams[State[DimState]]
    )


@dataclass(kw_only=True)
class PACERResult(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    phases: PhasesCollection[NumDemos, NumPoints]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    action_trust_values: TrustValuesCollection[NumDemos, NumPoints]
    state_trust_values: TrustValuesCollection[NumDemos, NumPoints] | None
    pseudo_labels: PseudoLabels[NumDemos, NumPoints, DimState, DimAction]


@dataclass
class PACER(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    config: PACERConfig[NumBins, DimState, DimAction] = field(
        default_factory=PACERConfig[NumBins, DimState, DimAction]
    )

    def run(self) -> PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        phases = PhasePipeline(
            self.demonstrations, config=self.config.phase_pipeline_config
        ).run()
        bins = Binner(
            self.demonstrations, phases, n_bins=self.config.n_bins
        ).make_bins()
        bins = RibbonTokenConsolidator(bins).consolidate_ribbon_tokens()
        action_trust_values = TrustValueComputer(
            self.demonstrations, bins, choice="Action"
        ).compute_trust_values(params=self.config.action_trust_value_params)
        state_trust_values = None
        if self.config.use_state_labels:
            state_trust_values = TrustValueComputer(
                self.demonstrations, bins, choice="State"
            ).compute_trust_values(params=self.config.state_trust_value_params)
        pseudo_labels = PseudoLabelComputer(
            self.demonstrations, bins
        ).compute_pseudo_labels(
            action_trust_values,
            state_trust_values,
            action_params=self.config.action_pseudo_label_params,
            state_params=self.config.state_pseudo_label_params,
        )
        return PACERResult(
            phases=phases,
            bins=bins,
            action_trust_values=action_trust_values,
            state_trust_values=state_trust_values,
            pseudo_labels=pseudo_labels,
        )


## ─────────────────────────────────────────────────────────────────────────────
