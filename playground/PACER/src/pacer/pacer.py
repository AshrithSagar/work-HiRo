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

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Self, TypeAlias, TypeVar, cast

import numpy as np
import numpy.linalg as la
from typingkit.core import RuntimeGeneric, TypedDict, TypedList
from typingkit.numpy._typed.helpers import Dim1

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
from pacer.typings import (
    BinIndex,
    DemoIndex,
    DimAction,
    DimState,
    NumBins,
    NumDemos,
    NumPoints,
    SampleIndex,
    TimeIndex,
    Vector,
    npDType,
)
from pacer.utils import EPS, MAD_SCALE, median, normalise

## ── PACER ────────────────────────────────────────────────────────────────────

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

# ──────────────────────────────────────────────────────────────────────────────

ZScore: TypeAlias = npDType  # z_{i, t}


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


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RobustStatistics(RuntimeGeneric[DimState, DimAction]):
    """Robust consensus statistics for a set of samples."""

    ## Stable anchors
    median_action: Action[DimAction]  # alpha_a[b] = median{ a_{i, t} : (i, t) \in I_b }
    median_state: State[DimState]  # alpha_s[b] = median{ x_{i, t} : (i, t) \in I_b }

    ## Pace
    median_action_strength: npDType
    # beta_a[b] = median{ ||a_{i, t}|| : (i, t) \in I_b }
    # Captures strength of actions
    median_state_change: npDType
    # beta_s[b] = median{ ||xdot_{i, t}|| : (i, t) \in I_b }
    # Captures typical rate of state change

    median_state_norm: npDType

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
        median_action_strength = npDType(median(action_norms, axis=0))
        median_state_norm = npDType(median(state_norms, axis=0))
        median_state_change = npDType(median(state_change_norms, axis=0))

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
    median_action_strength: npDType  # beta_a[b]
    median_state: State[DimState]  # alpha_s[b]
    median_state_change: npDType  # beta_s[b]

    median_state_norm: npDType

    ## Local task dynamics
    action_tangent: Action[DimAction] = field(init=False)
    # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = field(init=False)
    # t_s[b] <- diff{ alpha_s[b] }

    MAD_action_residual: Residual  # Median Absolute Deviation of action residuals


@dataclass(kw_only=True)
class Bin(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
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


@dataclass
class Binner(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    phases: PhasesCollection[NumDemos, NumPoints]
    n_bins: NumBins = field(default=cast(NumBins, 96), kw_only=True)  # B

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


@dataclass
class TrustValueComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
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
        self,
        *,
        tukey_cutoff: npDType | float,  # c
        min_trust: npDType | float,  # w_min
    ) -> TrustValuesCollection[NumDemos, NumPoints]:  # (N x T_)
        assert 3 <= tukey_cutoff <= 5
        trust_values = TrustValuesCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)
        z_scores = self.compute_z_scores()
        for i, scores in z_scores.items():
            for t, z_score in enumerate(scores):
                if z_score <= tukey_cutoff:
                    trust_value = (1 - (z_score / tukey_cutoff) ** 2) ** 2
                else:
                    trust_value = TrustValue(0)
                if trust_value < min_trust:
                    trust_value = TrustValue(min_trust)
                trust_values[i][t] = trust_value
        return trust_values  # [[w_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}


@dataclass(kw_only=True)
class PseudoLabels(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    actions: ActionsCollection[NumDemos, NumPoints, DimAction]
    states: StatesCollection[NumDemos, NumPoints, DimState] | None = None


_VecT = TypeVar("_VecT", bound=Vector, default=Vector)  # State / Action
_Coll = TypeVar("_Coll")  # StatesCollection / ActionsCollection


@dataclass(frozen=True)
class VectorMode(
    RuntimeGeneric[_Coll, _VecT, NumDemos, NumPoints, DimState, DimAction]
):
    # Field access
    vec_from_sample: Callable[[Sample[DimState, DimAction]], _VecT]
    anchor_from_stats: Callable[[RobustStatistics[DimState, DimAction]], _VecT]
    strength_from_token: Callable[[RibbonToken[DimState, DimAction]], npDType]

    # Construction
    wrap: Callable[[Any], _VecT]
    make_collection: Callable[
        [Demonstrations[NumDemos, NumPoints, DimState, DimAction]], _Coll
    ]
    set_item: Callable[[_Coll, DemoIndex, TimeIndex, _VecT], None]
    get_item: Callable[[_Coll, DemoIndex, TimeIndex], _VecT]

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
    return VectorMode(
        vec_from_sample=lambda sample: sample.action,
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
    return VectorMode(
        vec_from_sample=lambda sample: sample.state,
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
class PseudoLabelParams:
    debias_weight: npDType | float = 0.5  # lambda_{debias}
    sideways_attenuation_shrinkage: npDType | float = 0.5  # rho_0
    speed_regularisation_influence: npDType | float = 0.5  # eta_0
    temporal_smoothing_weight: npDType | float = 0.0  # kappa


@dataclass
class PseudoLabelComputer(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]

    def _apply_correction(
        self,
        v: Vector[Dim1],
        *,
        params: PseudoLabelParams,
        anchor: Vector[Dim1],
        unit_tangent: Vector[Dim1],
        median_strength: npDType,
        trust: npDType,
        apply_sideways_attenuation: bool,
    ) -> Vector[Dim1]:
        # Debiasing towards the anchor
        gamma = 1 - params.debias_weight * (1 - trust)  # gamma_{i, t}
        assert 0 <= gamma <= 1
        y1 = gamma * v + (1 - gamma) * anchor  # y^{(1)}_{i, t}

        # Sideways attentuation
        y1_pll = np.dot(y1, unit_tangent) * unit_tangent
        y1_perp = y1 - y1_pll
        rho = (
            params.sideways_attenuation_shrinkage * (1 - trust)
            if apply_sideways_attenuation
            else npDType(0)
        )
        y2 = y1_pll + (1 - rho) * y1_perp  # y^{(2)}_{i, t}

        # Speed regularisation
        eta = params.speed_regularisation_influence * (1 - trust)  # eta_{i, t}
        s = (1 - eta) * la.norm(y2) + eta * median_strength  # s_{i, t}
        y3 = s * (y2 / (la.norm(y2) + EPS))  # y^{(3)}_{i, t}

        return Vector[Dim1](y3)

    def _compute_labels(
        self,
        trust_values: TrustValuesCollection[NumDemos, NumPoints],
        mode: VectorMode[_Coll, _VecT, NumDemos, NumPoints, DimState, DimAction],
        params: PseudoLabelParams,
    ) -> _Coll:
        rho_0 = npDType(params.sideways_attenuation_shrinkage)
        assert 0 <= rho_0 <= 1
        eta_0 = npDType(params.speed_regularisation_influence)
        assert 0 <= eta_0 <= 1
        kappa = npDType(params.temporal_smoothing_weight)

        pre_smooth = mode.make_collection(self.demonstrations)  # y^{(3)} for all (i, t)
        smoothed = mode.make_collection(self.demonstrations)  # y* for all (i, t)

        for bin in self.bins:
            token = bin.ribbon_token
            has_state_tangent = token.state_tangent is not None

            tangent = (
                token.state_tangent
                if token.state_tangent is not None
                else token.action_tangent
            )
            unit_tangent = Vector[int](normalise(tangent, method="NORM"))

            apply_attenuation = (
                not mode.attenuation_requires_state_tangent
            ) or has_state_tangent

            for j in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=j
                )
                anchor = mode.anchor_from_stats(loo_stats)
                median_strength = mode.strength_from_token(token)

                for t, sample in bin.samples_collection[j].items():
                    w = trust_values[j][t]
                    y3 = self._apply_correction(
                        mode.vec_from_sample(sample),
                        params=params,
                        anchor=anchor,
                        unit_tangent=unit_tangent,
                        median_strength=median_strength,
                        trust=w,
                        apply_sideways_attenuation=apply_attenuation,
                    )
                    mode.set_item(pre_smooth, j, t, mode.wrap(y3))

        # Temporal smoothing
        for i in self.demonstrations.demo_indices:
            prev = None
            for t in self.demonstrations[i].time_indices:
                y3 = mode.get_item(pre_smooth, i, t)
                prev = (
                    y3 if prev is None else mode.wrap((1 - kappa) * y3 + kappa * prev)
                )
                mode.set_item(smoothed, i, t, prev)

        return smoothed

    def compute_pseudo_labels(
        self,
        action_trust_values: TrustValuesCollection[NumDemos, NumPoints],
        state_trust_values: TrustValuesCollection[NumDemos, NumPoints] | None = None,
        *,
        action_params: PseudoLabelParams,
        state_params: PseudoLabelParams | None = None,
    ) -> PseudoLabels[NumDemos, NumPoints, DimState, DimAction]:
        actions = self._compute_labels(
            action_trust_values, action_mode(), action_params
        )
        states = (
            self._compute_labels(state_trust_values, state_mode(), state_params)
            if state_trust_values is not None and state_params is not None
            else None
        )
        return PseudoLabels(actions=actions, states=states)


## ─────────────────────────────────────────────────────────────────────────────
