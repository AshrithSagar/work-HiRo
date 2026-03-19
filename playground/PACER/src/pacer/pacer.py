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
from typing import Any, Self, TypeAlias, cast

import numpy as np
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
    Vector,
    npDType,
)
from pacer.utils import EPS, MAD_SCALE, SEED, median, normalise, set_seed

## ── PACER ────────────────────────────────────────────────────────────────────

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

# ──────────────────────────────────────────────────────────────────────────────

ZScore: TypeAlias = npDType  # z_{i, t}


class ZScores(TypedList[NumPoints, ZScore]):
    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.time_indices.length
        return cls.full(T_i, ZScore(0))


class ZScoresCollection(TypedDict[NumDemos, DemoIndex, ZScores[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demonstrations.demo_indices,
            lambda i: ZScores[NumPoints].zeros_like(demonstrations[i]),
        )


# ──────────────────────────────────────────────────────────────────────────────

TrustValue: TypeAlias = npDType  # w_{i, t}


class TrustValues(TypedList[NumPoints, TrustValue]):
    @classmethod
    def zeros_like(
        cls, demonstration: Demonstration[NumPoints, DimState, DimAction]
    ) -> Self:
        T_i = demonstration.time_indices.length
        return cls.full(T_i, TrustValue(0))


class TrustValuesCollection(TypedDict[NumDemos, DemoIndex, TrustValues[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demonstrations.demo_indices,
            lambda i: TrustValues[NumPoints].zeros_like(demonstrations[i]),
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
        state_change_norms = [
            la.norm(states[t + 1] - states[t]) for t in range(len(states) - 1)
        ]

        median_action = Action[DimAction](median(actions, axis=0))
        median_state = State[DimState](median(states, axis=0))
        median_action_strength = npDType(median(action_norms, axis=0))
        median_state_change = npDType(median(state_change_norms, axis=0))

        return cls(
            median_action=median_action,
            median_state=median_state,
            median_action_strength=median_action_strength,
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

    def compute_z_scores(self) -> ZScoresCollection[NumDemos, NumPoints]:  # (N x T_)
        z_scores = ZScoresCollection[NumDemos, NumPoints].zeros_like(
            self.demonstrations
        )  # (N x T_)

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=i
                )
                loo_median_action = loo_stats.median_action  # alpha_a^{(-i)}[b]

                loo_residuals = [
                    Residual(la.norm(action - loo_median_action))
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
                    self_residual = Residual(
                        la.norm(sample.action - loo_median_action)
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


@dataclass
class PACER(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    trust_values: TrustValuesCollection[NumDemos, NumPoints]
    seed: int = field(default=SEED, kw_only=True)
    ##
    pseudo_labels: ActionsCollection[NumDemos, NumPoints, DimAction] = field(init=False)

    def compute_pseudo_labels(
        self,
        trust_values: TrustValuesCollection[NumDemos, NumPoints],
        *,
        debias_weight: npDType | float,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> ActionsCollection[NumDemos, NumPoints, DimAction]:  # (N x T_)
        pseudo_labels = ActionsCollection[NumDemos, NumPoints, DimAction].zeros_like(
            self.demonstrations
        )  # [[ystar_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        _labels = ActionsCollection[NumDemos, NumPoints, DimAction].zeros_like(
            self.demonstrations
        )  # [[y^{(3)}_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        rho_0 = sideways_attenuation_shrinkage
        assert 0 <= rho_0 <= 1
        eta_0 = speed_regularisation_influence
        assert 0 <= eta_0 <= 1
        kappa = temporal_smoothing_weight

        for bin in self.bins:
            # Alignment with ribbon tangent
            token = bin.ribbon_token  # z_b
            tangent = (
                token.state_tangent
                if token.state_tangent is not None
                else token.action_tangent
            )
            unit_tangent = Vector[int](normalise(tangent, method="NORM"))  # t_{dir}[b]

            for j in self.demonstrations.demo_indices:
                loo_stats = RobustStatistics[DimState, DimAction].for_bin(
                    bin, LOO_demo_index=j
                )
                bin_median_action = loo_stats.median_action  # alpha_a^{(-j)}[b]

                demo_samples = bin.samples_collection[j]
                for t, sample in demo_samples.items():
                    w = trust_values[j][t]  # w_{i, t}

                    # Debiasing towards the anchor
                    gamma = 1 - debias_weight * (1 - w)  # gamma_{i, t}
                    assert 0 <= gamma <= 1
                    y1 = Action[DimAction](
                        gamma * sample.action + (1 - gamma) * bin_median_action
                    )  # y^{(1)}_{i, t}

                    # Sideways attentuation
                    y1_pll = Action[DimAction](np.dot(y1, unit_tangent) * unit_tangent)
                    y1_perp = Action[DimAction](y1 - y1_pll)
                    has_state_tangent = token.state_tangent is not None
                    rho = rho_0 * (1 - w) if has_state_tangent else npDType(0)
                    y2 = Action[DimAction](
                        y1_pll + (1 - rho) * y1_perp
                    )  # y^{(2)}_{i, t}

                    # Speed regularisation
                    eta = eta_0 * (1 - w)  # eta_{i, t}
                    beta_a = bin.ribbon_token.median_action_strength
                    s = (1 - eta) * la.norm(y2) + eta * beta_a  # s_{i, t}
                    y3 = Action[DimAction](
                        s * (y2 / (la.norm(y2) + EPS))
                    )  # y^{(3)}_{i, t}

                    _labels[j][t] = y3

        # Temporal smoothing
        for i in self.demonstrations.demo_indices:
            ystar_prev: Action[DimAction] | None = None
            for t in self.demonstrations.demos[i].time_indices:
                y3 = _labels[i][t]
                if ystar_prev is None:  # t = 0
                    ystar_prev = y3
                ystar = Action[DimAction]((1 - kappa) * y3 + kappa * ystar_prev)
                pseudo_labels[i][t] = ystar
                ystar_prev = ystar

        return pseudo_labels

    def prepare(
        self,
        *,
        debias_weight: npDType | float = 0.5,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> None:
        set_seed(self.seed)
        self.pseudo_labels = self.compute_pseudo_labels(
            self.trust_values,
            debias_weight=debias_weight,
            sideways_attenuation_shrinkage=sideways_attenuation_shrinkage,
            speed_regularisation_influence=speed_regularisation_influence,
            temporal_smoothing_weight=temporal_smoothing_weight,
        )


## ─────────────────────────────────────────────────────────────────────────────
