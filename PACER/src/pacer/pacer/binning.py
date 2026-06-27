"""
Binning
=======
"""
# src/pacer/pacer/binning.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Self, cast

import numpy.linalg as la
from typingkit.core import RuntimeGeneric, TypedList

from pacer.base import (
    Action,
    Actions,
    Demonstrations,
    Sample,
    Samples,
    SamplesCollection,
    State,
    StateActionPair,
    StateActionPairs,
    States,
)
from pacer.pacer.base import MetricValue, Residual
from pacer.pacer.consensus import ConsensusConfig
from pacer.phase import Phase, PhasesCollection
from pacer.typings import (
    BinIndex,
    DemoIndex,
    DimAction,
    DimState,
    NumBins,
    NumDemos,
    NumPoints,
    SampleIndex,
)

## ── Binning ──────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class ConsensusStatistics(RuntimeGeneric[DimState, DimAction]):
    """Consensus statistics for a set of samples."""

    ## Stable anchors
    action_anchor: Action[DimAction]  # alpha_a[b]
    state_anchor: State[DimState]  # alpha_s[b]

    ## Pace
    action_strength: MetricValue  # beta_a[b]
    # Captures strength of actions
    state_change: MetricValue  # beta_s[b]
    # Captures typical rate of state change

    state_norm: MetricValue

    ## Local task dynamics
    # NOTE: `action_tangent` and `state_tangent` are not stored here,
    # but instead, computed and stored in RibbonToken.

    @classmethod
    def for_bin(
        cls,
        bin: Bin[NumDemos, NumPoints, DimState, DimAction],
        *,
        consensus_config: ConsensusConfig,
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

        return cls(
            action_anchor=consensus_config.vector_estimator.compute_action(actions),
            state_anchor=consensus_config.vector_estimator.compute_state(states),
            action_strength=consensus_config.scalar_estimator.compute(action_norms),
            state_norm=consensus_config.scalar_estimator.compute(state_norms),
            state_change=consensus_config.scalar_estimator.compute(state_change_norms),
        )


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RibbonToken(RuntimeGeneric[DimState, DimAction]):  # z_b
    """
    Robust structured descriptor for a single phase bin `b`.\\
    Encodes both consensus behaviour and degree of variability present at phase `b`.
    """

    action_anchor: Action[DimAction]  # alpha_a[b]
    action_strength: MetricValue  # beta_a[b]
    state_anchor: State[DimState]  # alpha_s[b]
    state_change: MetricValue  # beta_s[b]

    state_norm: MetricValue

    ## Local task dynamics
    action_tangent: Action[DimAction] = field(init=False)
    # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = field(init=False)
    # t_s[b] <- diff{ alpha_s[b] }

    action_residual_scale: Residual  # (Median Absolute Deviation of action residuals)


# ──────────────────────────────────────────────────────────────────────────────


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


class Bins(TypedList[NumBins, Bin[NumDemos, NumPoints, DimState, DimAction]]):
    """Ordered collection of phase bins."""

    @property
    def consensus_trajectory(self) -> StateActionPairs[NumBins, DimState, DimAction]:
        return StateActionPairs[NumBins, DimState, DimAction](
            StateActionPair(
                state=bin.ribbon_token.state_anchor,
                action=bin.ribbon_token.action_anchor,
            )
            for bin in self
        )


# ──────────────────────────────────────────────────────────────────────────────


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


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RibbonTokenConsolidator(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """Computes ribbon tokens from binned samples."""

    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)

    def consolidate_ribbon_tokens(
        self,
    ) -> Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        bin_action_anchors = Actions[NumBins, DimAction]()
        bin_state_anchors = States[NumBins, DimState]()

        for bin in self.bins:
            stats = ConsensusStatistics[DimState, DimAction].for_bin(
                bin, consensus_config=self.consensus_config
            )

            bin_action_anchor = stats.action_anchor  # alpha_a[b]
            bin_action_residuals = list[Residual]()
            for action in bin.actions():
                residual = Residual(la.norm(action - bin_action_anchor))  # r_{i, t}
                bin_action_residuals.append(residual)
            action_residual_scale: Residual = (
                self.consensus_config.residual_scale_estimator.compute(
                    bin_action_residuals
                )
            )

            bin.ribbon_token = RibbonToken(
                action_anchor=stats.action_anchor,
                action_strength=stats.action_strength,
                state_anchor=stats.state_anchor,
                state_change=stats.state_change,
                state_norm=stats.state_norm,
                action_residual_scale=action_residual_scale,
            )
            bin_action_anchors.append(stats.action_anchor)
            bin_state_anchors.append(stats.state_anchor)

        tangent_estimator = self.consensus_config.tangent_estimator
        action_tangents = tangent_estimator.compute(bin_action_anchors)
        state_tangents = tangent_estimator.compute(bin_state_anchors)
        for bin in self.bins:
            bin.ribbon_token.action_tangent = action_tangents[bin.index]
            bin.ribbon_token.state_tangent = state_tangents[bin.index]

        return self.bins


## ─────────────────────────────────────────────────────────────────────────────
